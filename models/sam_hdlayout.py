import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import ImageEncoderViT, PromptEncoder
from models.transformer import build_transformer_decoder
from models.dino_hdlayout import MLPRegressor
from models.dino_hdlayout_v5 import SinusoidalBoxPosEmbed #
from models.config import Config
from models.matcher import build_matcher
from utils.misc import (get_world_size, is_dist_avail_and_initialized)
from utils.boxOps import box_cxcywh_to_xyxy, generalized_box_iou

class SAM_HDLayout(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        self.cfg = cfg
        feat_dim = 256  # SAM 默认维度
        
        # 1. SAM Backbone
        self.visual_encoder = ImageEncoderViT(
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,      # 从 768 改为 1024
            depth=24,            # 从 12 改为 24 (ViT-L 有 24 层)
            num_heads=16,         # 从 12 改为 16
            mlp_ratio=4,
            out_chans=256,       # SAM 最终输出维度固定为 256，这个不需要改
            qkv_bias=True,
            use_rel_pos=True,
            window_size=14,
            # ViT-L 的全局注意力索引通常是 [5, 11, 17, 23]
            global_attn_indexes=[5, 11, 17, 23] 
        )

        # 2. SAM Prompt Encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=feat_dim,
            image_embedding_size=(64, 64), # 1024/16
            input_image_size=(1024, 1024),
            mask_in_chans=16
        )

        # 3. Block Level
        self.block_queries = nn.Parameter(torch.randn(1, cfg.region_bbox_num, feat_dim))
        self.prompt_fusion = nn.Linear(feat_dim, feat_dim) # 交互信号融合
        
        self.block_decoder = build_transformer_decoder(args, dec_layers=3, return_intermediate=True)
        self.block_reg = MLPRegressor(feat_dim, out_dim=cfg.region_bbox_dim)
        self.block_class = nn.Linear(feat_dim, cfg.num_classes + 1)
        self.block_pe = SinusoidalBoxPosEmbed(embed_dim=feat_dim) #

        # 4. Line & Char Level (Query Expansion)
        self.lines_per_block = max(1, cfg.line_bbox_num // cfg.region_bbox_num)
        self.line_sub_embed = nn.Embedding(self.lines_per_block, feat_dim)
        self.line_reg = MLPRegressor(feat_dim, out_dim=cfg.line_bbox_dim)
        self.line_class = nn.Linear(feat_dim, cfg.num_classes + 1)
        self.line_pe = SinusoidalBoxPosEmbed(embed_dim=feat_dim)

        self.chars_per_line = max(1, cfg.char_bezier_num // cfg.line_bbox_num)
        self.char_sub_embed = nn.Embedding(self.chars_per_line, feat_dim)
        self.char_reg = MLPRegressor(feat_dim, out_dim=cfg.char_bezier_dim)
        self.char_class = nn.Linear(feat_dim, cfg.num_classes + 1)

        if args.sam_checkpoint:
            self.load_sam_weights(args.sam_checkpoint)

    def load_sam_weights(self, checkpoint_path):
        print(f"=> Loading SAM weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        state_dict = checkpoint
        
        visual_encoder_dict = {k.replace('image_encoder.', ''): v for k, v in state_dict.items() if k.startswith('image_encoder.')}
        prompt_encoder_dict = {k.replace('prompt_encoder.', ''): v for k, v in state_dict.items() if k.startswith('prompt_encoder.')}
        
        # 加载权重
        msg1 = self.visual_encoder.load_state_dict(visual_encoder_dict, strict=False)
        msg2 = self.prompt_encoder.load_state_dict(prompt_encoder_dict, strict=False)
        
        print(f"Visual Encoder Load Msg: {msg1}")
        print(f"Prompt Encoder Load Msg: {msg2}")
        
    def forward(self, x, prompts=None):
        """
        prompts: 列表，每个元素是字典 {'points':..., 'labels':..., 'boxes':...}
        """
        B = x.shape[0]
        img_embeddings = self.visual_encoder(x) # [B, 256, 64, 64]
        memory = img_embeddings.flatten(2).permute(0, 2, 1) # [B, 4096, 256]
        
        sparse_embeddings_list = []
        for i in range(B):
            se, _ = self.prompt_encoder(
                points=(prompts[i].get('points'), prompts[i].get('labels')) if (prompts[i].get('points') is not None) else None,
                boxes=prompts[i].get('boxes'), masks=None
            )
            sparse_embeddings_list.append(se.mean(dim=1)) # 取平均
        sparse_embeddings = torch.stack(sparse_embeddings_list) # [B, 1, 256]
        block_q = self.block_queries.repeat(B, 1, 1) + self.prompt_fusion(sparse_embeddings)

        # Block Level
        hs_block = self.block_decoder(tgt=block_q.transpose(0, 1), memory=memory.transpose(0, 1))
        out_block = hs_block[-1].transpose(0, 1)
        block_bbox = self.block_reg(out_block)
        
        # Line Level
        line_q = (out_block + self.block_pe(block_bbox)).unsqueeze(2) + \
                 self.line_sub_embed.weight.unsqueeze(0).unsqueeze(0)
        line_q = line_q.flatten(1, 2)

        line_bbox = self.line_reg(line_q)
        
        # Char Level
        char_q = (line_q + self.line_pe(line_bbox)).unsqueeze(2) + \
                 self.char_sub_embed.weight.unsqueeze(0).unsqueeze(0)
        char_q = char_q.flatten(1, 2)
        char_bezier = self.char_reg(char_q)

        return {
            'pred_block': block_bbox, 'pred_block_logits': self.block_class(out_block),
            'pred_line': line_bbox, 'pred_line_logits': self.line_class(line_q),
            'pred_char': char_bezier, 'pred_char_logits': self.char_class(char_q)
        }

# =============================================================================
# Set Criterion (Fix for Mismatch Error)
# =============================================================================
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


class SetCriterion(nn.Module):
    """
    针对单类别前景(0)优化的损失函数类。
    """
    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes # 应该为 1
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        
    def _get_src_permutation_idx(self, indices):
        # 辅助函数：获取匹配后的 batch 索引和 query 索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes, loss_key):
        """
        分类损失：使用 Focal Loss。
        针对前景标签 0，我们只关心 logits 的第一个通道（或者如果是单通道输出）。
        """
        assert loss_key in outputs, f"Key {loss_key} not found"
        src_logits = outputs[loss_key] # [B, N, 2] 或 [B, N, 1]

        layer_name = 'block' if 'block' in loss_key else ('line' if 'line' in loss_key else 'char')
        
        # 构造二分类 Target: 匹配到的 query 为 1，未匹配为 0
        target_classes = torch.zeros_like(src_logits)
        idx = self._get_src_permutation_idx(indices)
        
        # 将匹配位置设为前景 (1.0)
        # 即使模型输出是 2 维(前景, 背景)，我们也只对第一个通道计算二分类 Focal Loss 也是合理的，
        # 或者更通用的做法是只取第一个通道：target_classes[idx, 0] = 1.0
        if src_logits.shape[-1] > 1:
            target_classes[idx[0], idx[1], 0] = 1.0
        else:
            target_classes[idx] = 1.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes, num_boxes)
        return {f'loss_{layer_name}_labels_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_name):
        """ L1 和 GIoU 损失 """
        bbox_key = f'pred_{layer_name}'
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[bbox_key][idx]
        target_boxes = torch.cat([t[f'{layer_name}_bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        
        # GIoU
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
        ))).sum() / num_boxes
        
        return {f'loss_{layer_name}_bbox': loss_bbox, f'loss_{layer_name}_giou': loss_giou}

    def loss_char_bezier(self, outputs, targets, indices, num_points, layer_name):
        """ 字符级 Bezier 曲线点 L1 损失 """
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_char'][idx]
        target_points = torch.cat([t['char_bezier'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_points = F.l1_loss(src_points, target_points, reduction='sum') / num_points
        return {'loss_char_bezier': loss_points}

    def match_wrapper(self, outputs, targets, layer_name):
        """ 适配 Matcher 的输入格式 """
        if layer_name == 'char':
            # 将 Bezier 点转为简单的 AABB 框进行匹配加速
            out_pts = outputs['pred_char']
            B, N, _ = out_pts.shape
            pts = out_pts.view(B, N, 8, 2)
            min_v, _ = pts.min(dim=2); max_v, _ = pts.max(dim=2)
            w_h = (max_v - min_v).clamp(min=1e-6)
            proxy_boxes = torch.cat([min_v + 0.5 * w_h, w_h], dim=-1)
            outputs_proxy = {'pred_char': outputs['pred_char'], 
                             'pred_char_logits': outputs['pred_char_logits'],
                             'pred_logits': outputs['pred_char_logits'], # 兼容内部调用
                             'pred_points': outputs['pred_char']}
        else:
            outputs_proxy = {'pred_logits': outputs[f'pred_{layer_name}_logits'],
                             f'pred_{layer_name}': outputs[f'pred_{layer_name}'],
                             'pred_boxes': outputs[f'pred_{layer_name}']}
        
        return self.matcher(outputs_proxy, targets)

    def forward(self, outputs, targets):
        device = next(iter(outputs.values())).device
        
        # 1. 独立计算每个层级的匹配和归一化因子
        def get_num_boxes(target_key):
            num = sum(len(t[target_key]) for t in targets)
            num = torch.as_tensor([num], dtype=torch.float, device=device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num)
            return torch.clamp(num / get_world_size(), min=1).item()

        n_block = get_num_boxes('block_bbox')
        n_line = get_num_boxes('line_bbox')
        n_char = get_num_boxes('char_bezier')

        # 2. 匹配
        indices_block = self.match_wrapper(outputs, targets, 'block')
        indices_line = self.match_wrapper(outputs, targets, 'line')
        indices_char = self.match_wrapper(outputs, targets, 'char')

        losses = {}
        for loss in self.losses:
            if 'block' in loss:
                idx, num = indices_block, n_block
                if 'labels' in loss:
                    losses.update(self.loss_labels(outputs, targets, idx, num, 'pred_block_logits'))
                else:
                    losses.update(self.loss_boxes(outputs, targets, idx, num, 'block'))
            elif 'line' in loss:
                idx, num = indices_line, n_line
                if 'labels' in loss:
                    losses.update(self.loss_labels(outputs, targets, idx, num, 'pred_line_logits'))
                else:
                    losses.update(self.loss_boxes(outputs, targets, idx, num, 'line'))
            elif 'char' in loss:
                idx, num = indices_char, n_char
                if 'labels' in loss:
                    losses.update(self.loss_labels(outputs, targets, idx, num, 'pred_char_logits'))
                else:
                    losses.update(self.loss_char_bezier(outputs, targets, idx, num, 'char'))

        return losses
# class SetCriterion(nn.Module):
#     def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
#         # self.eos_coef = eos_coef
#         # empty_weight = torch.ones(self.num_classes + 1)
#         # empty_weight[-1] = self.eos_coef
#         # self.register_buffer('empty_weight', empty_weight)
        
#     def match_wrapper(self, outputs, targets, layer_name):
#         """
#         适配器：将自定义键名转换为 Matcher 所需的标准格式。
#         避免了 for 循环导致的 Key 错配问题。
#         """
#         # 1. 准备 Outputs Proxy
#         if layer_name == 'char':
#             out_pts = outputs['pred_char'] # [B, N, 16]
#             B, N, _ = out_pts.shape
#             pts = out_pts.view(B, N, 8, 2)
#             min_vals, _ = pts.min(dim=2)
#             max_vals, _ = pts.max(dim=2)
            
#             # Bezier -> BBox for Matching
#             x1, y1 = min_vals[..., 0], min_vals[..., 1]
#             x2, y2 = max_vals[..., 0], max_vals[..., 1]
#             w, h = (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)
#             cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
            
#             proxy_boxes = torch.stack([cx, cy, w, h], dim=-1)
#             proxy_logits = outputs['pred_char_logits']
#         else:
#             proxy_boxes = outputs[f'pred_{layer_name}']
#             proxy_logits = outputs[f'pred_{layer_name}_logits']

#         # 构造给 Matcher 的字典，必须使用 matcher.py 中识别的键
#         # 你的 matcher.py 识别 "pred_block" 和 "pred_line" 键来决定行为
#         # 所以我们这里伪造一下
#         outputs_proxy = {}
#         if layer_name == 'block':
#             outputs_proxy['pred_block'] = proxy_boxes
#             outputs_proxy['pred_block_logits'] = proxy_logits
#         elif layer_name == 'line':
#             outputs_proxy['pred_line'] = proxy_boxes
#             outputs_proxy['pred_line_logits'] = proxy_logits
#         else:
#             outputs_proxy['pred_char'] = outputs['pred_char']
#             outputs_proxy['pred_char_logits'] = outputs['pred_char_logits']
            
#         # 2. 准备 Targets Proxy (Matcher 会自己从 targets 里取 block_bbox 等)
#         # 所以直接传原始 targets 即可
        
#         return self.matcher(outputs_proxy, targets)

#     def loss_labels(self, outputs, targets, indices, num_boxes, loss_key):
#         """
#         Classification loss (NLL)
#         保持原函数的多分类交叉熵逻辑，但采用新的 loss_key 解析风格和返回键名。
#         """
#         assert loss_key in outputs, f"Loss key {loss_key} not found in outputs"
#         src_logits = outputs[loss_key]
        
#         # 1. 自动解析 Target Key 和 Layer Name (新风格)
#         if 'block' in loss_key: 
#             target_key = 'block_bbox_labels'
#             layer_name = 'block'
#         elif 'line' in loss_key: 
#             target_key = 'line_bbox_labels'
#             layer_name = 'line'
#         else: 
#             target_key = 'char_bezier_labels'
#             layer_name = 'char'
        
#         # 2. 获取匹配的索引
#         idx = self._get_src_permutation_idx(indices)
        
#         # 3. 提取对应的 GT 标签 (原逻辑)
#         target_classes_o = torch.cat([t[target_key][J] for t, (_, J) in zip(targets, indices)])
        
#         # 4. 创建目标张量，默认填充背景类 self.num_classes (原逻辑)
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o

#         # 5. 计算多分类交叉熵 (原逻辑)
#         # 注意：F.cross_entropy 在处理多维输入时，要求 shape 为 (N, C, H, W) 或 (N, C, L)
#         # 因此需要 transpose(1, 2) 将类别维度 C 换到第二位
#         loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

#         # 6. 返回结果字典 (确保键名与 weight_dict 匹配)
#         return {f'loss_{layer_name}_labels_ce': loss_ce}

   
#     def loss_boxes(self, outputs, targets, indices, num_boxes, layer_name):
#         bbox_key = f'pred_{layer_name}'
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs[bbox_key][idx]
#         target_boxes = torch.cat([t[f'{layer_name}_bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)

#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
#         loss_giou = (1 - torch.diag(generalized_box_iou(
#             box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
#         ))).sum() / num_boxes
        
#         return {f'loss_{layer_name}_bbox': loss_bbox, f'loss_{layer_name}_giou': loss_giou}
    
#     def loss_char_bezier(self, outputs, targets, indices, num_points, loss):
#         idx = self._get_src_permutation_idx(indices)
#         src_points = outputs['pred_char'][idx]
#         target_points = torch.cat([t['char_bezier'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#         loss_points = F.l1_loss(src_points, target_points, reduction='sum') / num_points
#         return {'loss_char_bezier': loss_points}

#     def _get_src_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
#         loss_map = {
#             'loss_block_bbox': lambda: self.loss_boxes(outputs, targets, indices, num, 'block'),
#             'loss_block_labels': lambda: self.loss_labels(outputs, targets, indices, num, 'pred_block_logits'),
#             'loss_line_bbox': lambda: self.loss_boxes(outputs, targets, indices, num, 'line'),
#             'loss_line_labels': lambda: self.loss_labels(outputs, targets, indices, num, 'pred_line_logits'),
#             'loss_char_bezier': lambda: self.loss_char_bezier(outputs, targets, indices, num, 'loss_char_bezier'),
#             'loss_char_labels': lambda: self.loss_labels(outputs, targets, indices, num, 'pred_char_logits'),
#         }
#         return loss_map[loss]()

#     def forward(self, outputs, targets):
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
#         # 1. Matching
#         indices_block = self.match_wrapper(outputs_without_aux, targets, 'block')
#         indices_line = self.match_wrapper(outputs_without_aux, targets, 'line')
#         indices_char = self.match_wrapper(outputs_without_aux, targets, 'char')
        
#         # === 修复开始 ===
#         # 提前获取 device，避免使用 .item() 后的变量获取 device
#         device = next(iter(outputs.values())).device
        
#         # 计算 Block 归一化因子
#         num_block = sum(len(t['block_bbox']) for t in targets)
#         num_block = torch.as_tensor([num_block], dtype=torch.float, device=device)
#         if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_block)
#         num_block = torch.clamp(num_block / get_world_size(), min=1).item()
        
#         # 计算 Line 归一化因子
#         num_line = sum(len(t['line_bbox']) for t in targets)
#         # 修复：使用 device 变量，而不是 num_block.device
#         num_line = torch.as_tensor([num_line], dtype=torch.float, device=device) 
#         if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_line)
#         num_line = torch.clamp(num_line / get_world_size(), min=1).item()

#         # 计算 Char 归一化因子
#         num_char = sum(len(t['char_bezier']) for t in targets)
#         # 修复：使用 device 变量
#         num_char = torch.as_tensor([num_char], dtype=torch.float, device=device)
#         if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_char)
#         num_char = torch.clamp(num_char / get_world_size(), min=1).item()
#         # === 修复结束 ===
        
#         losses = {}
#         for loss in self.losses:
#             if 'block' in loss:
#                 losses.update(self.get_loss(loss, outputs, targets, indices_block, num_block))
#             elif 'line' in loss:
#                 losses.update(self.get_loss(loss, outputs, targets, indices_line, num_line))
#             elif 'char' in loss:
#                 losses.update(self.get_loss(loss, outputs, targets, indices_char, num_char))

#         if 'aux_outputs' in outputs:
#             for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 for loss in self.losses:
#                     if 'char' in loss: continue
#                     if 'block' in loss:
#                         indices = self.match_wrapper(aux_outputs, targets, 'block')
#                         l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_block)
#                     elif 'line' in loss:
#                         indices = self.match_wrapper(aux_outputs, targets, 'line')
#                         l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_line)
#                     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#         return losses

# =============================================================================
# Post Processor & Build
# =============================================================================

class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_block = outputs['pred_block']
        out_line = outputs['pred_line']
        out_char = outputs['pred_char']
        
        # out_block_scores = outputs['pred_block_logits'].sigmoid().squeeze(-1)
        # out_line_scores = outputs['pred_line_logits'].sigmoid().squeeze(-1)
        # out_char_scores = outputs['pred_char_logits'].sigmoid().squeeze(-1)
        out_block_scores = F.softmax(outputs['pred_block_logits'], -1)
        out_line_scores = F.softmax(outputs['pred_line_logits'], -1)
        out_char_scores = F.softmax(outputs['pred_char_logits'], -1)
        
        prob_block, label_block = out_block_scores.max(-1)
        prob_line, label_line = out_line_scores.max(-1)
        prob_char, label_char = out_char_scores.max(-1)
        
        results = []
        for i in range(len(target_sizes)):
            # t = 0.4
            
            # keep_block = out_block_scores[i] > t
            # keep_line = out_line_scores[i] > t
            # keep_char = out_char_scores[i] > t
            keep_block = label_block[i] != 1
            keep_line = label_line[i] != 1
            keep_char = label_char[i] != 1
            
            img_h, img_w = target_sizes[i]
            scale = torch.tensor([img_w, img_h, img_w, img_h], device=out_block.device)
            
            block_boxes = box_cxcywh_to_xyxy(out_block[i][keep_block]) * scale
            line_boxes = box_cxcywh_to_xyxy(out_line[i][keep_line]) * scale
            char_beziers = out_char[i][keep_char] * target_sizes[i].repeat(8)
            
            results.append({
                'block_bbox': [block_boxes],
                'line_bbox': [line_boxes],
                'char_bezier': [char_beziers]
            })
        return results



def build(args):
    device = torch.device(args.device)
    cfg = Config()
    cfg.num_classes = 1 
    
    model = SAM_HDLayout(cfg, args)
    matcher = build_matcher(args)
    
    weight_dict = {
        'loss_block_bbox': 5, 'loss_block_giou': 2, 'loss_block_labels_ce': 2,
        'loss_line_bbox': 5, 'loss_line_giou': 2, 'loss_line_labels_ce': 2,
        'loss_char_bezier': 5, 'loss_char_labels_ce': 2,
    }
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(3 - 1): 
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['loss_block_bbox', 'loss_block_labels', 
              'loss_line_bbox', 'loss_line_labels',
              'loss_char_bezier', 'loss_char_labels']

    criterion = SetCriterion(1, matcher, weight_dict, 0.1, losses)
    criterion.to(device)
    postprocessors = {'res': PostProcess()}

    return model, criterion, postprocessors