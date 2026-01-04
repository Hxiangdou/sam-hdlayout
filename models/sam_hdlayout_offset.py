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
import numpy as np

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
        
        prior_prob = 0.05  # 初始时只有 5% 的概率是前景
        bias_value = -np.log((1 - prior_prob) / prior_prob)

        torch.nn.init.constant_(self.block_class.bias, bias_value)
        torch.nn.init.constant_(self.line_class.bias, bias_value)
        torch.nn.init.constant_(self.char_class.bias, bias_value)

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
        B = x.shape[0]
        img_embeddings = self.visual_encoder(x)
        memory = img_embeddings.flatten(2).permute(0, 2, 1)
        
        sparse_embeddings_list = []
        for i in range(B):
            se, _ = self.prompt_encoder(
                points=(prompts[i].get('points'), prompts[i].get('labels')) if (prompts[i].get('points') is not None) else None,
                boxes=prompts[i].get('boxes'), masks=None
            )
            sparse_embeddings_list.append(se.mean(dim=1)) # 取平均
        sparse_embeddings = torch.stack(sparse_embeddings_list) # [B, 1, 256]
        # prompt_enbeddings = self.prompt_fusion(sparse_embeddings)
        
        block_q = self.block_queries.repeat(B, 1, 1) + self.prompt_fusion(sparse_embeddings)

        # 1. Block Level (预测绝对坐标 0-1)
        hs_block = self.block_decoder(tgt=block_q.transpose(0, 1), memory=memory.transpose(0, 1))
        out_block = hs_block[-1].transpose(0, 1)
        # 使用 sigmoid 确保 block 在图像内
        block_bbox = self.block_reg(out_block).sigmoid() 
        
        # 2. Line Level (预测相对于 Block 的偏移)
        line_q = (out_block + self.block_pe(block_bbox)).unsqueeze(2) + \
                 self.line_sub_embed.weight.unsqueeze(0).unsqueeze(0)
        line_q = line_q.flatten(1, 2)
        
        # 预测相对偏移 [B, N_line, 4]，使用 sigmoid 限制在 [0, 1]
        line_offsets = self.line_reg(line_q).sigmoid() 
        
        # 核心：将 line 坐标投影到所属 block 内部
        # 找到每个 line 对应的 parent block
        # line_q 的顺序是 [block0_line0, block0_line1, ..., block1_line0, ...]
        parent_blocks = block_bbox.repeat_interleave(self.lines_per_block, dim=1)
        
        b_cx, b_cy, b_w, b_h = parent_blocks.unbind(-1)
        l_dx, l_dy, l_dw, l_dh = line_offsets.unbind(-1)
        
        # 计算 line 的绝对中心和宽高
        # (l_dx - 0.5) 让行可以在 block 内部中心上下左右移动
        line_cx = b_cx + (l_dx - 0.5) * b_w
        line_cy = b_cy + (l_dy - 0.5) * b_h
        line_w = l_dw * b_w
        line_h = l_dh * b_h
        line_bbox = torch.stack([line_cx, line_cy, line_w, line_h], dim=-1)
        
        # 3. Char Level (预测相对于 Line 的偏移)
        char_q = (line_q + self.line_pe(line_bbox)).unsqueeze(2) + \
                 self.char_sub_embed.weight.unsqueeze(0).unsqueeze(0)
        char_q = char_q.flatten(1, 2)
        
        # 预测 8 个点的相对偏移 [B, N_char, 16]
        char_offsets = self.char_reg(char_q).sigmoid()
        
        # 找到每个 char 对应的 parent line
        parent_lines = line_bbox.repeat_interleave(self.chars_per_line, dim=1)
        l_cx, l_cy, l_w, l_h = parent_lines.unbind(-1)
        
        # 还原 8 个控制点 (x, y)
        char_offsets = char_offsets.view(B, -1, 8, 2)
        char_x = l_cx.unsqueeze(-1) + (char_offsets[..., 0] - 0.5) * l_w.unsqueeze(-1)
        char_y = l_cy.unsqueeze(-1) + (char_offsets[..., 1] - 0.5) * l_h.unsqueeze(-1)
        char_bezier = torch.stack([char_x, char_y], dim=-1).flatten(2)

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
        """
        适配器：将自定义键名转换为 Matcher 所需的标准格式。
        避免了 for 循环导致的 Key 错配问题。
        """
        # 1. 准备 Outputs Proxy
        if layer_name == 'char':
            out_pts = outputs['pred_char'] # [B, N, 16]
            B, N, _ = out_pts.shape
            pts = out_pts.view(B, N, 8, 2)
            min_vals, _ = pts.min(dim=2)
            max_vals, _ = pts.max(dim=2)
            
            # Bezier -> BBox for Matching
            x1, y1 = min_vals[..., 0], min_vals[..., 1]
            x2, y2 = max_vals[..., 0], max_vals[..., 1]
            w, h = (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)
            cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
            
            proxy_boxes = torch.stack([cx, cy, w, h], dim=-1)
            proxy_logits = outputs['pred_char_logits']
        else:
            proxy_boxes = outputs[f'pred_{layer_name}']
            proxy_logits = outputs[f'pred_{layer_name}_logits']

        # 构造给 Matcher 的字典，必须使用 matcher.py 中识别的键
        # 你的 matcher.py 识别 "pred_block" 和 "pred_line" 键来决定行为
        # 所以我们这里伪造一下
        outputs_proxy = {}
        if layer_name == 'block':
            outputs_proxy['pred_block'] = proxy_boxes
            outputs_proxy['pred_block_logits'] = proxy_logits
        elif layer_name == 'line':
            outputs_proxy['pred_line'] = proxy_boxes
            outputs_proxy['pred_line_logits'] = proxy_logits
        else:
            outputs_proxy['pred_char'] = outputs['pred_char']
            outputs_proxy['pred_char_logits'] = outputs['pred_char_logits']
            
        # 2. 准备 Targets Proxy (Matcher 会自己从 targets 里取 block_bbox 等)
        # 所以直接传原始 targets 即可
        
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


# =============================================================================
# Post Processor & Build
# =============================================================================


from torchvision.ops import nms

class PostProcess(nn.Module):
    def __init__(self, max_select=16, nms_thresh=0.1):
        super().__init__()
        self.max_select = max_select # 每个层级最多选出的数量
        self.nms_thresh = nms_thresh
        self.lines_per_block = 4
        self.chars_per_line = 1
        self.belong_thresh = 0.6 # 归属关系阈值
        
    @torch.no_grad()
    def get_bezier_aabb(self, control_points, sample_num=10):
        """
        内部函数：利用采样点计算三阶 Bezier 曲线的精确 AABB
        control_points: [N, 8, 2]
        """
        N = control_points.shape[0]
        device = control_points.device
        t = torch.linspace(0, 1, sample_num, device=device).view(1, sample_num, 1)
        
        p_top = control_points[:, 0:4, :]
        p_bottom = control_points[:, 4:8, :]
        
        t_inv = 1.0 - t
        bezier_basis = torch.cat([
            t_inv ** 3, 3 * t * (t_inv ** 2), 3 * (t ** 2) * t_inv, t ** 3
        ], dim=-1).view(1, sample_num, 4)
        
        pts_top = torch.matmul(bezier_basis, p_top)
        pts_bottom = torch.matmul(bezier_basis, p_bottom)
        all_pts = torch.cat([pts_top, pts_bottom], dim=1)
        
        # 计算精确 AABB
        aabb = torch.cat([all_pts.min(dim=1)[0], all_pts.max(1)[0]], dim=1)
        return aabb

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        def get_probs(logits):
            probs = logits.sigmoid()
            return probs[..., 0] if probs.shape[-1] > 1 else probs.squeeze(-1)

        b_probs = get_probs(outputs['pred_block_logits'])
        l_probs = get_probs(outputs['pred_line_logits'])
        c_probs = get_probs(outputs['pred_char_logits'])

        results = []
        for i in range(len(target_sizes)):
            img_h, img_w = target_sizes[i]
            scale_box = torch.tensor([img_w, img_h, img_w, img_h], device=b_probs.device)
            scale_pts = target_sizes[i].repeat(8).to(b_probs.device)

            def filter_layer(probs, raw_data, parent_raw=None, expansion_factor=1, is_bezier=False):
                # 1. Top-K 粗选候选 Query
                k = min(self.max_select, probs.shape[0])
                topk_values, topk_indices = torch.topk(probs, k)
                
                keep_mask = topk_values > 0.1 
                if not keep_mask.any():
                    keep_mask[0] = True
                
                select_scores = topk_values[keep_mask]
                select_indices = topk_indices[keep_mask]
                
                # 2. 准备候选数据与对应的精确 AABB
                if is_bezier:
                    # 原始控制点数据 (用于最终返回) [K, 8, 2]
                    ctrl_pts = (raw_data[select_indices] * scale_pts).view(-1, 8, 2)
                    # 内部计算精确 AABB (用于 NMS 和从属判断)
                    boxes = self.get_bezier_aabb(ctrl_pts)
                    # data 此时应保存控制点，以便最终返回
                    data = ctrl_pts 
                else:
                    data = box_cxcywh_to_xyxy(raw_data[select_indices]) * scale_box
                    boxes = data

                # 3. 归属关系判断 (使用子项 AABB 与父项 BBox 的重叠度)
                if parent_raw is not None:
                    p_indices = select_indices // expansion_factor
                    p_boxes = box_cxcywh_to_xyxy(parent_raw[p_indices]) * scale_box
                    
                    ix1, iy1 = torch.max(boxes[:, 0], p_boxes[:, 0]), torch.max(boxes[:, 1], p_boxes[:, 1])
                    ix2, iy2 = torch.min(boxes[:, 2], p_boxes[:, 2]), torch.min(boxes[:, 3], p_boxes[:, 3])
                    
                    inters = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                    child_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    
                    belong_ratio = inters / (child_areas + 1e-6)
                    valid_mask = belong_ratio > self.belong_thresh
                    
                    if not valid_mask.any():
                        valid_mask[belong_ratio.argmax()] = True
                    
                    # 联动更新：同时更新返回数据、AABB 和分数
                    data = data[valid_mask]
                    boxes = boxes[valid_mask]
                    select_scores = select_scores[valid_mask]
                else:
                    # Block 层级处理
                    data[:, [0, 2]] = data[:, [0, 2]].clamp(0, img_w)
                    data[:, [1, 3]] = data[:, [1, 3]].clamp(0, img_h)

                # 4. NMS 去重 (基于 boxes 坐标，但筛选 data 数据)
                if boxes.shape[0] > 0:
                    keep_nms = nms(boxes, select_scores, self.nms_thresh)
                    # 返回经过 NMS 的原始控制点 (如果是 bezier) 或框 (如果是 block/line)
                    final_data = data[keep_nms]
                    # 如果是 Bezier，将其展平回 [N, 16] 格式，方便可视化工具调用
                    if is_bezier:
                        final_data = final_data.flatten(1)
                    return final_data, select_scores[keep_nms]
                
                return data, select_scores

            # 执行过滤逻辑
            res_b, _ = filter_layer(b_probs[i], outputs['pred_block'][i])
            res_l, _ = filter_layer(l_probs[i], outputs['pred_line'][i], 
                                    parent_raw=outputs['pred_block'][i], 
                                    expansion_factor=self.lines_per_block)
            res_c, _ = filter_layer(c_probs[i], outputs['pred_char'][i], 
                                    parent_raw=outputs['pred_line'][i], 
                                    expansion_factor=self.chars_per_line, 
                                    is_bezier=True)

            results.append({
                'block_bbox': [res_b],
                'line_bbox': [res_l],
                'char_bezier': [res_c],
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

    criterion = SetCriterion(1, matcher, weight_dict, losses)
    criterion.to(device)
    postprocessors = {'res': PostProcess()}

    return model, criterion, postprocessors