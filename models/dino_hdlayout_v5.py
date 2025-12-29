"""
DINO-HDLayout v5: Hierarchical Query Expansion & Refinement
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from models.matcher import build_matcher
from models.transformer import build_transformer_decoder
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list, 
                       get_world_size, is_dist_avail_and_initialized)
import utils.boxOps as boxOps
from models.config import Config
from utils.boxOps import box_cxcywh_to_xyxy, generalized_box_iou
import math
from models.dino_hdlayout import Encoder, MLPRegressor

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Modules
# =============================================================================

class SinusoidalBoxPosEmbed(nn.Module):
    """用于将 BBox 坐标 [x1, y1, x2, y2] 编码为特征向量"""
    def __init__(self, embed_dim=256, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, boxes):
        if boxes.size(-1) == 4:
            # 修复：将 //4 改为 //8 以匹配输出维度
            dim_t = torch.arange(self.embed_dim // 8, dtype=torch.float32, device=boxes.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / (self.embed_dim // 8))
            
            pos = boxes.unsqueeze(-1) / dim_t
            pos = torch.stack((pos.sin(), pos.cos()), dim=-1).flatten(-2)
            return pos.flatten(-2) # [..., embed_dim]
        else:
            raise ValueError("SinusoidalBoxPosEmbed expects 4 coordinates input")

# =============================================================================
# Main Model: DINO_HDLayout v5
# =============================================================================

class DINO_HDLayout(nn.Module):
    def __init__(self, cfg: Config, args=None):
        super().__init__()
        self.cfg = cfg
        feat_dim = 768 // 2 # 384
        self.num_classes = cfg.num_classes
        self.aux_loss = args.aux_loss
        
        # --- 1. Backbone ---
        self.shared = Encoder(arch=self.cfg.backbone_name, 
                              pretrained=self.cfg.backbone_pretrained, 
                              pretrained_weights=self.cfg.backbone_pretrained_weight, 
                              patch_size=16, 
                              out_dim=feat_dim,
                              device=args.device)

        # --- 2. Block Level (Root) ---
        self.region_bbox_num = cfg.region_bbox_num
        self.block_queries = nn.Parameter(torch.randn(1, self.region_bbox_num, feat_dim))
        
        # Block Refinement
        self.cross_attn_block = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        self.block_decoder = build_transformer_decoder(args, dec_layers=3, return_intermediate=True)
        
        self.block_reg = MLPRegressor(feat_dim, out_dim=cfg.region_bbox_dim)
        self.block_class = nn.Linear(feat_dim, self.num_classes + 1)
        self.block_pe = SinusoidalBoxPosEmbed(embed_dim=feat_dim)

        # --- 3. Line Level (Query Expansion) ---
        self.lines_per_block = max(1, cfg.line_bbox_num // max(1, cfg.region_bbox_num))
        print(f"[Model] Query Expansion: {self.lines_per_block} lines per block.")
        
        self.line_sub_embed = nn.Embedding(self.lines_per_block, feat_dim)
        
        self.cross_attn_line = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        self.line_decoder = build_transformer_decoder(args, dec_layers=3, return_intermediate=True)
        
        self.line_reg = MLPRegressor(feat_dim, out_dim=cfg.line_bbox_dim)
        self.line_class = nn.Linear(feat_dim, self.num_classes + 1)
        self.line_pe = SinusoidalBoxPosEmbed(embed_dim=feat_dim)

        # --- 4. Char Level (Query Expansion) ---
        self.chars_per_line = max(1, cfg.char_bezier_num // max(1, cfg.line_bbox_num))
        print(f"[Model] Query Expansion: {self.chars_per_line} chars per line.")
        
        self.char_sub_embed = nn.Embedding(self.chars_per_line, feat_dim)
        self.cross_attn_char = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        self.char_reg = MLPRegressor(feat_dim, out_dim=cfg.char_bezier_dim)
        self.char_class = nn.Linear(feat_dim, self.num_classes + 1)
        # position embedding
        self.pos_embed = self.prepare_pos_embed(B=args.batch_size, N_orig=196, N_memory=1024, C=feat_dim) # [B, N_pixels, D]
        
    def _cxcywh_to_x1y1x2y2(self, bbox):
        cx, cy, w, h = bbox.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)

    def prepare_pos_embed(self, B, N_orig=196, N_memory=1444, C=384):
        # B, N, C = memory.shape # N=196, C=384
        orig_size = int(N_orig ** 0.5) # 14
        target_size = int(N_memory ** 0.5) # 38
        
        # reshape 为 [Batch, Channel, H, W] 进行插值
        pos_embed_no_cls = self.shared.get_pos_encoding()[:, 1:, :] # [B, N, D]
        pos_tokens = pos_embed_no_cls.reshape(1, orig_size, orig_size, C).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(target_size, target_size), mode='bicubic', align_corners=False)

        # 展回 [B, N, D]
        pos_embed_reshaped = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        return pos_embed_reshaped.permute(1, 0, 2) # [B, N, D]
    
    def forward(self, x):
        B = x.shape[0]
        
        # 1. Backbone
        x = self.shared.net.prepare_tokens(x)
        features = self.shared(x) # [B, N_tokens, D]
        memory = features[:, 1:, :] 
        memory_dec = memory.transpose(0, 1) # [N_pixels, B, D]

        
        # ================= Block Level =================
        block_q = self.block_queries.repeat(B, 1, 1)
        
        hs_block_coarse, _ = self.cross_attn_block(query=block_q, key=memory, value=memory)
        
        tgt_block = hs_block_coarse.transpose(0, 1)
        query_pos_block = block_q.transpose(0, 1)
        
        hs_block_refined = self.block_decoder(
            tgt=tgt_block, memory=memory_dec, pos=self.pos_embed.repeat(1, B, 1), query_pos=query_pos_block
        ) # [L, N, B, D]
        
        output_block = hs_block_refined[-1].transpose(0, 1) # [B, N_blk, D]
        
        block_bbox = self.block_reg(output_block)
        block_class = self.block_class(output_block)
        
        block_pos = self.block_pe(self._cxcywh_to_x1y1x2y2(block_bbox))
        hs_block_guided = output_block + block_pos

        # ================= Line Level =================
        # Expansion: [B, N_blk, 1, D] + [1, 1, N_sub, D]
        line_q_expanded = hs_block_guided.unsqueeze(2) + \
                          self.line_sub_embed.weight.unsqueeze(0).unsqueeze(0)
        line_q = line_q_expanded.flatten(1, 2) # [B, N_line, D]
        
        hs_line_coarse, _ = self.cross_attn_line(query=line_q, key=memory, value=memory)
        
        tgt_line = hs_line_coarse.transpose(0, 1)
        query_pos_line = line_q.transpose(0, 1)
        
        hs_line_refined = self.line_decoder(
            tgt=tgt_line, memory=memory_dec, pos=self.pos_embed.repeat(1, B, 1), query_pos=query_pos_line
        )
        
        output_line = hs_line_refined[-1].transpose(0, 1) # [B, N_line, D]
        
        line_bbox = self.line_reg(output_line)
        line_class = self.line_class(output_line)
        
        line_pos = self.line_pe(self._cxcywh_to_x1y1x2y2(line_bbox))
        hs_line_guided = output_line + line_pos

        # ================= Char Level =================
        char_q_expanded = hs_line_guided.unsqueeze(2) + \
                          self.char_sub_embed.weight.unsqueeze(0).unsqueeze(0)
        char_q = char_q_expanded.flatten(1, 2) # [B, N_char, D]
        
        hs_char, _ = self.cross_attn_char(query=char_q, key=memory, value=memory)
        
        char_bezier = self.char_reg(hs_char)
        char_class = self.char_class(hs_char)
        
        out = {
            'pred_block': block_bbox,
            'pred_block_logits': block_class,
            'pred_line': line_bbox,
            'pred_line_logits': line_class,
            'pred_char': char_bezier,
            'pred_char_logits': char_class
        }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                hs_block_refined[:-1].transpose(0, 1),
                hs_line_refined[:-1].transpose(0, 1)
            )
            
        return out

    @torch.jit.unused
    def _set_aux_loss(self, block_outputs, line_outputs):
        aux_list = []
        n_layers = min(len(block_outputs), len(line_outputs))
        for i in range(n_layers):
            aux_list.append({
                'pred_block': self.block_reg(block_outputs[i]),
                'pred_block_logits': self.block_class(block_outputs[i]),
                'pred_line': self.line_reg(line_outputs[i]),
                'pred_line_logits': self.line_class(line_outputs[i])
            })
        return aux_list

# =============================================================================
# Set Criterion (Fix for Mismatch Error)
# =============================================================================

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
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

    def loss_labels(self, outputs, targets, indices, num_boxes, loss_key):
        """
        Classification loss (NLL)
        保持原函数的多分类交叉熵逻辑，但采用新的 loss_key 解析风格和返回键名。
        """
        assert loss_key in outputs, f"Loss key {loss_key} not found in outputs"
        src_logits = outputs[loss_key]
        
        # 1. 自动解析 Target Key 和 Layer Name (新风格)
        if 'block' in loss_key: 
            target_key = 'block_bbox_labels'
            layer_name = 'block'
        elif 'line' in loss_key: 
            target_key = 'line_bbox_labels'
            layer_name = 'line'
        else: 
            target_key = 'char_bezier_labels'
            layer_name = 'char'
        
        # 2. 获取匹配的索引
        idx = self._get_src_permutation_idx(indices)
        
        # 3. 提取对应的 GT 标签 (原逻辑)
        target_classes_o = torch.cat([t[target_key][J] for t, (_, J) in zip(targets, indices)])
        
        # 4. 创建目标张量，默认填充背景类 self.num_classes (原逻辑)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # 5. 计算多分类交叉熵 (原逻辑)
        # 注意：F.cross_entropy 在处理多维输入时，要求 shape 为 (N, C, H, W) 或 (N, C, L)
        # 因此需要 transpose(1, 2) 将类别维度 C 换到第二位
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        # 6. 返回结果字典 (确保键名与 weight_dict 匹配)
        return {f'loss_{layer_name}_labels_ce': loss_ce}

   
    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_name):
        bbox_key = f'pred_{layer_name}'
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[bbox_key][idx]
        target_boxes = torch.cat([t[f'{layer_name}_bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
        ))).sum() / num_boxes
        
        return {f'loss_{layer_name}_bbox': loss_bbox, f'loss_{layer_name}_giou': loss_giou}
    
    def loss_char_bezier(self, outputs, targets, indices, num_points, loss):
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_char'][idx]
        target_points = torch.cat([t['char_bezier'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_points = F.l1_loss(src_points, target_points, reduction='sum') / num_points
        return {'loss_char_bezier': loss_points}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'loss_block_bbox': lambda: self.loss_boxes(outputs, targets, indices, num, 'block'),
            'loss_block_labels': lambda: self.loss_labels(outputs, targets, indices, num, 'pred_block_logits'),
            'loss_line_bbox': lambda: self.loss_boxes(outputs, targets, indices, num, 'line'),
            'loss_line_labels': lambda: self.loss_labels(outputs, targets, indices, num, 'pred_line_logits'),
            'loss_char_bezier': lambda: self.loss_char_bezier(outputs, targets, indices, num, 'loss_char_bezier'),
            'loss_char_labels': lambda: self.loss_labels(outputs, targets, indices, num, 'pred_char_logits'),
        }
        return loss_map[loss]()

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # 1. Matching
        indices_block = self.match_wrapper(outputs_without_aux, targets, 'block')
        indices_line = self.match_wrapper(outputs_without_aux, targets, 'line')
        indices_char = self.match_wrapper(outputs_without_aux, targets, 'char')
        
        # === 修复开始 ===
        # 提前获取 device，避免使用 .item() 后的变量获取 device
        device = next(iter(outputs.values())).device
        
        # 计算 Block 归一化因子
        num_block = sum(len(t['block_bbox']) for t in targets)
        num_block = torch.as_tensor([num_block], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_block)
        num_block = torch.clamp(num_block / get_world_size(), min=1).item()
        
        # 计算 Line 归一化因子
        num_line = sum(len(t['line_bbox']) for t in targets)
        # 修复：使用 device 变量，而不是 num_block.device
        num_line = torch.as_tensor([num_line], dtype=torch.float, device=device) 
        if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_line)
        num_line = torch.clamp(num_line / get_world_size(), min=1).item()

        # 计算 Char 归一化因子
        num_char = sum(len(t['char_bezier']) for t in targets)
        # 修复：使用 device 变量
        num_char = torch.as_tensor([num_char], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized(): torch.distributed.all_reduce(num_char)
        num_char = torch.clamp(num_char / get_world_size(), min=1).item()
        # === 修复结束 ===
        
        losses = {}
        for loss in self.losses:
            if 'block' in loss:
                losses.update(self.get_loss(loss, outputs, targets, indices_block, num_block))
            elif 'line' in loss:
                losses.update(self.get_loss(loss, outputs, targets, indices_line, num_line))
            elif 'char' in loss:
                losses.update(self.get_loss(loss, outputs, targets, indices_char, num_char))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    if 'char' in loss: continue
                    if 'block' in loss:
                        indices = self.match_wrapper(aux_outputs, targets, 'block')
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_block)
                    elif 'line' in loss:
                        indices = self.match_wrapper(aux_outputs, targets, 'line')
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_line)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

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
    
    model = DINO_HDLayout(cfg, args)
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