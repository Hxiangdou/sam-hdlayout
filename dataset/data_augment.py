"""
Robust Data Augmentation for HDLayout (Region/Line/Char)
Supports: Multi-scale Resize, Random Flip, Color Jitter, Normalization, 
          Random Crop, Random Move (Translation)
"""
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import math

# =============================================================================
# Helper Functions
# =============================================================================

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def hflip_box(boxes, width):
    # boxes: [N, 4] (x1, y1, x2, y2)
    boxes = boxes.clone()
    old_x1 = boxes[:, 0].clone()
    old_x2 = boxes[:, 2].clone()
    
    boxes[:, 0] = width - old_x2
    boxes[:, 2] = width - old_x1
    return boxes

def hflip_bezier(beziers, width):
    # beziers: [N, 16]
    beziers = beziers.clone()
    # 1. Flip all x coordinates (even indices)
    beziers[:, 0::2] = width - beziers[:, 0::2]
    
    # 2. Reorder control points
    permutation = [6, 7, 4, 5, 2, 3, 0, 1, 
                   14, 15, 12, 13, 10, 11, 8, 9]
    return beziers[:, permutation]

def crop(image, target, region):
    """
    执行剪裁操作，并处理所有关联的 bbox, bezier 和 labels
    region: [top, left, height, width]
    """
    i, j, h, w = region
    cropped_image = F.crop(image, i, j, h, w)
    
    target = target.copy()
    target["size"] = torch.tensor([h, w])
    
    # --- Process Boxes (Block & Line) ---
    for key in ['block_bbox', 'line_bbox']:
        if key in target and len(target[key]) > 0:
            # 修复：转换为 Tensor 并展平为 [N, 4]
            boxes = torch.as_tensor(target[key]).float()
            boxes = boxes.view(-1, 4) 
            
            # Shift coordinates: box - [left, top, left, top]
            boxes = boxes - torch.tensor([j, i, j, i], device=boxes.device)
            
            # Clamp to new image boundaries
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)
            
            # Filter invalid boxes (width <= 0 or height <= 0)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            target[key] = boxes[keep]
            
            # Synchronize Labels
            label_key = key + '_labels'
            if label_key in target:
                target[label_key] = target[label_key][keep]

    # --- Process Bezier (Char) ---
    if 'char_bezier' in target and len(target['char_bezier']) > 0:
        # 修复：确保转换为 Tensor 并展平为 [N, 16]
        beziers = torch.as_tensor(target['char_bezier']).float()
        beziers = beziers.view(-1, 16)
        
        # Shift coordinates
        shift = torch.tensor([j, i] * 8, device=beziers.device)
        beziers = beziers - shift
        
        # Filter visibility
        pts = beziers.view(-1, 8, 2)
        min_vals = pts.min(dim=1)[0] # [N, 2]
        max_vals = pts.max(dim=1)[0] # [N, 2]
        
        visible = (max_vals[:, 0] > 0) & (max_vals[:, 1] > 0) & \
                  (min_vals[:, 0] < w) & (min_vals[:, 1] < h)
        
        target['char_bezier'] = beziers[visible]
        if 'char_bezier_labels' in target:
            target['char_bezier_labels'] = target['char_bezier_labels'][visible]
            
    return cropped_image, target

# =============================================================================
# Transforms Classes
# =============================================================================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            w, h = img.size
            img = F.hflip(img)
            target = target.copy()
            
            for key in ['block_bbox', 'line_bbox']:
                if key in target and len(target[key]) > 0:
                    # 确保 [N, 4]
                    boxes = torch.as_tensor(target[key]).float().view(-1, 4)
                    target[key] = hflip_box(boxes, w)
            
            if 'char_bezier' in target and len(target['char_bezier']) > 0:
                # 确保 [N, 16]
                beziers = torch.as_tensor(target['char_bezier']).float().view(-1, 16)
                target['char_bezier'] = hflip_bezier(beziers, w)
                
        return img, target

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        
        def get_size(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))
            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            return (oh, ow)

        new_h, new_w = get_size(img.size, size, self.max_size)
        rescaled_image = F.resize(img, (new_h, new_w))
        
        if target is None:
            return rescaled_image, None
        
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, img.size))
        ratio_w, ratio_h = ratios
        
        target = target.copy()
        for key in ['block_bbox', 'line_bbox']:
            if key in target and len(target[key]) > 0:
                # 修复：确保 [N, 4]
                boxes = torch.as_tensor(target[key]).float().view(-1, 4)
                target[key] = boxes * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])

        if 'char_bezier' in target and len(target['char_bezier']) > 0:
            # 修复：确保 [N, 16]
            beziers = torch.as_tensor(target['char_bezier']).float().view(-1, 16)
            target['char_bezier'] = beziers * torch.tensor([ratio_w, ratio_h] * 8)

        target['size'] = torch.tensor([new_h, new_w])
        return rescaled_image, target

class RandomCrop(object):
    """
    随机剪裁图片和目标
    """
    def __init__(self, min_scale=0.8, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, target):
        w, h = img.size
        # 1. 随机选择剪裁尺寸
        scale = random.uniform(self.min_scale, self.max_scale)
        
        # 2. 计算新尺寸，并强制约束
        new_w = min(w, int(w * scale))
        new_h = min(h, int(h * scale))
        
        if w == new_w and h == new_h:
            return img, target
        
        h_diff = max(0, h - new_h)
        w_diff = max(0, w - new_w)
            
        i = random.randint(0, h_diff)
        j = random.randint(0, w_diff)
        
        return crop(img, target, [i, j, new_h, new_w])

class RandomMove(object):
    """
    随机上下左右平移图片 (Translation)
    """
    def __init__(self, max_shift_ratio=0.1, p=0.5):
        self.max_shift_ratio = max_shift_ratio
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            w, h = img.size
            max_dx = int(w * self.max_shift_ratio)
            max_dy = int(h * self.max_shift_ratio)
            
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            
            if dx == 0 and dy == 0:
                return img, target
            
            img = F.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0, fill=255)
            
            target = target.copy()
            
            # Boxes
            for key in ['block_bbox', 'line_bbox']:
                if key in target and len(target[key]) > 0:
                    # 修复：确保转换为 Tensor 并展平为 [N, 4]
                    boxes = torch.as_tensor(target[key]).float()
                    boxes = boxes.view(-1, 4)
                    
                    boxes = boxes + torch.tensor([dx, dy, dx, dy], device=boxes.device)
                    # Clamp
                    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
                    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)
                    # Filter
                    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    target[key] = boxes[keep]
                    if key + '_labels' in target:
                        target[key + '_labels'] = target[key + '_labels'][keep]
            
            # Beziers
            if 'char_bezier' in target and len(target['char_bezier']) > 0:
                # 修复：确保转换为 Tensor 并展平为 [N, 16]
                beziers = torch.as_tensor(target['char_bezier']).float()
                beziers = beziers.view(-1, 16)
                
                beziers = beziers + torch.tensor([dx, dy] * 8, device=beziers.device)
                
                pts = beziers.view(-1, 8, 2)
                min_vals = pts.min(dim=1)[0]
                max_vals = pts.max(dim=1)[0]
                visible = (max_vals[:, 0] > 0) & (max_vals[:, 1] > 0) & \
                          (min_vals[:, 0] < w) & (min_vals[:, 1] < h)
                
                target['char_bezier'] = beziers[visible]
                if 'char_bezier_labels' in target:
                    target['char_bezier_labels'] = target['char_bezier_labels'][visible]
                    
        return img, target

class ColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p
        
    def __call__(self, img, target):
        if random.random() < self.p:
            img = self.transform(img)
        return img, target

class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        
        target = target.copy()
        h, w = image.shape[-2:]
        
        for key in ['block_bbox', 'line_bbox']:
            if key in target and len(target[key]) > 0:
                boxes = target[key]
                # 确保转换为 Tensor 并展平
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.as_tensor(boxes).float()
                boxes = boxes.view(-1, 4)
                
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target[key] = boxes
        
        if 'char_bezier' in target and len(target['char_bezier']) > 0:
            beziers = target['char_bezier']
            if not isinstance(beziers, torch.Tensor):
                beziers = torch.as_tensor(beziers).float()
            beziers = beziers.view(-1, 16)
            
            scale = torch.tensor([w, h] * 8, dtype=torch.float32)
            target['char_bezier'] = beziers / scale
            
        return image, target

# =============================================================================
# Factory Function
# =============================================================================

class RandomSelect(object):
    def __init__(self, t1, t2, p=0.5):
        self.t1 = t1
        self.t2 = t2
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            return self.t1(img, target)
        return self.t2(img, target)

def make_hdlayout_transforms(image_set):
    normalize = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return Compose([
            # 先做色彩抖动
            ColorJitter(p=0.3),
            
            # 随机平移
            RandomMove(max_shift_ratio=0.1, p=0.3),
            
            # 随机翻转
            RandomHorizontalFlip(p=0.5),
            
            # 混合增强
            RandomSelect(
                RandomResize([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], max_size=1333),
                Compose([
                    RandomCrop(min_scale=0.8, max_scale=1.0),
                    RandomResize([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], max_size=1333),
                ])
            ),
            
            normalize,
        ])
    
    if image_set == 'val':
        return Compose([
            RandomResize([800], max_size=1333),
            normalize,
        ])

    return normalize