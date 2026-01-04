import random
from dataset.hdlayout3k_gt import HDLayoutDataset, make_hdlayout_transforms
from pathlib import Path
import torch

class SAMHDLayoutDataset(HDLayoutDataset):
    def __init__(self, img_folder, json_file, transforms, num_queries, expand_factor=1):
        super().__init__(img_folder, json_file, transforms, num_queries, expand_factor)
    
    def __len__(self):
        # return super().__len__()
        return 3
        # return 30
    
    def __getitem__(self, idx):
        img, target, img_path = super().__getitem__(idx)
        
        # 设定 50% 的概率生成交互信号
        if random.random() <= 1.0 and len(target['block_bbox']) > 0:
            # --- 1. 生成正样本点 (Label 1) ---
            t_idx = random.randint(0, len(target['block_bbox']) - 1)
            pos_bbox = target['block_bbox'][t_idx]  # [cx, cy, w, h] (归一化)
            
            cx, cy, w, h = pos_bbox.tolist()
            # 在目标框内随机取一点
            px = (cx + (random.random() - 0.5) * w * 0.8) * 1024
            py = (cy + (random.random() - 0.5) * h * 0.8) * 1024
            
            points = [[px, py]]
            labels = [1]
            
            # --- 2. 生成负样本点 (Label 0) ---
            # 尝试最多 5 次采样，寻找一个不在任何 Block 内的点
            for _ in range(2): # 尝试添加 1-2 个负样本点
                for _ in range(10): # 寻找背景点的尝试次数
                    nx = random.random() * 1024
                    ny = random.random() * 1024
                    
                    # 检查点 (nx/1024, ny/1024) 是否在任何 GT Block 内
                    is_inside = False
                    for b in target['block_bbox']:
                        bcx, bcy, bw, bh = b.tolist()
                        if (bcx - bw/2 <= nx/1024 <= bcx + bw/2) and \
                           (bcy - bh/2 <= ny/1024 <= bcy + bh/2):
                            is_inside = True
                            break
                    
                    if not is_inside:
                        points.append([nx, ny])
                        labels.append(0)
                        break
            
            target['prompts'] = {
                'points': torch.tensor([points], dtype=torch.float32), # [1, N, 2]
                'labels': torch.tensor([labels], dtype=torch.int64),   # [1, N]
                'boxes': None
            }
        else:
            target['prompts'] = {}
            
        return img, target, img_path
    
    
    
def buildDataset(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "jsons" ),
        "val": (root / "val" / "images", root / "val" / "jsons" ),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = SAMHDLayoutDataset(
        img_folder, 
        ann_file, 
        transforms=make_hdlayout_transforms(image_set), 
        num_queries=args.num_queries,
        expand_factor=2 if image_set=='train' else 1
    )
    return dataset