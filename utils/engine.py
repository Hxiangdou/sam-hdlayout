# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np
import torch

import utils.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    
    # freeze_epochs = 10
    # --- 动态冷冻策略 ---
    # if epoch < freeze_epochs:
    #     print(f"Epoch {epoch}: Freezing SAM Backbone (visual_encoder & prompt_encoder)")
        # 锁定 SAM 权重
    for name, param in model.named_parameters():
        if "visual_encoder" in name or "prompt_encoder" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # 确保检测头是开启的
    # else:
    #     if epoch == freeze_epochs:
    #         print(f"Epoch {epoch}: Unfreezing all parameters for full fine-tuning")
    #         # 全量微调
    #         for param in model.parameters():
    #             param.requires_grad = True
            
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    ispower = False
    accumulation_steps = 4
    optimizer.zero_grad()
    i = 0
    for imgs, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        imgs = imgs.to(device)
        # block_bboxes = torch.tensor(np.array([bbox[0] for bbox in block_bboxes])).to(device)
        prompts = [t.get('prompts') for t in targets]
        prompts = [{k: v.to(device) if v is not None else None for k, v in p.items()} if p else {} for p in prompts]
        
        targets = [{k: v.to(device) for k, v in t.items() if k != 'prompts'} for t in targets]
        imgs = imgs.tensors
        
        outputs = model(imgs, prompts=prompts)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # if epoch % 10 == 1 and not ispower:
        #     weight_dict['loss_overlap_line1'] *= 10
        #     ispower = True
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        losses = losses / accumulation_steps
        losses.backward()
        
        if (i + 1) % accumulation_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        i += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, eval=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    render_res = []
    json_res = []
    for imgs, targets, img_paths in metric_logger.log_every(data_loader, 10, header):
        imgs = imgs.to(device)
        # block_bboxes = torch.tensor(np.array([bbox[0] for bbox in block_bboxes])).to(device)
        prompts = [t.get('prompts') for t in targets]
        prompts = [{k: v.to(device) if v is not None else None for k, v in p.items()} if p else {} for p in prompts]
        
        targets = [{k: v.to(device) for k, v in t.items() if k != 'prompts'} for t in targets]
        imgs = imgs.tensors
        
        # prompts = [t.get('prompts') for t in targets]
        outputs = model(imgs, prompts=prompts)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['res'](outputs, orig_target_sizes)

        if len(render_res) < 6:
            render_res.append({
                "img_path": img_paths[0],
                "prompts": prompts[0],
                "results": results[0]
                })
        if eval:
            for result, img_path in zip(results, img_paths):
                json_res.append({
                    "img_path": img_path,
                    "results": result
                })
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if eval:
        return stats, json_res
    return stats, render_res
