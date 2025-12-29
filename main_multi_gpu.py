# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils.misc as utils
# from dataset import buildDataset
from dataset.hdlayout3k import buildDataset
from utils.engine import evaluate, train_one_epoch
# from models.dino_hdlayout_v5 import build
from models.sam_hdlayout import build
from utils.plotUtils import DataRender, json_save
from torch.utils.tensorboard import SummaryWriter
from models.vit_util import custom_collate

def get_args_parser():
    parser = argparse.ArgumentParser('Set HDLayout model', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int, help="Total batch size across all GPUs")
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=[1, 4, 1], type=list,
                        help="Number of query slots")
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--intermediate', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_points', default=5, type=float,
                        help="Points coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_overlap', default=0.5, type=float,
                        help="overlap coefficient in the matching cost")

    # dataset parameters
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/data1/fength/HDDiffuser-Text/outputs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


def main(args):
    # ================= 1. 环境与设备设置 =================
    # 不使用 utils.init_distributed_mode(args)
    print(args)

    # 自动检测 GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    # 简单的 Seed 设置
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # ================= 2. 构建模型 =================
    model, criterion, postprocessors = build(args)
    model.to(device)

    model_without_ddp = model
    
    # 启用 DataParallel
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        # DataParallel 包装后，原模型在 model.module 中
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # ================= 3. 优化器与调度器 =================
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # ================= 4. 数据加载 =================
    dataset_train = buildDataset(image_set='train', args=args)
    dataset_val = buildDataset(image_set='val', args=args)

    # 使用普通 Sampler，而非 DistributedSampler
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # ================= 5. 加载 Checkpoint / 预训练 =================
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # 输出目录设置
    output_dir = Path(args.output_dir)
    time_now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = Path(args.output_dir) / time_now
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Resume 逻辑
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        # 处理可能的 DDP module 前缀问题 (以防万一)
        state_dict = checkpoint['model']
        # 如果 checkpoint 是 DataParallel 保存的，可能没有 module. 前缀，或者有
        # 这里我们统一加载到 model_without_ddp，所以确保 key 匹配
        if list(state_dict.keys())[0].startswith('module.'):
             state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model_without_ddp.load_state_dict(state_dict)
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # ================= 6. 评估模式 =================
    if args.eval:
        test_stats, json_res = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir, args.eval
        )
        print(f"samples result save: {output_dir}")
        json_path = os.path.join(output_dir, 'jsons')
        os.makedirs(json_path, exist_ok=True)
        json_save(json_res, json_path)
        return

    # ================= 7. 训练循环 =================
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # DataParallel 不需要 set_epoch
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                # 无论是否多卡，都保存 model_without_ddp
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, render_res = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir
        )
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        writer = SummaryWriter(f'/data1/fength/HDlayout-logs/{time_now}')
        
        # Log metrics
        for key, value in log_stats.items():
            writer.add_scalar(key, value, epoch)
        
        render = DataRender()
        res_render = render.render(render_res)
        
        for key, value in res_render.items():
            writer.add_images(key, render.pil_to_nchw(value), epoch)
        
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HDLayout training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # 路径处理
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)