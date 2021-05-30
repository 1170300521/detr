import argparse
import datetime
import json
import time
import os
import os.path as osp
import torch
from torch._C import device

from torch.utils.data import DataLoader, DistributedSampler, sampler
from yacs.config import CfgNode as CN


import datasets
import util.misc as utils
from datasets.ref_data import get_data, collater
from engine import evaluate
from models import build_model
from main import get_args_parser


def test(args):
    """
    Test on all testing dataset for some  benchmark
    """
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)
    device = torch.device(args.device)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    args.ds_info = CN(json.load(open(args.ds_info)))
    dataset = get_data(args, args.ds_info)['test']
    sampler_dict = {}
    if args.distributed:
        for k in dataset:
            sampler_dict[k] = DistributedSampler(dataset[k])
    else:
        for k in dataset:
            sampler_dict[k] = torch.utils.data.SequentialSampler(dataset[k])
    
    dataloader_dict = {}
    for k in dataset:
        dataloader_dict[k] = DataLoader(dataset[k], args.batch_size, sampler=sampler_dict[k],
            drop_last=False, collate_fn=collater, num_workers=args.num_workers)

    if args.resume:
        print("Use resume :", args.resume)
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    else:
        raise RuntimeError("No pretrained weights from {} !".format(args.resume))
    
    test_stats = {}
    for k in dataset:
        # visualize_dir = os.path.join(args.output_dir, args.visualize_dir, k)
        # if not os.path.exists(visualize_dir):
        #     os.makedirs(visualize_dir)
        os.makedirs(osp.join(args.output_dir, k), exist_ok=True)
        test_stats[k] = evaluate(model, criterion, postprocessors, dataloader_dict[k], 
            device, osp.join(args.output_dir, k))
    with open(osp.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.ds_name, args.lab_name)
    if utils.is_main_process() and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    test(args)
