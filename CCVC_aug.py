import argparse
import sys
import time
from copy import deepcopy
import logging
import os
import pprint
import math
import pdb
import random
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
try:
    from tensorboardX import SummaryWriter
except ImportError:  # PyTorch 2.x images (Kaggle) may lack tensorboardX; CSV logs are the primary record
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter(object):
            def __init__(self, *a, **k): pass
            def add_scalar(self, *a, **k): pass
import yaml
from dataset.aug_semi2 import SemiDataset
from model.semseg.deeplabv3plus import Discrepancy_DeepLabV3Plus
from valid import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log_save, vote_label_selection, soft_label_selection, vote_soft_label_selection, vote_threshold_label_selection
from util.run_utils import RunContext, get_logger, count_lines, exit_code
from util.dist_helper import setup_distributed, find_free_port, init_ddp, barrier, unwrap


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
# default settings
parser.add_argument('--config', default='configs/pascal.yaml', type=str)
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--labeled_id_path', default='partitions/pascal/366/labeled.txt', type=str)
parser.add_argument('--unlabeled_id_path', default='partitions/pascal/366/unlabeled.txt', type=str)
parser.add_argument('--save_path', default='test', type=str)
parser.add_argument('--data_root', default=None, type=str)                    # overrides cfg['data_root']
parser.add_argument('--val_id_path', default=None, type=str)                  # default: partitions/<dataset>/val.txt
parser.add_argument('--load_path', default='test', type=str) 
parser.add_argument('--port', default=0, type=int)                              # 0: pick a free port
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch_size', default=6, type=int)                        # batch size per GPU
parser.add_argument('--crop_size', default=321, type=int)                       # 512 or 321
parser.add_argument('--seed', default=22, type=int)                             # 512 or 321   
parser.add_argument('--num_workers', default=2, type=int)                             # 512 or 321   
# ddp settings
parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int, help='number of gpus per node')
parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])  # cpu: tests only
parser.add_argument('--amp', default=True, type=str2bool)                       # mixed precision (deviation from paper, see README_OEM)
parser.add_argument('--nodes', default=1, type=int, help='number of nodes')
parser.add_argument("--ddp", default=True, type=str2bool, help='distributed data parallel training or not')
# network settings
parser.add_argument('--mode_criterion', default='CE', type=str, choices=['CE', 'OHEM'])
parser.add_argument('--mode_mapping', default='else', type=str)                         # both or else, the only difference is whether to use mapping on branch1
parser.add_argument('--use_MLP', default=False, type=str2bool)
parser.add_argument('--use_norm', default=False, type=str2bool)
parser.add_argument('--use_dropout', default=False, type=str2bool)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--conf_threshold', default=0.5, type=float)
parser.add_argument('--conf_prob', default=True, type=str2bool)                 # True: gamma vs softmax prob (paper); False: vs raw logit (upstream)
parser.add_argument('--mode_confident', default='no', type=str, choices=['normal', 'soft', 'vote', 'vote_threshold', 'vote_soft', 'no']) 
parser.add_argument('--use_SPL', default=False, type=str2bool)
parser.add_argument('--use_aug', default=True, type=str2bool)
# weight settings
parser.add_argument('--w_CE', default=1.0, type=float)
parser.add_argument('--w_con', default=1.0, type=float)
parser.add_argument('--w_dis', default=1.0, type=float)
parser.add_argument('--w_confident', default=1.5, type=float)
parser.add_argument('--w_unconfident', default=0.75, type=float)
# optimizer settings
parser.add_argument('--base_lr', default=0.001, type=float) 
parser.add_argument('--lr_network', default=5.0, type=float)                            # coefficient of the lr of other modules of the model
parser.add_argument('--lr_backbone', default=5.0, type=float)                           # coefficient of the lr of the backbone of the model
parser.add_argument('--mul_scheduler', default=0.9, type=float)                         # coefficient of the exp scheduler
# run settings (OEM / Kaggle reproduction)
parser.add_argument('--eval_every', default=1, type=int)                          # upstream: every epoch
parser.add_argument('--trial_iters', default=0, type=int)                         # >0: short trial run + trial_report.json
parser.add_argument('--max_session_hours', default=0.0, type=float)               # >0: stop after the epoch running past it
parser.add_argument('--session_start', default=0.0, type=float)                   # unix time the Kaggle session started
parser.add_argument('--log_every', default=50, type=int)                          # train_log.csv period (iterations)
parser.add_argument('--vis_num', default=4, type=int)                             # fixed val images in the example panels
parser.add_argument('--vis_seed', default=0, type=int)
parser.add_argument('--run_meta', default='', type=str)                           # json written by tools/run_baseline.sh

args = parser.parse_args()

args.world_size = args.gpus * args.nodes
args.ddp = True if (args.gpus > 1 and args.device == 'cuda') else False
args.world_size = args.world_size if args.ddp else 1

def main(gpu, ngpus_per_node, cfg, args):

    args.local_rank = gpu
    args.device = torch.device('cuda', max(gpu, 0)) if args.device == 'cuda' else torch.device('cpu')

    if args.local_rank <= 0:
        os.makedirs(args.save_path, exist_ok=True)
        
    logger = get_logger(args)                    # rank 0: stdout + <save_path>/log.txt; other ranks: silent

    if args.local_rank <= 0:
        tb_dir = args.save_path
        tb = SummaryWriter(log_dir=tb_dir)

    args.dist_backend = 'none'
    if args.ddp:
        args.dist_backend = init_ddp(args.local_rank, args.world_size, args.ports)
    elif args.device.type == 'cuda':
        torch.cuda.set_device(args.device)

    if args.local_rank <= 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    args.amp = args.amp and args.device.type == 'cuda'
    model = Discrepancy_DeepLabV3Plus(args, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    if args.local_rank <= 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    # pdb.set_trace()
    # TODO: check the parameter !!!

    optimizer = SGD([{'params': model.branch1.backbone.parameters(), 'lr': args.base_lr * args.lr_backbone}, 
                     {'params': model.branch2.backbone.parameters(), 'lr': args.base_lr * args.lr_backbone},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': args.base_lr * args.lr_network}], lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    if args.ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(args.device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=False)

    # # try to load saved model
    # try:   
    #     model.load_model(args.load_path)
    #     if args.local_rank <= 0:
    #         logger.info('load saved model')
    # except:
    #     if args.local_rank <= 0:
    #         logger.info('no saved model')

    # ---- #
    # loss #
    # ---- #
    # CE loss for labeled data
    if args.mode_criterion == 'CE':
        criterion_l = nn.CrossEntropyLoss(reduction='mean', ignore_index=255).to(args.device)
    else:
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).to(args.device)
    
    # consistency loss for unlabeled data
    criterion_u = nn.CrossEntropyLoss(reduction='none').to(args.device)

    # ------- #
    # dataset #
    # ------- #
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', val_id_path=args.val_id_path)

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l, num_replicas=args.world_size,
                                                                     rank=max(args.local_rank, 0), seed=args.seed)
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size,
                               pin_memory=True, num_workers=args.num_workers, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u, num_replicas=args.world_size,
                                                                     rank=max(args.local_rank, 0), seed=args.seed)
    
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,
                               pin_memory=True, num_workers=args.num_workers, drop_last=True, sampler=trainsampler_u)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    # if args.ddp:
    #     valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # else:
    #     valsampler = None
    
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False, sampler=valsampler)

    iters_per_epoch = len(trainloader_u)
    total_iters = iters_per_epoch * args.epochs

    # config.json, resume from last.pth (model / optimizer / GradScaler / epoch / RNG), run summary
    ctx = RunContext(args, cfg, logger, os.path.basename(__file__))
    start_epoch, (previous_best, previous_best1, previous_best2) = ctx.start(
        model, optimizer, scaler, count_lines(args.labeled_id_path), count_lines(args.unlabeled_id_path),
        valset, iters_per_epoch)

    # can change with epochs, add SPL here
    conf_threshold = args.conf_threshold

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        if args.local_rank <= 0:
            logger.info('===========> Epoch: {:}, backbone1 LR: {:.4f}, backbone2 LR: {:.4f}, segmentation LR: {:.4f}'.format(
                epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[-1]['lr']))
            logger.info('===========> Epoch: {:}, Previous best of ave: {:.2f}, Previous best of branch1: {:.2f}, Previous best of branch2: {:.2f}'.format(
                epoch, previous_best, previous_best1, previous_best2))

        total_loss, total_loss_CE, total_loss_con, total_loss_dis = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        total_labeled = 0
        total_unlabeled = 0

        for i, ((labeled_img, labeled_img_mask), (unlabeled_img, aug_unlabeled_img, ignore_img_mask, unlabeled_cutmix_box)) in enumerate(loader):

            labeled_img, labeled_img_mask = labeled_img.to(args.device), labeled_img_mask.to(args.device)
            unlabeled_img, aug_unlabeled_img, ignore_img_mask, unlabeled_cutmix_box = unlabeled_img.to(args.device), aug_unlabeled_img.to(args.device), ignore_img_mask.to(args.device), unlabeled_cutmix_box.to(args.device)

            optimizer.zero_grad()

            barrier(args)

            num_lb, num_ulb = labeled_img.shape[0], unlabeled_img.shape[0]

            total_labeled += num_lb
            total_unlabeled += num_ulb

            # =========================================================================================
            # labeled data: labeled_img, labeled_img_mask
            # =========================================================================================
            model.train()
            labeled_logits = model(labeled_img)

            # =========================================================================================
            # unlabeled data: unlabeled_img, ignore_img_mask, cutmix_box
            # =========================================================================================
            # first feed the data into the model with no grad to get gt
            with torch.no_grad():
                model.eval()
                unlabeled_logits = model(unlabeled_img)

                unlabeled_pseudo_label1 = unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()
                unlabeled_pseudo_label2 = unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()
                unlabeled_pred1 = unlabeled_logits['pred1'].detach()
                unlabeled_pred2 = unlabeled_logits['pred2'].detach()

            # then perform cutmix
            aug_unlabeled_img_for_mix = aug_unlabeled_img.clone()
            aug_unlabeled_ignore_img_mask_for_mix = ignore_img_mask.clone()
            aug_unlabeled_pseudo_label1_for_mix = unlabeled_pseudo_label1.clone()
            aug_unlabeled_pseudo_label2_for_mix = unlabeled_pseudo_label2.clone()
            aug_unlabeled_pred1_for_mix = unlabeled_pred1.clone()
            aug_unlabeled_pred2_for_mix = unlabeled_pred2.clone()

            # initial cutmix is companied with a probablility 0.5, increase the data divergence
            aug_unlabeled_img_for_mix[unlabeled_cutmix_box.unsqueeze(1).expand(aug_unlabeled_img_for_mix.shape) == 1] = aug_unlabeled_img_for_mix.flip(0)[unlabeled_cutmix_box.unsqueeze(1).expand(aug_unlabeled_img_for_mix.shape) == 1]
            aug_unlabeled_ignore_img_mask_for_mix[unlabeled_cutmix_box == 1] = aug_unlabeled_ignore_img_mask_for_mix.flip(0)[unlabeled_cutmix_box == 1]
            aug_unlabeled_pseudo_label1_for_mix[unlabeled_cutmix_box == 1] = aug_unlabeled_pseudo_label1_for_mix.flip(0)[unlabeled_cutmix_box == 1]
            aug_unlabeled_pseudo_label2_for_mix[unlabeled_cutmix_box == 1] = aug_unlabeled_pseudo_label2_for_mix.flip(0)[unlabeled_cutmix_box == 1]
            aug_unlabeled_pred1_for_mix[unlabeled_cutmix_box.unsqueeze(1).expand(aug_unlabeled_pred1_for_mix.shape) == 1] = aug_unlabeled_pred1_for_mix.flip(0)[unlabeled_cutmix_box.unsqueeze(1).expand(aug_unlabeled_pred1_for_mix.shape) == 1]
            aug_unlabeled_pred2_for_mix[unlabeled_cutmix_box.unsqueeze(1).expand(aug_unlabeled_pred2_for_mix.shape) == 1] = aug_unlabeled_pred2_for_mix.flip(0)[unlabeled_cutmix_box.unsqueeze(1).expand(aug_unlabeled_pred2_for_mix.shape) == 1]

            # finally feed the mixed data into the model
            model.train()
            cutmixed_aug_unlabeled_logits = model(aug_unlabeled_img_for_mix)

            # one extra branch: unlabeled data
            raw_unlabeled_logits = model(unlabeled_img)


            # to count the confident predictions
            unlabeled_pred_confidence1 = raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[0]
            unlabeled_pred_confidence2 = raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[0]

            # =========================================================================================
            # calculate loss
            # =========================================================================================

            # -------------
            # labeled
            # -------------
            # CE loss
            labeled_pred1 = labeled_logits['pred1']
            labeled_pred2 = labeled_logits['pred2']

            loss_CE1 = criterion_l(labeled_pred1, labeled_img_mask)
            loss_CE2 = criterion_l(labeled_pred2, labeled_img_mask)

            loss_CE = (loss_CE1 + loss_CE2) / 2
            loss_CE = loss_CE * args.w_CE

            # -------------
            # unlabeled
            # -------------
            # consistency loss
            raw_unlabeled_pred1 = raw_unlabeled_logits['pred1']
            raw_unlabeled_pred2 = raw_unlabeled_logits['pred2']

            cutmixed_aug_unlabeled_pred1 = cutmixed_aug_unlabeled_logits['pred1']
            cutmixed_aug_unlabeled_pred2 = cutmixed_aug_unlabeled_logits['pred2']

            if args.mode_confident == 'normal':
                loss_con1 = criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * ((aug_unlabeled_pred1_for_mix.softmax(dim=1).max(dim=1)[0] > conf_threshold) & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con1 = torch.sum(loss_con1) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()
                loss_con2 = criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * ((aug_unlabeled_pred2_for_mix.softmax(dim=1).max(dim=1)[0] > conf_threshold) & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2 = torch.sum(loss_con2) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()

                loss_raw_con1 = criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * ((raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[0] > conf_threshold) & (ignore_img_mask != 255))
                loss_raw_con1 = torch.sum(loss_raw_con1) / torch.sum(ignore_img_mask != 255).item()
                loss_raw_con2 = criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * ((raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[0] > conf_threshold) & (ignore_img_mask != 255))
                loss_raw_con2 = torch.sum(loss_raw_con2) / torch.sum(ignore_img_mask != 255).item()

            elif args.mode_confident == 'soft':
                confident_pred1, confident_pred2, unconfident_pred1, unconfident_pred2 = soft_label_selection(aug_unlabeled_pred1_for_mix, aug_unlabeled_pred2_for_mix, conf_threshold)

                loss_con1_confident = criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (confident_pred1 & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_confident = criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (confident_pred2 & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1_unconfident = 0.5 * criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (unconfident_pred1 & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_unconfident = 0.5 * criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (unconfident_pred2 & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1 = (torch.sum(loss_con1_confident) + torch.sum(loss_con1_unconfident)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()
                loss_con2 = (torch.sum(loss_con2_confident) + torch.sum(loss_con2_unconfident)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()

                raw_confident_pred1, raw_confident_pred2, raw_unconfident_pred1, raw_unconfident_pred2 = soft_label_selection(raw_unlabeled_pred1, raw_unlabeled_pred2, conf_threshold)

                loss_raw_con1_confident = criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_confident_pred1 & (ignore_img_mask != 255))
                loss_raw_con2_confident = criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_confident_pred2 & (ignore_img_mask != 255))

                loss_raw_con1_unconfident = 0.5 * criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_unconfident_pred1 & (ignore_img_mask != 255))
                loss_raw_con2_unconfident = 0.5 * criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_unconfident_pred2 & (ignore_img_mask != 255))

                loss_raw_con1 = (torch.sum(loss_raw_con1_confident) + torch.sum(loss_raw_con1_unconfident)) / torch.sum(ignore_img_mask != 255).item()
                loss_raw_con2 = (torch.sum(loss_raw_con2_confident) + torch.sum(loss_raw_con2_unconfident)) / torch.sum(ignore_img_mask != 255).item()

            elif args.mode_confident == 'vote':
                same_pred, different_pred = vote_label_selection(unlabeled_pred1, unlabeled_pred2)

                loss_con1_same = criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (same_pred & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_same = criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (same_pred & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1_different = 1.5 * criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (different_pred & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_different = 1.5 * criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (different_pred & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1 = (torch.sum(loss_con1_same) + torch.sum(loss_con1_different)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()
                loss_con2 = (torch.sum(loss_con2_same) + torch.sum(loss_con2_different)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()

                raw_same_pred, raw_different_pred = vote_label_selection(raw_unlabeled_pred1, raw_unlabeled_pred2)

                loss_raw_con1_same = criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_same_pred & (ignore_img_mask != 255))
                loss_raw_con2_same = criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_same_pred & (ignore_img_mask != 255))

                loss_raw_con1_different = 1.5 * criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different_pred & (ignore_img_mask != 255))
                loss_raw_con2_different = 1.5 * criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different_pred & (ignore_img_mask != 255))

                loss_raw_con1 = (torch.sum(loss_raw_con1_same) + torch.sum(loss_raw_con1_different)) / torch.sum(ignore_img_mask != 255).item()
                loss_raw_con2 = (torch.sum(loss_raw_con2_same) + torch.sum(loss_raw_con2_different)) / torch.sum(ignore_img_mask != 255).item()

            elif args.mode_confident == 'vote_threshold':
                different1_confident, different1_else, different2_confident, different2_else = vote_threshold_label_selection(unlabeled_pred1, unlabeled_pred2, conf_threshold, use_prob=args.conf_prob)

                loss_con1_else = criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (different1_else & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_else = criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (different2_else & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1_cc = args.w_confident * criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (different1_confident & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_cc = args.w_confident * criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (different2_confident & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1 = (torch.sum(loss_con1_else) + torch.sum(loss_con1_cc)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()
                loss_con2 = (torch.sum(loss_con2_else) + torch.sum(loss_con2_cc)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()

                raw_different1_confident, raw_different1_else, raw_different2_confident, raw_different2_else = vote_threshold_label_selection(raw_unlabeled_pred1, raw_unlabeled_pred2, conf_threshold, use_prob=args.conf_prob)

                loss_raw_con1_else = criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different1_else & (ignore_img_mask != 255))
                loss_raw_con2_else = criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different2_else & (ignore_img_mask != 255))

                loss_raw_con1_cc = args.w_confident * criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different1_confident & (ignore_img_mask != 255))
                loss_raw_con2_cc = args.w_confident * criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different2_confident & (ignore_img_mask != 255))

                loss_raw_con1 = (torch.sum(loss_raw_con1_else) + torch.sum(loss_raw_con1_cc)) / torch.sum(ignore_img_mask != 255).item()
                loss_raw_con2 = (torch.sum(loss_raw_con2_else) + torch.sum(loss_raw_con2_cc)) / torch.sum(ignore_img_mask != 255).item()

            elif args.mode_confident == 'vote_soft':
                same_pred, different_confident_pred1, different_confident_pred2, different_unconfident_pred1, different_unconfident_pred2 = vote_soft_label_selection(unlabeled_pred1, unlabeled_pred2, conf_threshold)

                loss_con1_same = criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (same_pred & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_same = criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (same_pred & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1_different_confident = 1.5 * criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (different_confident_pred1 & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_different_confident = 1.5 * criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (different_confident_pred2 & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1_different_unconfident = 0.5 * criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (different_unconfident_pred1 & (aug_unlabeled_ignore_img_mask_for_mix != 255))
                loss_con2_different_unconfident = 0.5 * criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (different_unconfident_pred2 & (aug_unlabeled_ignore_img_mask_for_mix != 255))

                loss_con1 = (torch.sum(loss_con1_same) + torch.sum(loss_con1_different_confident) + torch.sum(loss_con1_different_unconfident)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()
                loss_con2 = (torch.sum(loss_con2_same) + torch.sum(loss_con2_different_confident) + torch.sum(loss_con2_different_unconfident)) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()

                raw_same_pred, raw_different_confident_pred1, raw_different_confident_pred2, raw_different_unconfident_pred1, raw_different_unconfident_pred2 = vote_soft_label_selection(unlabeled_pred1, unlabeled_pred2, conf_threshold)

                loss_raw_con1_same = criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_same_pred & (ignore_img_mask != 255))
                loss_raw_con2_same = criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_same_pred & (ignore_img_mask != 255))

                loss_raw_con1_different_confident = 1.5 * criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different_confident_pred1 & (ignore_img_mask != 255))
                loss_raw_con2_different_confident = 1.5 * criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different_confident_pred2 & (ignore_img_mask != 255))

                loss_raw_con1_different_unconfident = 0.5 * criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different_unconfident_pred1 & (ignore_img_mask != 255))
                loss_raw_con2_different_unconfident = 0.5 * criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (raw_different_unconfident_pred2 & (ignore_img_mask != 255))

                loss_raw_con1 = (torch.sum(loss_raw_con1_same) + torch.sum(loss_raw_con1_different_confident) + torch.sum(loss_raw_con1_different_unconfident)) / torch.sum(ignore_img_mask != 255).item()
                loss_raw_con2 = (torch.sum(loss_raw_con2_same) + torch.sum(loss_raw_con2_different_confident) + torch.sum(loss_raw_con2_different_unconfident)) / torch.sum(ignore_img_mask != 255).item()

            else:
                loss_con1 = criterion_u(cutmixed_aug_unlabeled_pred2, aug_unlabeled_pseudo_label1_for_mix) * (aug_unlabeled_ignore_img_mask_for_mix != 255)
                loss_con1 = torch.sum(loss_con1) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()
                loss_con2 = criterion_u(cutmixed_aug_unlabeled_pred1, aug_unlabeled_pseudo_label2_for_mix) * (aug_unlabeled_ignore_img_mask_for_mix != 255)
                loss_con2 = torch.sum(loss_con2) / torch.sum(aug_unlabeled_ignore_img_mask_for_mix != 255).item()

                loss_raw_con1 = criterion_u(raw_unlabeled_pred2, raw_unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (ignore_img_mask != 255)
                loss_raw_con1 = torch.sum(loss_raw_con1) / torch.sum(ignore_img_mask != 255).item()
                loss_raw_con2 = criterion_u(raw_unlabeled_pred1, raw_unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (ignore_img_mask != 255)
                loss_raw_con2 = torch.sum(loss_raw_con2) / torch.sum(ignore_img_mask != 255).item()

            loss_con = (loss_con1 + loss_con2 + loss_raw_con1 + loss_raw_con2) / 4
            loss_con_raw = loss_con
            loss_con = loss_con * args.w_con

            # -------------
            # both
            # -------------
            # discrepancy loss
            cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)

            # labeled
            labeled_feature1 = labeled_logits['feature1']
            labeled_feature2 = labeled_logits['feature2']
            loss_dis_labeled1 = 1 + cos_dis(labeled_feature1.detach(), labeled_feature2).mean()
            loss_dis_labeled2 = 1 + cos_dis(labeled_feature2.detach(), labeled_feature1).mean()
            loss_dis_labeled = (loss_dis_labeled1 + loss_dis_labeled2) / 2

            # unlabeled
            cutmixed_aug_unlabeled_feature1 = cutmixed_aug_unlabeled_logits['feature1']
            cutmixed_aug_unlabeled_feature2 = cutmixed_aug_unlabeled_logits['feature2']
            loss_dis_cutmixed_aug_unlabeled1 = 1 + cos_dis(cutmixed_aug_unlabeled_feature1.detach(), cutmixed_aug_unlabeled_feature2).mean()
            loss_dis_cutmixed_aug_unlabeled2 = 1 + cos_dis(cutmixed_aug_unlabeled_feature2.detach(), cutmixed_aug_unlabeled_feature1).mean()
            loss_dis_cutmixed_aug_unlabeled = (loss_dis_cutmixed_aug_unlabeled1 + loss_dis_cutmixed_aug_unlabeled2) / 2

            raw_unlabeled_feature1 = raw_unlabeled_logits['feature1']
            raw_unlabeled_feature2 = raw_unlabeled_logits['feature2']
            loss_dis_raw_unlabeled1 = 1 + cos_dis(raw_unlabeled_feature1.detach(), raw_unlabeled_feature2).mean()
            loss_dis_raw_unlabeled2 = 1 + cos_dis(raw_unlabeled_feature2.detach(), raw_unlabeled_feature1).mean()
            loss_dis_raw_unlabeled = (loss_dis_raw_unlabeled1 + loss_dis_raw_unlabeled2) / 2

            loss_dis_unlabeled = (loss_dis_cutmixed_aug_unlabeled + loss_dis_raw_unlabeled) / 2

            loss_dis = (loss_dis_labeled + loss_dis_unlabeled) / 2
            loss_dis_raw = loss_dis
            loss_dis = loss_dis * args.w_dis

            # -------------
            # total
            # -------------
            loss = loss_CE + loss_con + loss_dis

            barrier(args)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_loss_CE += loss_CE.item()
            total_loss_con += loss_con.item()
            total_loss_dis += loss_dis.item()

            total_confident = (((unlabeled_pred_confidence1 >= 0.95) & (ignore_img_mask != 255)).sum().item() + ((unlabeled_pred_confidence2 >= 0.95) & (ignore_img_mask != 255)).sum().item()) / 2
            total_mask_ratio += total_confident / (ignore_img_mask != 255).sum().item()

            iters = epoch * iters_per_epoch + i

            # update lr

            backbone_lr = args.base_lr * (1 - iters / total_iters) ** args.mul_scheduler
            backbone_lr = backbone_lr * args.lr_backbone

            seg_lr = args.base_lr * (1 - iters / total_iters) ** args.mul_scheduler
            seg_lr = seg_lr * args.lr_network
                
            optimizer.param_groups[0]['lr'] = backbone_lr
            optimizer.param_groups[1]['lr'] = backbone_lr
            for ii in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[ii]['lr'] = seg_lr

            if (i % max(iters_per_epoch // 8, 1) == 0) and (args.local_rank <= 0):
                tb.add_scalar('train_loss_total', total_loss / (i+1), iters)
                tb.add_scalar('train_loss_CE', total_loss_CE / (i+1), iters)
                tb.add_scalar('train_loss_con', total_loss_con / (i+1), iters)
                tb.add_scalar('train_loss_dis', total_loss_dis / (i+1), iters)

            if (i % max(iters_per_epoch // 8, 1) == 0) and (args.local_rank <= 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss CE: {:.3f}, '
                            'Loss consistency: {:.3f}, Loss discrepancy: {:.3f}, Mask: {:.3f}'.format(
                    i, total_loss / (i+1), total_loss_CE / (i+1), total_loss_con / (i+1), total_loss_dis / (i+1), 
                    total_mask_ratio / (i+1)))

            if ctx.after_iter(epoch, i, iters, optimizer, loss, loss_CE, loss_con, loss_dis, loss_con_raw, loss_dis_raw,
                              (True, True, True), unlabeled_pred1, unlabeled_pred2, ignore_img_mask != 255):
                break

        if ctx.trial_done:
            break

        if args.use_SPL:
            conf_threshold += 0.01
            if conf_threshold >= 0.95:
                conf_threshold = 0.95

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < args.epochs - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        
        barrier(args)

        # test with different branches: every --eval_every epochs and at the last epoch (rank 0 only)
        evaluate_result, eval_time = None, None
        if ctx.should_eval(epoch):
            if args.local_rank <= 0:
                evaluate_result = ctx.run_eval(model, valloader, eval_mode, epoch, iters + 1)
                eval_time = evaluate_result['eval_time_s']

                mIOU1 = evaluate_result['IOU1']
                mIOU2 = evaluate_result['IOU2']
                mIOU_ave = evaluate_result['IOU_ave']

                tb.add_scalar('meanIOU_branch1', mIOU1, epoch)
                tb.add_scalar('meanIOU_branch2', mIOU2, epoch)
                tb.add_scalar('meanIOU_ave', mIOU_ave, epoch)

                logger.info('***** Evaluation with branch 1 {} ***** >>>> meanIOU: {:.2f}'.format(evaluate_result['mode'], mIOU1))
                logger.info('***** Evaluation with branch 2 {} ***** >>>> meanIOU: {:.2f}'.format(evaluate_result['mode'], mIOU2))
                logger.info('***** Evaluation with two branches {} ***** >>>> meanIOU: {:.2f}'.format(evaluate_result['mode'], mIOU_ave))

                # best-on-val is kept for reference only: no 'best' checkpoints (only last.pth / final.pth)
                previous_best1 = max(previous_best1, mIOU1)
                previous_best2 = max(previous_best2, mIOU2)
                previous_best = max(previous_best, mIOU_ave)
            barrier(args)

        # last.pth every epoch (final.pth at the end); stop cleanly past --max_session_hours
        if ctx.end_epoch(epoch, iters, model, optimizer, scaler, (previous_best, previous_best1, previous_best2),
                         last_eval=evaluate_result, epoch_time=time.time() - epoch_t0 - (eval_time or 0.0),
                         eval_time=eval_time):
            return

    if ctx.trial_done:
        ctx.finish_trial(model, valloader, 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original',
                         epoch, iters + 1)


if __name__ == '__main__':

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    if args.data_root:
        cfg['data_root'] = args.data_root

    cfg['n_gpu'] = args.world_size

    # two ports chosen once here so that every rank agrees: NCCL, then gloo fallback
    args.ports = (args.port or find_free_port(), find_free_port())
    # args.base_lr = args.base_lr * args.world_size

    if args.ddp:
        mp.spawn(main, nprocs=cfg['n_gpu'], args=(cfg['n_gpu'], cfg, args))
    else:
        main(0, 1, cfg, args)

    # 0 = finished (or trial done), 3 = stopped by --max_session_hours (resume next session), 1 = failure
    sys.exit(exit_code(args.save_path, args.trial_iters > 0))
