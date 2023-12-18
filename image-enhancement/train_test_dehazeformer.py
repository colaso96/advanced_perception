# Bronte Sihan Li, Cole Crescas, Karan Shah
# 2023-09-23
# CS 7180
# Some of the code is heavily inspired by the DehazeFormer repo (https://github.com/IDKiro/DehazeFormer/tree/main)

"""
This is a wrapper script for training and testing the DehazeFormer models.
We use the default settings from the DehazeFormer repo.

Usage:
For training and testing:
python train_test_dehazeformer.py --model dehazeformer-s

For test only on a pretrained model:
python train_test_dehazeformer.py --model dehazeformer-s --test_only

"""

import sys, os
import argparse
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from collections import OrderedDict
from Dehaze.utils import AverageMeter, write_img, chw_to_hwc
from Dehaze.models import *
from pytorch_msssim import ssim

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument(
    '-t', '--test_only', action='store_true', help='test the model only'
)
args = parser.parse_args()

# seed everything

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.cuda.amp import autocast, GradScaler
import json

sys.path.append('Dehaze')
from Dehaze.datasets.loader import PairLoader, SingleLoader
from Dehaze.models import *

#Path to our selected dataset and training images
DATASET_DIR = 'data/a2i2/'
IMAGE_DIR = DATASET_DIR + 'UAV-train/paired_dehaze/images/'


def train(train_loader, network, criterion, optimizer, scaler):
    """
    Training loop for the model
    """
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast():
            output = network(source_img)
            loss = criterion(output, target_img)

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    """
    Validation loop for the model
    """
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img).clamp_(-1, 1)

        mse_loss = F.mse_loss(
            output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none'
        ).mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


def single(save_dir):
    """
    Helper function to load a single model from saved directory
    """
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    """
    Testing loop for the model
    """
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
            ssim_val = ssim(
                F.adaptive_avg_pool2d(
                    output, (int(H / down_ratio), int(W / down_ratio))
                ),
                F.adaptive_avg_pool2d(
                    target, (int(H / down_ratio), int(W / down_ratio))
                ),
                data_range=1,
                size_average=False,
            ).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print(
            'Test: [{0}]\t'
            'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
            'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'.format(idx, psnr=PSNR, ssim=SSIM)
        )

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        input_img = chw_to_hwc(input.detach().cpu().squeeze(0).numpy() * 0.5 + 0.5)
        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        target_img = chw_to_hwc(target.detach().cpu().squeeze(0).numpy())
        # Concactenate output and target image
        out_img = np.concatenate((input_img, out_img, target_img), axis=1)
        write_img(os.path.join(result_dir, 'imgs', f'dehaze_{filename}'), out_img)

    f_result.close()

    os.rename(
        os.path.join(result_dir, 'results.csv'),
        os.path.join(result_dir, '%.02f | %.04f.csv' % (PSNR.avg, SSIM.avg)),
    )


if __name__ == '__main__':
    model_name = args.model
    # Make directory for train and validation images
    train_dir = IMAGE_DIR + 'train/'
    val_dir = IMAGE_DIR + 'val/'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # Make directory for hazy and GT images
    os.makedirs(train_dir + 'hazy', exist_ok=True)
    os.makedirs(train_dir + 'GT', exist_ok=True)
    os.makedirs(val_dir + 'hazy', exist_ok=True)
    os.makedirs(val_dir + 'GT', exist_ok=True)

    # Load the dataset
    train_dataset = PairLoader(data_dir=train_dir, sub_dir='', mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    val_dataset = PairLoader(data_dir=val_dir, sub_dir='', mode='valid')
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )
    # Make directory for saving the model and logs
    save_dir = f'models/finetune/{model_name}/unfreeze_last2/'
    os.makedirs(save_dir, exist_ok=True)
    log_dir = f'logs/finetune/{model_name}/unfreeze_last2/'
    os.makedirs(log_dir, exist_ok=True)

    if not args.test_only:
        settings = json.load(open(f'Dehaze/configs/outdoor/{model_name}.json', 'r'))

        model = eval(args.model.replace('-', '_'))()
        model = nn.DataParallel(model).cuda()

        criterion = nn.L1Loss()

        optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings['epochs'], eta_min=settings['lr'] * 1e-2
        )
        scaler = GradScaler()

        # Load the pretrained model
        checkpoint = torch.load(f'models/outdoor/{model_name}.pth')
        model.load_state_dict(checkpoint['state_dict'])

        # Freeze all of the parameters except the last layer or 2
        for param in model.parameters():
            param.requires_grad = False
        for param in model.module.layer4.parameters():
            param.requires_grad = True
        for param in model.module.layer5.parameters():
            param.requires_grad = True

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

        writer = SummaryWriter(log_dir=log_dir)

        best_psnr = 0
        print(
            f'Start training for {settings["epochs"]} epochs (eval every {settings["eval_freq"]} epochs'
        )
        for epoch in tqdm(range(settings['epochs'] + 1)):
            print(f'Epoch: {epoch}')
            loss = train(train_loader, model, criterion, optimizer, scaler)

            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % settings['eval_freq'] == 0:
                avg_psnr = valid(val_loader, model)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                saved_model_dir = save_dir + f'{model_name}_{datetime.now()}.pth'
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save(
                        {'state_dict': model.state_dict()},
                        saved_model_dir,
                    )

                writer.add_scalar('best_psnr', best_psnr, epoch)

    # See the output of some examples
    result_dir = f'results/finetune/{model_name}/'
    os.makedirs(result_dir, exist_ok=True)

    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = eval(args.model.replace('-', '_'))()
    model = model.cuda()
    # Load last saved model from save_dir
    files = os.listdir(save_dir)
    latest_model_file = max(
        files, key=lambda x: os.path.getmtime(os.path.join(save_dir, x))
    )
    model.load_state_dict(single(os.path.join(save_dir, latest_model_file)))
    test(test_loader, model, result_dir)
