import os
import sys
import logging
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from DiffVNet.unet_teacher import Unet
from utils import maybe_mkdir, get_lr, fetch_data, poly_lr
from utils.loss import DC_and_CE_loss
from data.data_loaders import DatasetAllTasks
from utils.config import Config
from data.StrongAug import get_StrongAug, ToTensor, CenterCrop
import torchvision

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='synapse', required=True)
parser.add_argument('--exp', type=str, default='teacher_two_stage')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, required=True)
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('--teacher_ckpt', type=str, required=True)
parser.add_argument('--max_epoch_stage1', type=int, default=30)
parser.add_argument('--max_epoch_stage2', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(args.task)

# -------------------------
# Fix Seed
# -------------------------
import random
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------
# Logger
# -------------------------
snapshot_path = f'./logs/{args.exp}/'
maybe_mkdir(snapshot_path)
maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
logging.basicConfig(
    filename=os.path.join(snapshot_path, 'train.log'),
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))

# -------------------------
# Data Loader
# -------------------------
transforms_train = get_StrongAug(config.patch_size, 3, 0.7)

labeled_loader = DataLoader(
    DatasetAllTasks(
        split=args.split_labeled,
        transform=transforms_train,
        task=args.task,
        num_cls=config.num_cls
    ),
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)

eval_loader = DataLoader(
    DatasetAllTasks(
        split=args.split_eval,
        is_val=True,
        task=args.task,
        num_cls=config.num_cls,
        transform=torchvision.transforms.Compose([
            CenterCrop(config.patch_size),
            ToTensor()
        ])
    ),
    pin_memory=True
)

logging.info(f'{len(labeled_loader)} iterations per epoch (labeled)')

# -------------------------
# Teacher Model
# -------------------------
teacher_model = Unet(
    dimension=3,
    input_nc=1,
    output_nc=16,
    num_downs=4,
    ngf=16,
    num_classes=config.num_cls
).cuda()

# Load checkpoint
state = torch.load(args.teacher_ckpt, map_location='cuda')
logging.info(f"[INFO] Loading teacher weights from {args.teacher_ckpt}")
teacher_model.load_state_dict(state, strict=False)

loss_fn = DC_and_CE_loss()

# -------------------------
# === Stage 1: Finetune All Layers ===
# -------------------------
logging.info("==== Stage 1: Finetune ALL layers ====")
for p in teacher_model.parameters():
    p.requires_grad = True

optimizer = optim.Adam(
    teacher_model.parameters(),
    lr=args.base_lr
)

best_eval = 0.0
best_epoch = 0

for epoch_num in range(args.max_epoch_stage1):
    teacher_model.train()
    loss_list = []
    for batch in tqdm(labeled_loader):
        optimizer.zero_grad()
        image, label = fetch_data(batch)
        label = label.long()
        logits, _ = teacher_model(image)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    mean_loss = np.mean(loss_list)
    writer.add_scalar('stage1/loss', mean_loss, epoch_num)
    logging.info(f'[Stage1] epoch {epoch_num} train loss: {mean_loss}')
    optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch_stage1, args.base_lr, 0.9)

# Save checkpoint after Stage1
save_path = os.path.join(snapshot_path, 'ckpts/best_teacher_stage1.pth')
torch.save({'state_dict': teacher_model.state_dict()}, save_path)
logging.info(f"[Stage1] Saved model to {save_path}")

# -------------------------
# === Stage 2: Freeze Backbone, train proj_to_classes ===
# -------------------------
logging.info("==== Stage 2: Freeze backbone, train projection layer only ====")
for name, param in teacher_model.named_parameters():
    if "proj_to_classes" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.Adam(
    teacher_model.proj_to_classes.parameters(),
    lr=args.base_lr * 2   # projection层可以用稍大lr
)

for epoch_num in range(args.max_epoch_stage2):
    teacher_model.train()
    loss_list = []
    for batch in tqdm(labeled_loader):
        optimizer.zero_grad()
        image, label = fetch_data(batch)
        label = label.long()
        logits, _ = teacher_model(image)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    mean_loss = np.mean(loss_list)
    writer.add_scalar('stage2/loss', mean_loss, epoch_num)
    logging.info(f'[Stage2] epoch {epoch_num} train loss: {mean_loss}')
    optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch_stage2, args.base_lr * 2, 0.9)

    # Validation
    dice_list = [[] for _ in range(config.num_cls - 1)]
    teacher_model.eval()
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            image, gt = fetch_data(batch)
            logits, _ = teacher_model(image)
            pred = torch.argmax(softmax(logits), dim=1)
            for c in range(1, config.num_cls):
                intersect = torch.sum((pred == c) & (gt == c)).float()
                union = torch.sum((pred == c) | (gt == c)).float()
                dice = (2 * intersect + 1e-5) / (union + intersect + 1e-5)
                dice_list[c - 1].append(dice.cpu().item())

    dice_mean = [np.mean(dl) for dl in dice_list]
    mean_dice = np.mean(dice_mean)
    writer.add_scalar('stage2/val_dice', mean_dice, epoch_num)
    logging.info(f'[Stage2] epoch {epoch_num} val dice: {mean_dice}, {dice_mean}')

    if mean_dice > best_eval:
        best_eval = mean_dice
        best_epoch = epoch_num      
        save_path = os.path.join(snapshot_path, 'ckpts/best_teacher_final.pth')
        torch.save({'state_dict': teacher_model.state_dict()}, save_path)
        logging.info(f"[Stage2] Saved BEST model to {save_path}")

writer.close()
