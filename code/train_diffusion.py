import os
import sys
import logging
from tqdm import tqdm
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='synapse', required=True)
parser.add_argument('--exp', type=str, default='diffusion')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, required=True)
parser.add_argument('-su', '--split_unlabeled', type=str, required=True)
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--unsup_loss', type=str, default='w_ce+dice')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--mu', type=float, default=2.0)
parser.add_argument('-s', '--ema_w', type=float, default=0.99)
parser.add_argument('-r', '--mu_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--rampup_epoch', type=float, default=None) # 100
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--computer_graph', type=str2bool, default=False)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--teacher_ckpt', type=str, default=None, help='Path to pretrained teacher UNet weights')
args = parser.parse_args()
if args.debug:
    import time 
    import debugpy
    print('start debug')
    rank = int(os.environ.get("RANK", 0))
    debug_port = 6006 + rank

    debugpy.listen(("0.0.0.0", debug_port))
    print(f"Rank {rank} waiting for debugger on port {debug_port}...")
    debugpy.wait_for_client()
    time.sleep(2)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from DiffVNet.diff_vnet import DiffVNet
from utils import EMA, maybe_mkdir, batch_label_similarity, get_lr, fetch_data, GaussianSmoothing, seed_worker, poly_lr, print_func, sigmoid_rampup
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetAllTasks
from utils.config import Config
from data.StrongAug import get_StrongAug, ToTensor, CenterCrop
from DiffVNet.unet_teacher import Unet


config = Config(args.task)

# def custom_repr(self):
#     return f'{{Shape:{tuple(self.shape)}}} {original_repr(self)}'

# original_repr = torch.Tensor.__repr__
# torch.Tensor.__repr__ = custom_repr 

def get_current_mu(epoch):
    if args.mu_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.rampup_epoch is None:
            args.rampup_epoch = args.max_epoch
        return args.mu * sigmoid_rampup(epoch, args.rampup_epoch)
    else:
        return args.mu

def get_current_temperature(epoch):
    if args.mu_rampup:
        if args.rampup_epoch is None:
            args.rampup_epoch = args.max_epoch
        return args.temperature * (1.0 - sigmoid_rampup(epoch, args.rampup_epoch))
    else:
        return args.temperature
    
def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=DatasetAllTasks, repeat=None, is_training=True, unlabeled=False, task="", transforms_tr=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=task,
            num_cls=config.num_cls
        )
        return DataLoader(
            dst,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True
        )
    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=task,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    model = DiffVNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return model, optimizer


class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)

        self.last_dice = cur_dice

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)

        cur_diff = torch.pow(cur_diff, 1/5)

        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        return weights * self.num_cls


def save_pred_visualization(image, label_u, pred_teacher, pred_xi, pred_psi, pred_theta, save_dir, idx):
    os.makedirs(save_dir, exist_ok=True)
    img = image[0, 0].detach().cpu().numpy()
    if img.ndim == 3:  # 3D
        mid = img.shape[0] // 2
        img = img[mid]
        label_u = label_u[0, 0, mid].detach().cpu().numpy()
        pred_teacher = pred_teacher[0, 0, mid].detach().cpu().numpy()
        pred_xi = pred_xi[0, 0, mid].detach().cpu().numpy()
        pred_psi = pred_psi[0, 0, mid].detach().cpu().numpy()
        pred_theta = pred_theta[0, 0, mid].detach().cpu().numpy()
    else:  # 2D
        label_u = label_u[0, 0].detach().cpu().numpy()
        pred_teacher = pred_teacher[0, 0].detach().cpu().numpy()
        pred_xi = pred_xi[0, 0].detach().cpu().numpy()
        pred_psi = pred_psi[0, 0].detach().cpu().numpy()
        pred_theta = pred_theta[0, 0].detach().cpu().numpy()
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Image')
    axs[1].imshow(label_u, vmin=0, vmax=3)
    axs[1].set_title('GT')
    axs[2].imshow(pred_teacher, vmin=0, vmax=3)
    axs[2].set_title('Teacher')
    axs[3].imshow(pred_xi, vmin=0, vmax=3)
    axs[3].set_title('Xi')
    axs[4].imshow(pred_psi, vmin=0, vmax=3)
    axs[4].set_title('Psi')
    axs[5].imshow(pred_theta, vmin=0, vmax=3)
    axs[5].set_title('Theta')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'vis_{idx}.png'))
    plt.close()

def get_edge_mask(label, kernel_size=3):
    if label.dim() == 5:
        conv_fn = F.conv3d
        kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=label.device)
    else:
        conv_fn = F.conv2d
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=label.device)
    edge = torch.zeros_like(label, dtype=torch.float)
    max_label = int(label.max().item())
    for i in range(1, max_label+1):
        mask = (label == i).float()
        dilated = conv_fn(mask, kernel, padding=kernel_size//2)
        edge += ((dilated > 0) & (dilated < kernel.numel())).float()
    return (edge > 0).float()


if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    transforms_train_labeled = get_StrongAug(config.patch_size, 3, 0.7)
    transforms_train_unlabeled = get_StrongAug(config.patch_size, 3, 0.7)

    if "mmwhs" not in args.task:
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, unlabeled=True)
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task, repeat=len(unlabeled_loader.dataset))
    else:
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task)
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, repeat=len(labeled_loader.dataset), unlabeled=True)
    eval_loader = make_loader(args.split_eval, task=args.task, is_training=False)



    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model, optimizer = make_model_all()
    teacher_model = Unet(
        dimension=3,
        input_nc=1,
        output_nc=16,
        num_downs=4,
        ngf=16,
        num_classes=config.num_cls
    ).cuda()
    if args.teacher_ckpt is not None:
        state = torch.load(args.teacher_ckpt, map_location='cuda')
        print(f"[INFO] Loading teacher weights from {args.teacher_ckpt}")
        teacher_model.load_state_dict(state["state_dict"], strict=True)
        for p in teacher_model.parameters():
            p.requires_grad = False
        for p in teacher_model.proj_to_classes.parameters():
            p.requires_grad = True


    diff = Difficulty(config.num_cls, accumulate_iters=50)

    deno_loss  = make_loss_function(args.sup_loss)
    sup_loss  = make_loss_function(args.sup_loss)
    unsup_loss  = make_loss_function(args.unsup_loss)
    kl_loss_fn = nn.KLDivLoss(reduction="mean")

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    mu = get_current_mu(0)
    warmup_epoch = 5
    full_consistency_epoch = 10
    best_eval = 0.0
    best_epoch = 0
    k = 0.2197  
    m = 0       
    print('[INFO] Start training with the following parameters:')
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_sup_list = []
        loss_diff_list = []
        loss_unsup_list = []
        loss_kl_list = []
        dice_teacher_all = []
        dice_xi_all = []
        dice_psi_all = []
        acc_teacher_all = []
        acc_xi_all = []
        acc_psi_all = []
        total_pixels = 0
        agree_pixels = 0
        hist_teacher = torch.zeros(config.num_cls, device="cuda")
        hist_xi = torch.zeros(config.num_cls, device="cuda")
        hist_psi = torch.zeros(config.num_cls, device="cuda")
        model.train()
        for batch_idx, (batch_l, batch_u) in enumerate(tqdm(zip(labeled_loader, unlabeled_loader))):
   
            for D_theta_name, D_theta_params in model.decoder_theta.named_parameters():
                if D_theta_name in model.denoise_model.decoder.state_dict().keys():
                    D_xi_params = model.denoise_model.decoder.state_dict()[D_theta_name]
                    D_psi_params = model.decoder_psi.state_dict()[D_theta_name]
                    if D_theta_params.shape == D_xi_params.shape:
                        D_theta_params.data = args.ema_w * D_theta_params.data + (1 - args.ema_w) * (D_xi_params.data + D_psi_params.data) / 2.0


            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            label_l = label_l.long()
            image_u, label_u = fetch_data(batch_u)  #   ------

            if args.mixed_precision:
                with autocast():
                    shp = (config.batch_size, config.num_cls)+config.patch_size

                    label_l_onehot = torch.zeros(shp).cuda()
                    label_l_onehot.scatter_(1, label_l, 1)
                    x_start = label_l_onehot * 2 - 1
                    x_t, t, noise = model(x=x_start, pred_type="q_sample")

                    student_logits, _ = model(x=x_t, step=t, image=image_l, pred_type="D_xi_l")

                    p_l_psi = model(image=image_l, pred_type="D_psi_l")

                    edge_mask = get_edge_mask(label_l)
                    edge_weight = 1.0 + 2.0 * edge_mask  

                    L_deno = deno_loss(student_logits, label_l)

                    weight_diff = diff.cal_weights(student_logits.detach(), label_l)
                    sup_loss.update_weight(weight_diff)
                    L_diff = (sup_loss(p_l_psi, label_l) * edge_weight).mean()

                    with torch.no_grad():
                        p_u_xi = model(image_u, pred_type="ddim_sample")
                        p_u_psi = model(image_u, pred_type="D_psi_l")
                        teacher_logit, _ = teacher_model(image_u)
                        teacher_prob = F.softmax(teacher_logit, dim=1)
                        teacher_confidence, _ = torch.max(teacher_prob, dim=1, keepdim=True)
                        confidence_threshold = 0.5 - 0.2 * (epoch_num / args.max_epoch)
                        confident_mask = (teacher_confidence > confidence_threshold).float()
                        p_teacher = teacher_prob.argmax(dim=1, keepdim=True) * confident_mask.long()
                        p_xi = F.softmax(p_u_xi, dim=1)
                        p_psi = F.softmax(p_u_psi, dim=1)
                        p_student = (p_xi + p_psi ) / 2.0
                        conf_teacher, _ = teacher_prob.max(dim=1, keepdim=True)
                        conf_xi, _ = p_xi.max(dim=1, keepdim=True)
                        conf_psi, _ = p_psi.max(dim=1, keepdim=True)
                        weight_sum = conf_teacher + conf_xi + conf_psi + 1e-8
                        w_teacher = conf_teacher / weight_sum
                        w_xi = conf_xi / weight_sum
                        w_psi = conf_psi / weight_sum
                        fused_prob = w_teacher * teacher_prob + w_xi * p_xi + w_psi * p_psi
                        pseudo_label = fused_prob.argmax(dim=1, keepdim=True)
                        consistency_loss = (F.mse_loss(p_xi, p_psi) + F.mse_loss(p_xi, teacher_prob) + F.mse_loss(p_psi, teacher_prob)) / 3


                    p_u_theta = model(image=image_u, pred_type="D_theta_u")
                    student_theta_log_prob = F.log_softmax(p_u_theta / get_current_temperature(epoch_num), dim=1)
                    teacher_prob_u = F.softmax(teacher_logit / get_current_temperature(epoch_num), dim=1)
                    teacher_prob_u = torch.clamp(teacher_prob_u, min=1e-6)

                    L_logits = kl_loss_fn(student_theta_log_prob, teacher_prob_u) * (get_current_temperature(epoch_num) ** 2)

                    L_u = unsup_loss(p_u_theta, pseudo_label.detach())
                    
                    alpha = 0.1 
                    loss = L_deno + L_diff +  L_u + mu * L_logits + alpha * consistency_loss

                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()

                if batch_idx == 0 and epoch_num % 1 == 0:
                    pred_teacher = teacher_prob.argmax(dim=1, keepdim=True)
                    pred_xi = p_xi.argmax(dim=1, keepdim=True)
                    pred_psi = p_psi.argmax(dim=1, keepdim=True)
                    pred_theta = p_u_theta.argmax(dim=1, keepdim=True)
                    save_pred_visualization(
                        image_u, label_u, pred_teacher, pred_xi, pred_psi, pred_theta,
                        save_dir=os.path.join(snapshot_path, 'vis_pred'), idx=epoch_num
                    )

            else:
                raise NotImplementedError
            
            loss_list.append(loss.item())
            loss_sup_list.append(L_deno.item())
            loss_diff_list.append(L_diff.item())
            loss_unsup_list.append(L_u.item())
            loss_kl_list.append(L_logits.item())
        mean_dice_teacher = np.mean(dice_teacher_all)
        mean_dice_xi = np.mean(dice_xi_all)
        mean_dice_psi = np.mean(dice_psi_all)

        mean_acc_teacher = np.mean(acc_teacher_all)
        mean_acc_xi = np.mean(acc_xi_all)
        mean_acc_psi = np.mean(acc_psi_all)
        
        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/deno', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/unsup', np.mean(loss_unsup_list), epoch_num)
        writer.add_scalar('loss/diff', np.mean(loss_diff_list), epoch_num)
        writer.add_scalar('loss/kl', np.mean(loss_kl_list), epoch_num)
        writer.add_scalars('class_weights', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_diff))), epoch_num)

        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | lr : {get_lr(optimizer)} | mu : {mu}')
        logging.info(f"     diff_w: {print_func(weight_diff)}")
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        mu = get_current_mu(epoch_num)

        if epoch_num % 1 == 0:

            dice_list = [[] for _ in range(config.num_cls-1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    p_u_theta = model(image, pred_type="D_theta_u")
                    del image

                    shp = (p_u_theta.shape[0], config.num_cls) + p_u_theta.shape[2:]
                    gt = gt.long()

                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    p_u_theta = torch.argmax(p_u_theta, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, p_u_theta, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            writer.add_scalar('val_dice', np.mean(dice_mean), epoch_num)
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()