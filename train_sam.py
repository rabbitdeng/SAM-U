# coding:utf-8
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from PIL import Image
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip,GaussNoise
import os
from eva import Uentropy,iou_score
from learnerable_seg import PromptSAM
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from scheduler import PolyLRScheduler
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

import random

parser = argparse.ArgumentParser("Learnable prompt")
parser.add_argument("--image", type=str, default="img/1_A.png",
                    help="path to the image that used to train the model")
parser.add_argument("--mask_path", type=str, default="fives/test/Ground truth/1_A.png",
                    help="path to the mask file for training")
parser.add_argument("--epoch", type=int, default=1000,
                    help="training epochs")
parser.add_argument("--checkpoint", type=str, default="checkpoints/sam_default.pth",
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="vit_h", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, default="./checkpoints",
                    help="save the weights of the model")
parser.add_argument("--num_classes", type=int, default=1)
parser.add_argument("--mix_precision", action="store_true", default=False,
                    help="whether use mix precison training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--optimizer", default="adam", type=str,
                    help="optimizer used to train the model")
parser.add_argument("--weight_decay", default=5e-4, type=float,
                    help="weight decay for the optimizer")
parser.add_argument("--momentum", default=0.99, type=float,
                    help="momentum for the sgd")


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


def main(args):
    TTA_num = 20
    img_path = args.image
    mask_path = args.mask_path
    epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    momentum = args.momentum
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes
    model = PromptSAM(model_name, checkpoint=checkpoint, num_classes=num_classes)
    img = Image.open(img_path).convert("RGB")
    img = np.asarray(img)
    mask = Image.open(mask_path).convert("L")
    mask = np.asarray(mask)
    # pixel_mean=[123.675, 116.28, 103.53],
    # pixel_std=[58.395, 57.12, 57.375],
    # pixel_mean = np.array(pixel_mean) / 255
    # pixel_std = np.array(pixel_std) / 255
    pixel_mean = [0.5] * 3
    pixel_std = [0.5] * 3
    lr = args.lr
    transform = Compose(
        [
            #HorizontalFlip(),
            #VerticalFlip(),
            Resize(1024, 1024),
            Normalize(mean=pixel_mean, std=pixel_std)
        ]
    )
    ttatransform = Compose(
        [
            GaussNoise(always_apply=True),
            Resize(1024, 1024),
            Normalize(mean=pixel_mean, std=pixel_std)
        ]
    )
    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if optimizer == "adam":
        optim = opt.Adam([{"params": model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = opt.SGD([{"params": model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=True)
    loss_dice = DiceLoss(mode="binary")
    loss_ce = nn.CrossEntropyLoss()
    loss_focal = FocalLoss(mode="binary")
    scheduler = PolyLRScheduler(optim, num_images=1, batch_size=1, epochs=epochs,gamma=0.9999)
    best_iou = 0
    for epoch in range(epochs):
        optim.zero_grad()

        aug_data = transform(image=img, mask=mask)
        tta = ttatransform(image=img, mask=mask)
        x = aug_data["image"]
        target = aug_data["mask"]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device,dtype = torch.float32)
        target = torch.from_numpy(target).unsqueeze(0).to(device, dtype=torch.int) //255
        x = x.to(device)
        model.train()
        with autocast(enabled=True):
            pred = model(x)
            loss = loss_dice(pred, target)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        pred = torch.sigmoid(pred)
        pred = pred.squeeze(0).detach().to('cpu').numpy()

        target = target.detach().to('cpu').numpy()
        #print(target.max())
        iou = iou_score(pred, target)
        scheduler.step()
        pred_tta =  []
        print("epoch-{}:{} iou:{}".format(epoch, loss, iou))
        if iou > best_iou:
            best_iou = iou
            torch.save(
                model.state_dict(), os.path.join(save_path, "sam_uncertaintyaug_{}.pth".format(model_name))
            )
    model.eval()
    for j in range(TTA_num):
        xtta = tta["image"]
        ttta = tta["mask"]
        xtta = torch.from_numpy(xtta).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
        ttta = torch.from_numpy(ttta).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
        xtta = xtta.to(device)
        with autocast(enabled=True):
            pred_ = model(xtta)
        pred_tta.append(pred_)
    # y = torch.zeros_like(pred_)
    # for j in range(TTA_num):
    #    y +=pred_tta[j]
    img = Image.fromarray(
        torch.where(torch.sigmoid(pred_) > 0.5, 1.0, 0.0).squeeze(0).squeeze(0).detach().to('cpu').numpy() * 255,
        mode="L")
    img.save("pred.png")
    img.show()
    y = torch.cat(pred_tta, dim=1)
    uncertainty = Uentropy(y, 2).squeeze(0).detach().to('cpu').numpy()  # 4numof tta
    unimg = Image.fromarray(uncertainty * 255.0, mode="L")
    unimg.show()
    unimg.save("uncertainty.png")


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(42)
    main(args)
