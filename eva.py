import os
import random

import torch
import torch.nn.functional as F
import math
import PIL.Image as Image
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from segment_anything import build_sam, SamAutomaticMaskGenerator,sam_model_registry
from tqdm import tqdm
#
# print(masks[0]["segmentation"])
# img = Image.fromarray(masks[0]["segmentation"])
# img.show()
def read_mask(mask_path):
    gt = Image.open(os.path.join(mask_path)).convert("L")
    gt = np.asarray(gt,dtype="uint8")
    return gt

def Uentropy(logits,c):
    # c = 4
    # logits = torch.randn(1, 4, 240, 240,155).cuda()
    pc = F.sigmoid(logits)  # 1 4 240 240 155
    logits = torch.log(logits)  # 1 4 240 240 155
    u_all = -pc * logits / math.log(c)
    NU = torch.sum(u_all[:,1:u_all.shape[1],:,:], dim=1)
    return NU


def iou_score(mask,gt):
    mask = np.where(mask>0.5,True,False)
    mask = np.where(mask > 0.5, True, False)
    intersection = np.logical_and(mask, gt)
    union = np.logical_or(mask, gt)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
def eval(input_dir,output_dir,gt_dir ,generator):
    files_list = os.listdir(input_dir)
    pbar = files_list
    score_list = []
    for i,item in enumerate(pbar):
        gt = read_mask(os.path.join(gt_dir,item))
        input_path = os.path.join(input_dir,item)
        input = cv2.imread(input_path)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        masks = generator.generate(input)
        iou_list=[]
        for k in range(len(masks)):
            iou = iou_score(masks[k]["segmentation"],gt)
            iou_list.append(iou)
        iou_list = np.array(iou_list)
        iou_max = iou_list.max()
        score_list.append(iou_max)
        print(f"iou max of {item}:{iou_max}")
    score_list = np.array(score_list)
    return score_list.mean(axis=0)

def masking(image_path,box_prompt,predictor):
    input = cv2.imread(image_path)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    predictor.set_image(input)
    masks, _, _ = predictor.predict( box = box_prompt,   multimask_output=False,return_logits=True)
    #for mask_item in masks:
    #    mask+=mask_item["segmentation"]
    return masks

def prompt_aug(prompt,low_bound,high_bound):
    left_x = prompt[0]+ random.randint(low_bound,high_bound)
    left_y = prompt[1] + random.randint(low_bound,high_bound)
    right_x = prompt[2] + random.randint(low_bound,high_bound)
    right_y = prompt[3]+ random.randint(low_bound,high_bound)
    return np.array([left_x,left_y,right_x,right_y])
if __name__ == "__main__":
    df = pd.read_excel("fives/Quality Assessment.xlsx")
    print(df.head())

    output_mode = "binary_mask"
    image_path = "img/cxk.jpg"
    from segment_anything import build_sam, SamPredictor
    sam = sam_model_registry["default"](checkpoint="checkpoints/sam_default.pth")
    predictor = SamPredictor(sam)
    prompt = np.array([300,300,2500,2500])
    aug_num = 5
    pred_list = []
    for i in range(aug_num):
        aug_prompt = prompt_aug(prompt=prompt,low_bound=-100,high_bound=100)
        mask = masking(image_path,aug_prompt,predictor)
        img = Image.fromarray(np.where(mask.squeeze(0)>0.5,True,False))
        print(i)
        img.save(f"result{i}.png")
        pred_list.append(torch.sigmoid(torch.from_numpy(mask).unsqueeze(0)))
    y = torch.cat(pred_list,dim=1)
    uncertainty = Uentropy(y, 2).squeeze(0).detach().to('cpu').numpy()  # 4numof tta
    unimg = Image.fromarray(uncertainty*255)
    unimg.show()
    unimg.save("uncertainty.png")
    #dicescore = eval(input_dir="fives/train/Original",output_dir="output",gt_dir="fives/train/Ground truth",generator=generator)
    #print(f"dice without prompt: {dicescore}")

