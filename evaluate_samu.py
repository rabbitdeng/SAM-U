import os
import random

import PIL.Image as Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import colors

from segment_anything import SamPredictor

from eval_function import Fmeasure_calu
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import smeasure, dc, entropy
from test_uncertainty import ece_binary
import seaborn as sns

sns.set(font_scale=2.0)
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



# print(masks[0]["segmentation"])
# img = Image.fromarray(masks[0]["segmentation"])
# img.show()
def read_mask(mask_path):
    gt = Image.open(os.path.join(mask_path)).convert("L")
    gt = np.asarray(gt, dtype="uint8")
    return gt


def eval(input_dir, output_dir, gt_dir, predictor, generator, with_prompt=True):
    files_list = os.listdir(input_dir)
    pbar = files_list
    dice_score = []
    ece_score = []
    sm_score = []
    fm_score = []
    for i, item in enumerate(pbar):
        gt = read_mask(os.path.join(gt_dir, item.replace("img", "msk")))
        input_path = os.path.join(input_dir, item)
        gt = np.where(gt > 254, 0, 1).astype("uint8")
        if with_prompt:
            col, row = np.nonzero(gt)
            prompt = np.array([row[0], col[0], row[-1], col[-1]])

            aug_prompt = prompt_aug(prompt=prompt, target=gt, aug=False)
            with torch.no_grad():
                mask = masking(input_path, aug_prompt, predictor)
            mask = mask.squeeze(0)
            mask = torch.sigmoid(torch.from_numpy(mask)).cpu().numpy()
            mask = np.where(mask >= 0.5, True, False)
            dice = dc(mask, gt)
            ece = ece_binary(mask, gt)
        else:
            input = cv2.imread(input_path)
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            masks = generator.generate(input)
            max_dc = 0.0
            for sample in masks:
                sample_dc = dc(sample["segmentation"], gt)
                if sample_dc > max_dc:
                    mask = sample["segmentation"]
                    max_dc = sample_dc
            # limited_mask = torch.sigmoid(mask).cpu().detach().numpy().copy().squeeze(0)
            dice = dc(mask, gt)
            ece = ece_binary(mask, gt)
        sm = smeasure(mask,
                      gt)

        fm = Fmeasure_calu(mask,
                           gt, threshold=0.5)
        ece_score.append(ece)
        dice_score.append(dice)
        sm_score.append(sm)
        fm_score.append(fm)
        print(f'{item}:dice={dice},ece={ece},sm = {sm},fm = {fm}')
    dice_score = np.array(dice_score)
    ece_score = np.array(ece_score)
    sm_score = np.array(sm_score)
    fm_score = np.array(fm_score)
    return dice_score.mean(axis=0), ece_score.mean(axis=0), sm_score.mean(axis=0), fm_score.mean(axis=0)


def eval_aug(input_dir, output_dir, gt_dir, predictor, generator, with_prompt=True):
    files_list = os.listdir(input_dir)
    pbar = files_list
    dice_score = []
    ece_score = []
    sm_score = []
    fm_score = []
    for i, item in enumerate(pbar):
        gt = read_mask(os.path.join(gt_dir, item.replace("img", "msk")))
        input_path = os.path.join(input_dir, item)
        gt = np.where(gt > 254, 0, 1).astype("uint8")
        if with_prompt:
            aug_num = 5
            pred_list = []
            col, row = np.nonzero(gt)
            prompt = np.array([row[0], col[0], row[-1], col[-1]])
            for i in range(aug_num):
                aug_prompt = prompt_aug(prompt=prompt, target=gt, aug=True)
                with torch.no_grad():
                    mask = masking(input_path, aug_prompt, predictor)
                # mask = mask.squeeze(0)
                pred_list.append(torch.from_numpy(mask))
            y = torch.cat(pred_list, dim=0)
            mask = torch.mean(y, dim=0)
            mask = torch.sigmoid(mask).cpu().numpy()
            mask = np.where(mask >= 0.5, True, False)
            dice = dc(mask, gt)
            ece = ece_binary(mask, gt)
        else:
            input = cv2.imread(input_path)
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            masks = generator.generate(input)
            max_dc = 0.0
            for sample in masks:
                sample_dc = dc(sample["segmentation"], gt)
                if sample_dc > max_dc:
                    mask = sample["segmentation"]
                    max_dc = sample_dc
            # limited_mask = torch.sigmoid(mask).cpu().detach().numpy().copy().squeeze(0)
            dice = dc(mask, gt)
            ece = ece_binary(mask, gt)
        sm = smeasure(mask,
                      gt)

        fm = Fmeasure_calu(mask,
                           gt, threshold=0.5)
        ece_score.append(ece)
        dice_score.append(dice)
        sm_score.append(sm)
        fm_score.append(fm)
        print(f'{item}:dice={dice},ece={ece},sm = {sm},fm = {fm}')
    dice_score = np.array(dice_score)
    ece_score = np.array(ece_score)
    sm_score = np.array(sm_score)
    fm_score = np.array(fm_score)
    return dice_score.mean(axis=0), ece_score.mean(axis=0), sm_score.mean(axis=0), fm_score.mean(axis=0)


def masking(image_path, box_prompt, predictor):
    input = cv2.imread(image_path)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    # gaussian_noise_sigma = 0.05
    # noise_add = np.random.normal(0, gaussian_noise_sigma * 255, input.shape)
    # input = input + noise_add
    # input = np.clip(input, 0, 255).astype("uint8")
    # c = Image.fromarray(input)
    # c.show()
    predictor.set_image(input)
    masks, _, _ = predictor.predict(box=box_prompt, multimask_output=False, return_logits=True)
    # for mask_item in masks:
    #    mask+=mask_item["segmentation"]
    return masks


def prompt_aug(prompt, target, aug=True):
    target_h, target_w = target.shape
    scale_factor = 32
    if aug:
        left_x = prompt[0] - 25 + random.randint(-target_w / scale_factor, target_w / scale_factor)
        left_y = prompt[1] - 25 + random.randint(-target_h / scale_factor, target_h / scale_factor)
        right_x = prompt[2] + 25 + random.randint(-target_w / scale_factor, target_w / scale_factor)
        right_y = prompt[3] + 25 + random.randint(-target_h / scale_factor, target_h / scale_factor)
    else:
        left_x = prompt[0] - 25
        left_y = prompt[1] - 25
        right_x = prompt[2] + 25
        right_y = prompt[3] + 25
    if left_x < 0:
        left_x = 0
    elif left_x > target_w:
        left_x = target_w
    else:
        pass
    if left_y < 0:
        left_y = 0
    elif left_y > target_h:
        left_y = target_h
    else:
        pass
    if right_x < 0:
        right_x = 0
    elif right_x > target_w:
        right_x = target_w
    else:
        pass
    if right_y < 0:
        right_y = 0
    elif right_y > target_h:
        right_y = target_h
    else:
        pass

    return np.array([left_x, left_y, right_x, right_y])


if __name__ == "__main__":
    set_seed(42)
    output_mode = "binary_mask"
    image_path01 = "REFUGE_resize/img/g0002_1_img.jpg"
    image_path02 = "REFUGE_resize/img/g0002_1_img.jpg"
    image_path03 = "REFUGE_resize/img/V0360_0_img.jpg"
    mask_path = "REFUGE_resize/msk/g0025_1_msk.jpg"
    # im = cv2.imread("_uncertainty.png")
    model_type = "h"
    if model_type == "b":
        sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b.pth")
    elif model_type == "l":
        sam = sam_model_registry["vit_l"](checkpoint="checkpoints/sam_vit_l_0b3195.pth")
    elif model_type == "h":
        sam = sam_model_registry["default"](checkpoint="checkpoints/sam_default.pth")
    sam.to("cuda:0")
    predictor = SamPredictor(sam)
    generator = SamAutomaticMaskGenerator(sam)

    # dicescore, ecescore, smscore, fmscore = eval(input_dir="cofe_lq_img_101", output_dir="output",
    #                                                  gt_dir="REFUGE_resize/msk",
    #                                                  predictor=predictor, generator=generator, with_prompt=True)
    # print(f"dice: {dicescore} ece:{ecescore} sm:{smscore} fm:{fmscore}")
    gt = read_mask(mask_path)
    col, row = np.nonzero(~gt)
    gt = np.where(gt > 254, 0, 1).astype("uint8")
    prompt = np.array([row[0], col[0], row[-1], col[-1]])
    box_num = 20
    pred_list = []
    for i in range(box_num):
        aug_prompt = prompt_aug(prompt=prompt, target=gt)
        with torch.no_grad():
            mask = masking(image_path01, aug_prompt, predictor)
        img = Image.fromarray(np.where(mask.squeeze(0) > 0.5, True, False))
        print(i)
        img.save(f"result{i}.png")
        pred_list.append(torch.from_numpy(mask).unsqueeze(0))
    y = torch.cat(pred_list, dim=1)
    y = torch.mean(torch.sigmoid(y), dim=1)
    uncertainty = entropy(y)  # 4numof tta
    ece = ece_binary(np.where(mask.squeeze(0) > 0.5, True, False), gt)
    pred = np.where(y.detach().cpu().numpy().squeeze(0) > 0.5, True, False)
    print(f"ece = {ece}")
    uncertainty = (uncertainty.squeeze(0))
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    norm = colors.Normalize(vmin=0, vmax=1)
    # 定义横纵坐标的刻度
    # ax.set_yticks(range(len(yLabel)))
    # ax.set_yticklabels(yLabel, fontproperties=font)
    # ax.set_xticks(range(len(xLabel)))
    # ax.set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择hot
    plt.axis("off")
    img = Image.fromarray(pred)
    img.save("predg002vith.png")
    im = ax.imshow(uncertainty, cmap=plt.cm.jet, norm=norm)
    # 增加右侧的颜色刻度条
    #plt.colorbar(im)
    # 增加标题

    # show
    plt.savefig("g002normvith.png", bbox_inches='tight', pad_inches=0.0)
    # plt.show()
