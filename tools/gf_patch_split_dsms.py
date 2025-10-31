import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Building label values: [B, G, R] (BGR order)
# ===== 定义每个类别的颜色 =====
Shadow_Background = np.array([0, 0, 0])        # label 0
Water            = np.array([200, 150, 0])     # label 1
Tree             = np.array([0, 100, 0])       # label 2
Grass            = np.array([113, 179, 60])    # label 3
Road             = np.array([212, 255, 127])   # label 4
Sand             = np.array([32, 165, 218])    # label 5
Soil_Bareland    = np.array([63, 133, 205])    # label 6
Building         = np.array([130, 0, 75])      # label 7
Cement_Road      = np.array([200, 200, 200])   # label 8
Other            = np.array([250, 250, 255])   # label 9
num_classes = 10


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="data\\test_images")
    parser.add_argument("--dsm-dir", default="data\\test_ads")
    parser.add_argument("--mask-dir", default="data\\test_masks")
    parser.add_argument("--output-img-dir", default="data\\test\\images")
    parser.add_argument("--output-ir-dir", default="data\\test\\irs")
    parser.add_argument("--output-dsm-dir", default="data\\test\\dsms")
    parser.add_argument("--output-mask-dir", default="data\\test\\masks")
    parser.add_argument("--eroded", action='store_true')
    parser.add_argument("--gt", action='store_true')  # output RGB mask
    parser.add_argument("--rgb-image", action='store_true', default=True)  # use Potsdam RGB format images
    parser.add_argument("--mode", type=str, default='val')
    parser.add_argument("--val-scale", type=float, default=1.0)  # ignore
    parser.add_argument("--split-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    return parser.parse_args()


def get_img_mask_padded(image, ir_n, dsm_n, mask, patch_size, mode):
    img, ir, dsm,  mask = np.array(image), np.array(ir_n), np.array(dsm_n), np.array(mask)
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad
    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                               border_mode=0, value=[0, 0, 0])(image=img)
    pad_ir = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                               border_mode=0, value=[0, 0, 0])(image=ir)
    pad_dsm = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                               border_mode=0, value=[0, 0, 0])(image=dsm)
    if mode == 'train':
        pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right')(image=img)
        pad_ir = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right')(image=ir)
        pad_dsm = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right')(image=dsm)

    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                                border_mode=0, value=[0, 0, 0])(image=mask)
    img_pad, ir_pad, dsm_pad, mask_pad = pad_img['image'], pad_ir['image'], pad_dsm['image'], pad_mask['image']
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    ir_pad = cv2.cvtColor(np.array(ir_pad), cv2.COLOR_RGB2BGR)
    dsm_pad = cv2.cvtColor(np.array(dsm_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    return img_pad, ir_pad, dsm_pad, mask_pad


def pv2rgb(mask: np.ndarray) -> np.ndarray:
    """
    将语义分割标签图 (单通道 mask) 转换为 BGR 彩色图。
    类别编号对应:
        0: Shadow_Background
        1: Water
        2: Tree
        3: Grass
        4: Road
        5: Sand
        6: Soil_Bareland
        7: Building
        8: Cement_Road
        9: Other
    """
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # 映射规则 (B, G, R)
    mask_rgb[mask == 0] = [0, 0, 0]         # Shadow_Background
    mask_rgb[mask == 1] = [200, 150, 0]     # Water
    mask_rgb[mask == 2] = [0, 100, 0]       # Tree
    mask_rgb[mask == 3] = [113, 179, 60]    # Grass
    mask_rgb[mask == 4] = [212, 255, 127]   # Road
    mask_rgb[mask == 5] = [32, 165, 218]    # Sand
    mask_rgb[mask == 6] = [63, 133, 205]    # Soil_Bareland
    mask_rgb[mask == 7] = [130, 0, 75]      # Building
    mask_rgb[mask == 8] = [200, 200, 200]   # Cement_Road
    mask_rgb[mask == 9] = [250, 250, 255]   # Other

    return mask_rgb


def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]

    return mask


def rgb_to_2D_label(_label: np.ndarray) -> np.ndarray:
    """
    将BGR彩色标签图转换为单通道类别图（0~9）
    输入:
        _label: ndarray(H, W, 3)，BGR格式图像（如cv2.imread结果）
    输出:
        label_seg: ndarray(H, W)，每个像素为类别编号(0-9)
    """
    h, w, _ = _label.shape
    label_seg = np.zeros((h, w), dtype=np.uint8)

    # 逐类别匹配（BGR）
    label_seg[np.all(_label == Shadow_Background, axis=-1)] = 0
    label_seg[np.all(_label == Water, axis=-1)] = 1
    label_seg[np.all(_label == Tree, axis=-1)] = 2
    label_seg[np.all(_label == Grass, axis=-1)] = 3
    label_seg[np.all(_label == Road, axis=-1)] = 4
    label_seg[np.all(_label == Sand, axis=-1)] = 5
    label_seg[np.all(_label == Soil_Bareland, axis=-1)] = 6
    label_seg[np.all(_label == Building, axis=-1)] = 7
    label_seg[np.all(_label == Cement_Road, axis=-1)] = 8
    label_seg[np.all(_label == Other, axis=-1)] = 9

    return label_seg

def image_augment(image, ir, dsm, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    ir_list = []
    dsm_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]
    dsm_width, dsm_height = dsm.size[1], dsm.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]


    assert image_height == mask_height == dsm_height and image_width == mask_width == dsm_width
    if mode == 'train':
        h_vlip = RandomHorizontalFlip(p=1.0)
        v_vlip = RandomVerticalFlip(p=1.0)
        image_h_vlip, ir_h_vlip, dsm_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(ir.copy()), h_vlip(dsm.copy()), h_vlip(mask.copy())
        image_v_vlip, ir_v_vlip, dsm_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(ir.copy()), v_vlip(dsm.copy()), v_vlip(mask.copy())

        image_list_train = [image, image_h_vlip, image_v_vlip]
        ir_list_train = [ir, ir_h_vlip, ir_v_vlip]
        dsm_list_train = [dsm, dsm_h_vlip, dsm_v_vlip]
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        for i in range(len(image_list_train)):
            image_tmp, ir_tmp, dsm_tmp, mask_tmp = get_img_mask_padded(image_list_train[i],ir_list_train[i],dsm_list_train[i], mask_list_train[i], patch_size, mode)
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            ir_list.append(ir_tmp)
            dsm_list.append(dsm_tmp)
            mask_list.append(mask_tmp)
    else:
        rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        image, ir, dsm, mask = rescale(image.copy()), rescale(ir.copy()), rescale(dsm.copy()), rescale(mask.copy())
        image, ir, dsm, mask = get_img_mask_padded(image.copy(), ir.copy(), dsm.copy(), mask.copy(), patch_size, mode)
        mask = rgb_to_2D_label(mask.copy())

        image_list.append(image)
        ir_list.append(ir)
        dsm_list.append(dsm)
        mask_list.append(mask)
    return image_list, ir_list, dsm_list, mask_list

def car_aug(image, dsm, mask):
    assert image.shape[:2] == mask.shape == dsm.shape
    resize_crop_1 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.25), height=int(image.shape[1] * 1.25)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), dsm=dsm.copy(), mask=mask.copy())
    resize_crop_2 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.5), height=int(image.shape[1] * 1.5)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), dsm=dsm.copy(), mask=mask.copy())
    resize_crop_3 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.75), height=int(image.shape[1] * 1.75)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), dsm=dsm.copy(), mask=mask.copy())
    resize_crop_4 = albu.Compose([albu.Resize(width=int(image.shape[0] * 2.0), height=int(image.shape[1] * 2.0)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), dsm=dsm.copy(), mask=mask.copy())
    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), dsm=dsm.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), dsm=dsm.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), dsm=dsm.copy(), mask=mask.copy())
    image_resize_crop_1, dsm_resize_crop_1, mask_resize_crop_1 = resize_crop_1['image'], resize_crop_1['dsm'], resize_crop_1['mask']
    image_resize_crop_2, dsm_resize_crop_2, mask_resize_crop_2 = resize_crop_2['image'], resize_crop_2['dsm'], resize_crop_2['mask']
    image_resize_crop_3, dsm_resize_crop_3, mask_resize_crop_3 = resize_crop_3['image'], resize_crop_3['dsm'], resize_crop_3['mask']
    image_resize_crop_4, dsm_resize_crop_4, mask_resize_crop_4 = resize_crop_4['image'], resize_crop_4['dsm'], resize_crop_4['mask']
    image_vflip, dsm_vflip, mask_vflip = v_flip['image'], v_flip['dsm'], v_flip['mask']
    image_hflip, dsm_hflip, mask_hflip = h_flip['image'], h_flip['dsm'], h_flip['mask']
    image_rotate, dsm_rotate, mask_rotate = rotate_90['image'], rotate_90['dsm'], rotate_90['mask']
    image_list = [image, image_resize_crop_1, image_resize_crop_2, image_resize_crop_3,
                  image_resize_crop_4, image_vflip, image_hflip, image_rotate]
    dsm_list = [dsm, dsm_resize_crop_1, dsm_resize_crop_2, dsm_resize_crop_3,
                  dsm_resize_crop_4, dsm_vflip, dsm_hflip, dsm_rotate]
    mask_list = [mask, mask_resize_crop_1, mask_resize_crop_2, mask_resize_crop_3,
                 mask_resize_crop_4, mask_vflip, mask_hflip, mask_rotate]

    return image_list, dsm_list, mask_list


def patch_format(inp):
    (img_path, dsm_path, mask_path, imgs_output_dir, irs_output_dir, dsms_output_dir, masks_output_dir, eroded, gt, rgb_image,
     mode, val_scale, split_size, stride) = inp
    img_filename = os.path.basename(img_path)
    dsm_filename = os.path.basename(dsm_path)
    mask_filename = os.path.basename(mask_path)
    # print(img_filename)
    # print(mask_filename)
    if eroded:
        mask_path = mask_path + '.png'
    else:
        mask_path = mask_path + '.png'
    if rgb_image:
        img_path = img_path + '.tif'
    else:
        img_path = img_path + '.tif'
    irrgb = Image.open(img_path).convert('RGBA')
    ir_np = np.array(irrgb)[:, :, 3]  # IR
    img_np = np.array(irrgb)[:, :, 0:3]  # RGB
    # === 转回 PIL.Image 格式 ===
    ir = Image.fromarray(ir_np)  # 单通道灰度图
    img = Image.fromarray(img_np, mode='RGB')  # 三通道彩色图
    # print(img)
    dsm_path = dsm_path + '_AD.png'
    dsm = Image.open(dsm_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    if gt:
        mask_ = car_color_replace(mask.copy())
        out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))
        cv2.imwrite(out_origin_mask_path, mask_)
    # print(mask)
    # print(img_path)
    # print(img.size, mask.size)
    # img and mask shape: WxHxC
    image_list, ir_list, dsm_list, mask_list = image_augment(image=img.copy(), ir = ir.copy(), dsm=dsm.copy(), mask=mask.copy(), patch_size=split_size,
                                          val_scale=val_scale, mode=mode)
    assert img_filename == mask_filename == dsm_filename and len(image_list) == len(mask_list) == len(dsm_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        ir = ir_list[m]
        dsm = dsm_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] == dsm.shape[0] and img.shape[1] == mask.shape[1] == dsm.shape[1]
        if gt:
            mask = pv2rgb(mask.copy())

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile_cut = img[y:y + split_size, x:x + split_size]
                ir_tile_cut = ir[y:y + split_size, x:x + split_size]
                dsm_tile_cut = dsm[y:y + split_size, x:x + split_size]
                mask_tile_cut = mask[y:y + split_size, x:x + split_size]
                img_tile, ir_tile, dsm_tile, mask_tile = img_tile_cut, ir_tile_cut, dsm_tile_cut, mask_tile_cut

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                    and dsm_tile.shape[0] == split_size and dsm_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size\
                            and ir_tile.shape[0] == split_size and ir_tile.shape[1] == split_size:


                    out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_ir_path = os.path.join(irs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                    cv2.imwrite(out_ir_path, ir_tile)

                    out_dsm_path = os.path.join(dsms_output_dir, "{}_{}_{}.tif".format(dsm_filename, m, k))
                    cv2.imwrite(out_dsm_path, dsm_tile)

                    out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                    cv2.imwrite(out_mask_path, mask_tile)

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    dsms_dir = args.dsm_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    irs_output_dir = args.output_ir_dir
    dsms_output_dir = args.output_dsm_dir
    masks_output_dir = args.output_mask_dir
    eroded = args.eroded
    gt = args.gt
    rgb_image = args.rgb_image
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    img_paths_raw = glob.glob(os.path.join(imgs_dir, "*.tif"))
    img_paths = [p[:-4] for p in img_paths_raw]
    if rgb_image:
        img_paths = [p[:-4] for p in img_paths_raw]
    dsm_paths_raw = glob.glob(os.path.join(dsms_dir, "*.png"))
    dsm_paths = [p[:-7] for p in dsm_paths_raw]
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.png"))
    if eroded:
        mask_paths = [(p[:-4]) for p in mask_paths_raw]
    else:
        mask_paths = [p[:-4] for p in mask_paths_raw]
    img_paths.sort()
    dsm_paths.sort()
    mask_paths.sort()
    # print(img_paths[:10])
    # print(mask_paths[:10])

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(irs_output_dir):
        os.makedirs(irs_output_dir)
    if not os.path.exists(dsms_output_dir):
        os.makedirs(dsms_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir+'/origin')

    inp = [(img_path, dsm_path, mask_path, imgs_output_dir, irs_output_dir, dsms_output_dir, masks_output_dir, eroded, gt, rgb_image,
            mode, val_scale, split_size, stride)
           for img_path, dsm_path, mask_path in zip(img_paths, dsm_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


