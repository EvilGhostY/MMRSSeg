import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import math
import numbers
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import maximum_filter
import matplotlib.patches as mpatches
from PIL import Image
import random

CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = ORIGIN_IMG_SIZE
TEST_IMG_SIZE = ORIGIN_IMG_SIZE

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, dsm, ir):
        assert img.size == mask.size and img.size == dsm.size and img.size == ir.size
        for t in self.transforms:
            img, mask, dsm, ir = t(img, mask, dsm, ir)
        return img, mask, dsm, ir

class RandomScale(object):
    def __init__(self, scale_list=[0.75, 1.0, 1.25], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask, dsm, ir):
        oh, ow = img.size
        scale_amt = 1.0
        if self.mode == 'value':
            scale_amt = np.random.choice(self.scale_list, 1)
        elif self.mode == 'range':
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])
        h = int(scale_amt * oh)
        w = int(scale_amt * ow)
        return img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST), dsm.resize((w, h), Image.BICUBIC), ir.resize((w, h), Image.BICUBIC)

class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size=512, ignore_index=12, nopad=True):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, dsm, ir, centroid=None):
        assert img.size == mask.size and img.size == dsm.size and img.size == ir.size
        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return img, mask, dsm, ir

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                dsm = ImageOps.expand(dsm, border=border, fill=self.pad_color)
                ir = ImageOps.expand(ir, border=border, fill=self.pad_color)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), dsm.crop((x1, y1, x1 + tw, y1 + th)), ir.crop((x1, y1, x1 + tw, y1 + th))

class SmartCropV1(object):
    def __init__(self, crop_size=512,
                 max_ratio=0.75,
                 ignore_index=12, nopad=False):
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask, dsm, ir):
        assert img.size == mask.size and img.size == dsm.size and img.size == ir.size
        count = 0
        while True:
            img_crop, mask_crop, dsm_crop, ir_crop = self.crop(img.copy(), mask.copy(), dsm.copy(), ir.copy())
            count += 1
            labels, cnt = np.unique(np.array(mask_crop), return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.max_ratio:
                break
            if count > 10:
                break

        return img_crop, mask_crop, dsm_crop, ir_crop

def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform, additional_targets={'dsm': 'mask', 'ir': 'image'})


def train_aug(img, mask, dsm, ir):
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75,
                                    ignore_index=len(CLASSES), nopad=False)])
    img, mask, dsm, ir = crop_aug(img, mask, dsm, ir)
    img, mask, dsm, ir = np.array(img), np.array(mask), np.array(dsm), np.array(ir)
    if img.shape[2] == 2:
        img = np.stack((img[:, :, 0], img[:, :, 1], ir), axis=2)

    aug = get_training_transform()(image=img.copy(), mask=mask.copy(), dsm = dsm.copy())
    img, mask, dsm = aug['image'], aug['mask'], aug['dsm']
    ir = img[:,:,2]
    img = img[:,:,:2]

    return img, mask, dsm, ir


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform, additional_targets={'dsm': 'mask', 'ir': 'image'})


def val_aug(img, mask, dsm, ir):

    img, mask, dsm, ir = np.array(img), np.array(mask), np.array(dsm), np.array(ir)
    if img.shape[2] == 2:
        img = np.stack((img[:, :, 0], img[:, :, 1], ir), axis=2)

    aug = get_val_transform()(image=img.copy(), mask=mask.copy(), dsm=dsm.copy())

    img, mask, dsm = aug['image'], aug['mask'], aug['dsm']
    ir = img[:, :, 2]
    img = img[:, :, :2]

    return img, mask, dsm, ir


class VaihingenDataset(Dataset):
    def __init__(self, data_root='data/vaihingen/test', mode='val', img_dir='images', mask_dir='masks', dsm_dir='dsms',
                 img_suffix='.tif', mask_suffix='.png', dsm_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dsm_dir = dsm_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.dsm_suffix = dsm_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir, dsm_dir=self.dsm_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask, dsm = self.load_img_and_mask(index)
            img_np = np.array(img)
            ir = Image.fromarray(img_np[:, :, 2].squeeze())
            img = Image.fromarray(img_np[:, :, :2])
            if self.transform:
                img, mask, dsm, ir = self.transform(img, mask, dsm, ir)
        else:
            img, mask, dsm = self.load_mosaic_img_and_mask(index)
         #   img, mask, dsm = self.load_img_and_mask(index)
            img_np = np.array(img)
            ir = Image.fromarray(img_np[:, :, 2].squeeze())
            img = Image.fromarray(img_np[:, :, :2])
            if self.transform:
                img, mask, dsm, ir = self.transform(img, mask, dsm, ir)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        ir = torch.from_numpy(ir).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        dsm = torch.from_numpy(dsm).unsqueeze(0).float()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, ir=ir, dsm=dsm, gt_semantic_seg=mask)

        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir, dsm_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        dsm_filename_list = os.listdir(osp.join(data_root, dsm_dir))
        assert len(img_filename_list) == len(mask_filename_list) and len(img_filename_list) == len(dsm_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        dsm_name = osp.join(self.data_root, self.dsm_dir, img_id + self.dsm_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        dsm = Image.open(dsm_name).convert('L')
        return img, mask, dsm

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a, dsm_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b, dsm_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c, dsm_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d, dsm_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a, dsm_a = np.array(img_a), np.array(mask_a), np.array(dsm_a)
        img_b, mask_b, dsm_b = np.array(img_b), np.array(mask_b), np.array(dsm_b)
        img_c, mask_c, dsm_c = np.array(img_c), np.array(mask_c), np.array(dsm_c)
        img_d, mask_d, dsm_d = np.array(img_d), np.array(mask_d), np.array(dsm_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

#        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
#        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
#        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
#        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        # 创建包含 additional_targets 的 Compose 裁剪操作
        random_crop_a = albu.Compose(
            [albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])],
            additional_targets={
                "dsm_a": "mask"
            }
        )
        random_crop_b = albu.Compose(
            [albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])],
            additional_targets={
                "dsm_b": "mask"
            }
        )
        random_crop_c = albu.Compose(
            [albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])],
            additional_targets={
                "dsm_c": "mask"
            }
        )
        random_crop_d = albu.Compose(
            [albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])],
            additional_targets={
                "dsm_d": "mask"
            }
        )


        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy(), dsm_a=dsm_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy(), dsm_b=dsm_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy(), dsm_c=dsm_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy(), dsm_d=dsm_d.copy())

        img_crop_a, mask_crop_a, dsm_crop_a = croped_a['image'], croped_a['mask'], croped_a['dsm_a']
        img_crop_b, mask_crop_b, dsm_crop_b = croped_b['image'], croped_b['mask'], croped_b['dsm_b']
        img_crop_c, mask_crop_c, dsm_crop_c = croped_c['image'], croped_c['mask'], croped_c['dsm_c']
        img_crop_d, mask_crop_d, dsm_crop_d = croped_d['image'], croped_d['mask'], croped_d['dsm_d']


        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        top_dsm = np.concatenate((dsm_crop_a, dsm_crop_b), axis=1)
        bottom_dsm = np.concatenate((dsm_crop_c, dsm_crop_d), axis=1)
        dsm = np.concatenate((top_dsm, bottom_dsm), axis=0)

        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        dsm = np.ascontiguousarray(dsm)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        dsm = Image.fromarray(dsm)

        return img, mask, dsm


def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.tif')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')