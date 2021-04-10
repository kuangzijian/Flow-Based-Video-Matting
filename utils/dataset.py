from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging, os, re
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, masks_dir, org_dir, scale=1, mask_suffix=''):
        self.masks_dir = masks_dir
        self.org_dir = org_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.org_prefix = 'original_'
        self.gt_prefix = 'gt_'

        alphanum_key = lambda key: [int(re.split('_', key)[1].split('.')[0])]
        files = sorted(os.listdir(org_dir), key=alphanum_key)
        sizes = [Image.open(os.path.join(org_dir, f), 'r').size for f in files]
        self.size = max(sizes)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0].split('_')[1] for file in listdir(org_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_input_with_int_mask(cls, int_mask, org_img, scale):
        w, h = int_mask.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        int_mask = int_mask.resize((newW, newH)).convert('L')
        org_img = org_img.resize((newW, newH))

        int_mask_nd = np.array(int_mask)
        int_mask_nd = np.expand_dims(int_mask_nd, axis=2)
        org_img_nd = np.array(org_img)
        concat_img_nd = np.concatenate((int_mask_nd, org_img_nd), axis=2)

        # HWC to CHW
        img_trans = concat_img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + self.gt_prefix + idx + self.mask_suffix + '.*')
        org_file = glob(self.org_dir + self.org_prefix + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(org_file) == 1, \
            f'Either no original image or multiple original images found for the ID {idx}: {org_file}'

        mask = Image.open(mask_file[0]).resize(self.size).convert('L')
        org_img = Image.open(org_file[0]).resize(self.size)

        img = self.preprocess(org_img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
