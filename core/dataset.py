import os
from glob import glob
import logging
from PIL import Image
import cv2
import numpy as np
import albumentations as A
import torch
import tifffile
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision.transforms import Resize, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation


transform = A.Compose([
    A.augmentations.crops.transforms.RandomSizedCrop(min_max_height=[56, 224], height=448, width=448, p=0.3),
    A.augmentations.transforms.ChannelShuffle(p=0.3),
    A.augmentations.geometric.transforms.ShiftScaleRotate(p=0.3),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
    A.VerticalFlip(p=0.3),
    A.HorizontalFlip(p=0.3)
])

class CustomDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, boundaries_dir, mask_suffix='', boundary_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.boundaries_dir = boundaries_dir
        self.mask_suffix = mask_suffix
        self.boundary_suffix = boundary_suffix

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        assert len(self.ids) == len(os.listdir(masks_dir)), \
            f'Img count not equal to mask count!'
        assert len(self.ids) == len(os.listdir(boundaries_dir)), \
            f'Img count not equal to borders count!'
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_labels(cls, pil_img):
        label_np = np.array(pil_img)

        if len(label_np.shape) == 2:
            label_np = np.expand_dims(label_np, axis=0)

        if label_np.max() > 1:
            label_np = label_np / 255

        return label_np


    @classmethod
    def preprocess_imgs(cls, img_np):
        if len(img_np.shape) == 3:
            img_np = np.transpose(img_np, [2, 0, 1])

        # normalize to simple 0-1 range
        img_np = cv2.normalize(img_np, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return img_np
    
    @classmethod
    def preprocess_to_inference(cls, img_np):
        dim = 224
        img_np = cv2.resize(img_np, (dim, dim), interpolation=cv2.INTER_LANCZOS4)

        if len(img_np.shape) == 3:
            img_np = np.transpose(img_np, [2, 0, 1])

        # normalize to simple 0-1 range
        img_np = cv2.normalize(img_np, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return img_np

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        boundary_file = glob(self.boundaries_dir + idx + self.boundary_suffix + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(boundary_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {boundary_file}'
        # Read img and mask in rgb mode
        img = tifffile.imread(img_file[0])

        # Read mask and border in grayscale mode
        mask = Image.open(mask_file[0]).convert('L')
        boundary = Image.open(boundary_file[0]).convert('L')

        img = self.preprocess_imgs(img)
        img = np.transpose(img, [1, 2, 0])
        mask = self.preprocess_labels(mask)
        mask = np.transpose(mask, [1, 2, 0])
        boundary = self.preprocess_labels(boundary)
        boundary = np.transpose(boundary, [1, 2, 0])

        assert img.shape[0] == mask.shape[0] == img.shape[1] == mask.shape[1] == boundary.shape[1] == boundary.shape[1], \
            f'Image and mask {idx} should be the same size, but are img resolution: {img.shape[0]}, {img.shape[1]}, \
                mask resolution: {mask.shape[0]}, {mask.shape[1]}, boundary resolution: {boundary.shape[0]}, {boundary.shape[1]}'

        mask_and_boundary = np.concatenate((mask, boundary), axis=2)

        transformed = transform(image=img, mask=mask_and_boundary)
        transformed_image = transformed['image']
        transformed_mask_and_boundary = transformed['mask']

        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        img_tensor = torch.permute(img_tensor, (2, 0, 1))
        mask_and_boundary_tensor = torch.from_numpy(mask_and_boundary).type(torch.FloatTensor)
        mask_and_boundary_tensor = torch.permute(mask_and_boundary_tensor, (2, 0, 1))

        return {
            'transformed_image': img_tensor,
            'transformed_mask_and_boundary': mask_and_boundary_tensor
        }

class InferenceAugmentor():
    """
    Inference augmentor
    Performs resize, flip, rotate and etc. augmentations during inference
    img_tensor (tensor of N * H * W shape ): image tensor to augment 
    augmentation (string): augmentation to perform
    size (string): size for image resizing (N * H * W image becomes N * size * size). Default: None.
    degrees (string): angle for rotation. Default: None.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def augment(cls, img_tensor, augmentation, size=None):
        if augmentation == 'resize':
            assert size != None
            size = int(size)
            img_np = img_tensor.cpu().numpy()
            img_np = np.transpose(img_np, axes=[1,2,0])
            img_np = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_LANCZOS4)
            augmented_img = torch.from_numpy(img_np)
            augmented_img = augmented_img.permute(2, 0, 1)
        if augmentation == 'v_flip':
            v_flip = RandomVerticalFlip(p=1.0)
            augmented_img = v_flip(img_tensor)
        if augmentation == 'h_flip':
            h_flip = RandomHorizontalFlip(p=1.0)
            augmented_img = h_flip(img_tensor)
        return augmented_img
    
    @classmethod
    def reverse_augment(cls, img_tensor, augmentation):
        if augmentation == 'resize':
            img_np = img_tensor.cpu().numpy()
            img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            augmented_img = torch.from_numpy(img_np)
        if augmentation == 'v_flip':
            v_flip = RandomVerticalFlip(p=1.0)
            augmented_img = v_flip(img_tensor)
        if augmentation == 'h_flip':
            h_flip = RandomHorizontalFlip(p=1.0)
            augmented_img = h_flip(img_tensor)
        return augmented_img
    
    @classmethod
    def return_augmented(cls, img_tensor, augmentations_list, sizes_list):
        all_inputs_dict = {}
        for augmentation in augmentations_list:
            if augmentation == 'resize':
                for size in sizes_list:
                    augmented_img = InferenceAugmentor.augment(img_tensor, augmentation, size=size)
                    all_inputs_dict[size] = augmented_img
            if augmentation == 'v_flip':
                augmented_img = InferenceAugmentor.augment(img_tensor, augmentation)
                all_inputs_dict[augmentation] = augmented_img
            if augmentation == 'h_flip':
                augmented_img = InferenceAugmentor.augment(img_tensor, augmentation)
                all_inputs_dict[augmentation] = augmented_img
        return all_inputs_dict
    
    @classmethod
    def return_reverse_augmented(cls, all_outputs_dict, sizes_list):
        reverse_augmented_outputs_list = []
        for augmentation_key, augmented_output in all_outputs_dict.items():
            if augmentation_key in sizes_list:
                reverse_augmented_output = InferenceAugmentor.reverse_augment(augmented_output, augmentation='resize')
                reverse_augmented_outputs_list.append(reverse_augmented_output)
            if augmentation_key in 'v_flip':
                reverse_augmented_output = InferenceAugmentor.reverse_augment(augmented_output, augmentation_key)
                reverse_augmented_outputs_list.append(reverse_augmented_output)
            if augmentation_key in 'h_flip':
                reverse_augmented_output = InferenceAugmentor.reverse_augment(augmented_output, augmentation_key)
                reverse_augmented_outputs_list.append(reverse_augmented_output)
        return reverse_augmented_outputs_list


