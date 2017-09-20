import chainer
import random
from PIL import Image, ImageMath, ImageEnhance
import numpy as np
import os


def color_aug(image):
    # 色相いじる
    h, s, v = image.convert("HSV").split()
    _h = ImageMath.eval("(h + {}) % 255".format(np.random.randint(-25, 25)), h=h).convert("L")
    img = Image.merge("HSV", (_h, s, v)).convert("RGB")

    # 彩度を変える
    saturation_converter = ImageEnhance.Color(img)
    img = saturation_converter.enhance(np.random.uniform(0.9, 1.1))

    # コントラストを変える
    contrast_converter = ImageEnhance.Contrast(img)
    img = contrast_converter.enhance(np.random.uniform(0.9, 1.1))

    # 明度を変える
    brightness_converter = ImageEnhance.Brightness(img)
    img = brightness_converter.enhance(np.random.uniform(0.9, 1.1))

    # シャープネスを変える
    sharpness_converter = ImageEnhance.Sharpness(img)
    img = sharpness_converter.enhance(np.random.uniform(0.9, 1.1))

    return img


class KomeDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base, original_size=256, crop_size=224, random=True, color_augmentation=False, rotate=True, nocrop=False):
        self.base = base
        self.image_size = original_size
        self.crop_size = crop_size
        self.nocrop = nocrop
        self.rotate = rotate
        self.color_augmentation = color_augmentation
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        # load data
        path, label = self.base[i]
        image = Image.open(path)

        # color augmentation
        if self.color_augmentation:
            image = color_aug(image)

        # rotation (degree, not radian)
        if self.rotate:
            r = np.random.choice([0, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
            image = image.transpose(r) if r > 0 else image

        image = np.asarray(image)

        # transpose
        image = image[:, :, ::-1].astype('float32')
        image -= np.array([104.0, 117.0, 123.0], dtype=np.float32)  # BGR
        image = image.transpose((2, 0, 1))

        # crop
        if not self.nocrop:
            crop_size = self.crop_size
            _, h, w = image.shape
            if random:
                # Randomly crop a region and flip the image
                top = random.randint(0, h - crop_size - 1)
                left = random.randint(0, w - crop_size - 1)
                if random.randint(0, 1):
                    image = image[:, :, ::-1]
            else:
                # Crop the center
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
            bottom = top + crop_size
            right = left + crop_size
            image = image[:, top:bottom, left:right]

        return image, label


