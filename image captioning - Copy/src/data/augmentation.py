"""Image transform pipelines for IU X-Ray chest radiographs.

Both pipelines first resize the (non-square) source image so its longer side
equals ``image_size``, preserving aspect ratio, then zero-pad to a square.
This avoids the geometric distortion a naive resize would introduce on the
wide lateral chest X-ray views.
"""

from typing import Sequence

import torchvision.transforms as T
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizePad:
    """Resize an image's longer side to ``size`` and zero-pad to a square.

    Args:
        size: Target square side length.
    """

    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        pad_w, pad_h = self.size - new_w, self.size - new_h
        left, top = pad_w // 2, pad_h // 2
        right, bottom = pad_w - left, pad_h - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)


def build_train_transform(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> T.Compose:
    """Training-time augmentation pipeline.

    Aspect-ratio-preserving resize+pad, then random horizontal flip, a
    small random rotation, mild color jitter, normalization, and random
    erasing -- per the architecture spec's augmentation policy.

    Args:
        image_size: Target square side length.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        A ``torchvision.transforms.Compose`` mapping a PIL image to a
        normalized ``(3, image_size, image_size)`` tensor.
    """
    return T.Compose(
        [
            ResizePad(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=10)], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.1, scale=(0.02, 0.33)),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> T.Compose:
    """Deterministic resize+pad and normalization, no augmentation.

    Args:
        image_size: Target square side length.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        A ``torchvision.transforms.Compose`` mapping a PIL image to a
        normalized ``(3, image_size, image_size)`` tensor.
    """
    return T.Compose(
        [
            ResizePad(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
