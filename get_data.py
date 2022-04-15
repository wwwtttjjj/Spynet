import itertools
import os
import re
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from torchvision.io.image import _read_png_16
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset


__all__ = (
    "KittiFlow",
    "Sintel",
    "FlyingThings3D",
    "FlyingChairs",
    "HD1K",
)


class FlowDataset(ABC, VisionDataset):
    # Some datasets like Kitti have a built-in valid_flow_mask, indicating which flow values are valid
    # For those we return (img1, img2, flow, valid_flow_mask), and for the rest we return (img1, img2, flow),
    # and it's up to whatever consumes the dataset to decide what valid_flow_mask should be.
    _has_builtin_flow_mask = False

    def __init__(self, root, transforms=None):

        super().__init__(root=root)
        self.transforms = transforms

        self._flow_list = []
        self._image_list = []

    def _read_img(self, file_name):
        img = Image.open(file_name)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @abstractmethod
    def _read_flow(self, file_name):
        # Return the flow or a tuple with the flow and the valid_flow_mask if _has_builtin_flow_mask is True
        pass

    def __getitem__(self, index):

        img1 = self._read_img(self._image_list[index][0])
        img2 = self._read_img(self._image_list[index][1])

        if self._flow_list:  # it will be empty for some dataset when split="test"
            flow = self._read_flow(self._flow_list[index])
            if self._has_builtin_flow_mask:
                flow, valid_flow_mask = flow
            else:
                valid_flow_mask = None
        else:
            flow = valid_flow_mask = None

        if self.transforms is not None:
            img1, img2, flow, valid_flow_mask = self.transforms(img1, img2, flow, valid_flow_mask)

        if self._has_builtin_flow_mask or valid_flow_mask is not None:
            # The `or valid_flow_mask is not None` part is here because the mask can be generated within a transform
            return img1, img2, flow, valid_flow_mask
        else:
            return img1, img2, flow

    def __len__(self):
        return len(self._image_list)

    def __rmul__(self, v):
        return torch.utils.data.ConcatDataset([self] * v)


class Sintel(FlowDataset):
    """`Sintel <http://sintel.is.tue.mpg.de/>`_ Dataset for optical flow.

    The dataset is expected to have the following structure: ::

        root
            Sintel
                testing
                    clean
                        scene_1
                        scene_2
                        ...
                    final
                        scene_1
                        scene_2
                        ...
                training
                    clean
                        scene_1
                        scene_2
                        ...
                    final
                        scene_1
                        scene_2
                        ...
                    flow
                        scene_1
                        scene_2
                        ...

    Args:
        root (string): Root directory of the Sintel Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        pass_name (string, optional): The pass to use, either "clean" (default), "final", or "both". See link above for
            details on the different passes.
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
            ``valid_flow_mask`` is expected for consistency with other datasets which
            return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
    """

    def __init__(self, root, split="train", pass_name="clean", transforms=None):
        super().__init__(root=root, transforms=transforms)

        verify_str_arg(split, "split", valid_values=("train", "test"))
        verify_str_arg(pass_name, "pass_name", valid_values=("clean", "final", "both"))
        passes = ["clean", "final"] if pass_name == "both" else [pass_name]

        root = Path(root) / "Sintel"
        flow_root = root / "training" / "flow"

        for pass_name in passes:
            split_dir = "training" if split == "train" else split
            image_root = root / split_dir / pass_name
            for scene in os.listdir(image_root):
                image_list = sorted(glob(str(image_root / scene / "*.png")))
                for i in range(len(image_list) - 1):
                    self._image_list += [[image_list[i], image_list[i + 1]]]

                if split == "train":
                    self._flow_list += sorted(glob(str(flow_root / scene / "*.flo")))

    def __getitem__(self, index):
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3-tuple with ``(img1, img2, flow)``.
            The flow is a numpy array of shape (2, H, W) and the images are PIL images.
            ``flow`` is None if ``split="test"``.
            If a valid flow mask is generated within the ``transforms`` parameter,
            a 4-tuple with ``(img1, img2, flow, valid_flow_mask)`` is returned.
        """
        return super().__getitem__(index)


    def _read_flow(self, file_name):
        return _read_flo(file_name)

def _read_flo(file_name):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    # Everything needs to be in little Endian according to
    # https://vision.middlebury.edu/flow/code/flow-code/README.txt
    with open(file_name, "rb") as f:
        magic = np.fromfile(f, "c", count=4).tobytes()
        if magic != b"PIEH":
            raise ValueError("Magic number incorrect. Invalid .flo file")

        w = int(np.fromfile(f, "<i4", count=1))
        h = int(np.fromfile(f, "<i4", count=1))
        data = np.fromfile(f, "<f4", count=2 * w * h)
        return data.reshape(h, w, 2).transpose(2, 0, 1)
