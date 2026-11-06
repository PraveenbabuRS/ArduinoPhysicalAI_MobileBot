"""THE preprocessing contract. One implementation, used by both training and
on-robot inference.

This module must be byte-identical wherever it runs. When policy_node is
written, this file is vendored into it unchanged -- not reimplemented, not
"ported". A divergence between training-time and inference-time preprocessing
is a silent domain shift: the model trains fine, validates fine, and drives
badly, with nothing in any log to explain why.

Dependencies are deliberately limited to numpy and cv2 so this can be dropped
into a ROS node without dragging torch onto the robot.

Contract version: crop96-640x384-resize160x120-v1
    source frame : 640x480 (usb_cam, raw_mjpeg passthrough)
    crop         : drop the top 96 px (the upper 1/5 -- sky and background
                   clutter that a path-following policy must not key on)
    resize       : 160x120, the DonkeyCar/PilotNet-scale input
    colour       : RGB, float32, scaled to [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CONTRACT_VERSION = 'crop96-640x384-resize160x120-v1'


@dataclass(frozen=True)
class PreprocessConfig:
    crop_x: int = 0
    crop_y: int = 96
    crop_width: int = 640
    crop_height: int = 384
    resize_width: int = 160
    resize_height: int = 120

    @property
    def version(self) -> str:
        return (f'crop{self.crop_y}-{self.crop_width}x{self.crop_height}'
                f'-resize{self.resize_width}x{self.resize_height}-v1')


DEFAULT = PreprocessConfig()


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG bytes -> BGR uint8 array. Returns None-safe empty on failure."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('cv2.imdecode failed -- truncated or corrupt JPEG')
    return img


def preprocess_bgr(img_bgr: np.ndarray, cfg: PreprocessConfig = DEFAULT) -> np.ndarray:
    """BGR uint8 (any size) -> RGB float32 [0,1], shape (H, W, 3).

    Crop is applied before resize so the discarded sky pixels never influence
    the interpolation.
    """
    cropped = img_bgr[cfg.crop_y:cfg.crop_y + cfg.crop_height,
                      cfg.crop_x:cfg.crop_x + cfg.crop_width]
    resized = cv2.resize(cropped, (cfg.resize_width, cfg.resize_height),
                        interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def preprocess_file(path: str, cfg: PreprocessConfig = DEFAULT) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'could not read image: {path}')
    return preprocess_bgr(img, cfg)


def to_nchw(img_hwc: np.ndarray) -> np.ndarray:
    """(H, W, 3) -> (1, 3, H, W), the layout the exported model expects."""
    return np.transpose(img_hwc, (2, 0, 1))[np.newaxis, ...]
