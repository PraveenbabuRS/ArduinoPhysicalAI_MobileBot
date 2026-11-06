"""Dataset, class derivation, balancing, and augmentation.

Everything distribution-dependent is computed from the data at load time, not
hardcoded. An earlier version of mobilebot-stats reported a badly wrong
steering histogram because of a unit bug; nothing here should ever depend on a
number a human transcribed from a report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mobilebot_tools.paths import resolve_dataset_paths
from mobilebot_tools.preprocess import PreprocessConfig, DEFAULT, preprocess_file

# Speed class definitions. Two-class is the proof-of-concept setting: the
# scene-triggered stop is the learnable signal, and finer speed levels need
# far more data per class than a first collection provides.
SPEED_CLASSES_2 = ['idle', 'forward']
SPEED_CLASSES_3 = ['reverse', 'idle', 'forward']

IDLE_THRESHOLD_MPS = 0.01


def speed_to_class(speed: float, num_classes: int) -> int:
    if num_classes == 2:
        return 1 if speed > IDLE_THRESHOLD_MPS else 0
    if speed < -IDLE_THRESHOLD_MPS:
        return 0
    if speed > IDLE_THRESHOLD_MPS:
        return 2
    return 1


def class_names(num_classes: int) -> list[str]:
    return SPEED_CLASSES_2 if num_classes == 2 else SPEED_CLASSES_3


@dataclass
class BalanceReport:
    kept: int
    dropped_straight: int
    steer_histogram: dict[str, int]
    speed_counts: dict[str, int]

    def describe(self) -> str:
        lines = [f'  kept {self.kept} frames (dropped {self.dropped_straight} '
                 f'over-represented near-zero-steering frames)']
        lines.append('  steering: ' + '  '.join(
            f'{k}={v}' for k, v in self.steer_histogram.items()))
        lines.append('  speed:    ' + '  '.join(
            f'{k}={v}' for k, v in self.speed_counts.items()))
        return '\n'.join(lines)


def balance_steering(frames: pd.DataFrame, straight_deg: float = 5.0,
                    keep_fraction: float = 0.35,
                    seed: int = 0) -> tuple[pd.DataFrame, BalanceReport]:
    """Downsample the dominant near-zero-steering bin.

    Straight driving dominates any driving dataset -- published lane-keeping
    work reports ~90% of frames inside +-5 deg. Left unbalanced, the network
    minimises loss by predicting ~0 everywhere and never turning. Downsampling
    the straight bin is the standard remedy and is applied to the TRAINING
    split only; validation must keep its natural distribution or the metric
    stops meaning anything.
    """
    steer_deg = np.degrees(frames['steer_cmd_rad'].to_numpy())
    is_straight = np.abs(steer_deg) < straight_deg

    rng = np.random.default_rng(seed)
    straight_idx = np.flatnonzero(is_straight)
    n_keep = int(len(straight_idx) * keep_fraction)
    keep_straight = rng.choice(straight_idx, size=n_keep, replace=False) \
        if n_keep < len(straight_idx) else straight_idx

    keep_mask = ~is_straight
    keep_mask[keep_straight] = True
    balanced = frames.loc[keep_mask].reset_index(drop=True)

    edges = [-90, -25, -15, -5, 5, 15, 25, 90]
    bal_deg = np.degrees(balanced['steer_cmd_rad'].to_numpy())
    counts, _ = np.histogram(bal_deg, bins=edges)
    hist = {f'[{edges[i]},{edges[i+1]})': int(c) for i, c in enumerate(counts) if c}

    report = BalanceReport(
        kept=len(balanced),
        dropped_straight=len(frames) - len(balanced),
        steer_histogram=hist,
        speed_counts={},
    )
    return balanced, report


class DrivingDataset(Dataset):

    def __init__(self, frames: pd.DataFrame, num_speed_classes: int = 2,
                 augment: bool = False, cfg: PreprocessConfig = DEFAULT,
                 seed: int = 0) -> None:
        self.frames = frames.reset_index(drop=True)
        self.num_speed_classes = num_speed_classes
        self.augment = augment
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frames)

    def speed_class_counts(self) -> dict[str, int]:
        names = class_names(self.num_speed_classes)
        classes = [speed_to_class(s, self.num_speed_classes)
                   for s in self.frames['speed_cmd_mps']]
        return {names[i]: int(classes.count(i)) for i in range(self.num_speed_classes)}

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights. Idle is a minority class by construction --
        most of a run is driving, not stopped -- so an unweighted loss would
        learn to never predict a stop."""
        counts = self.speed_class_counts()
        names = class_names(self.num_speed_classes)
        total = sum(counts.values())
        weights = [total / (self.num_speed_classes * max(counts[n], 1)) for n in names]
        return torch.tensor(weights, dtype=torch.float32)

    def _augment(self, img: np.ndarray, steer: float) -> tuple[np.ndarray, float]:
        # Horizontal flip with steering negation. The highest-value
        # augmentation here: this dataset is strongly right-biased, and without
        # flipping the policy will not learn to turn left.
        if self.rng.random() < 0.5:
            img = img[:, ::-1, :].copy()
            steer = -steer

        # Brightness -- outdoor illumination varies far more than any single
        # session captures.
        img = np.clip(img * self.rng.uniform(0.6, 1.4), 0.0, 1.0)

        # Synthetic shadow over a random vertical band, mimicking dappled light
        # and cast shadows across the path.
        if self.rng.random() < 0.4:
            h, w, _ = img.shape
            x0 = self.rng.integers(0, w)
            x1 = np.clip(x0 + self.rng.integers(w // 6, w // 2), 0, w)
            img[:, x0:x1, :] *= self.rng.uniform(0.45, 0.8)

        return img.astype(np.float32), float(steer)

    def __getitem__(self, index: int):
        row = self.frames.iloc[index]
        img = preprocess_file(row['image_path'], self.cfg)
        steer = float(row['steer_cmd_rad'])

        if self.augment:
            img, steer = self._augment(img, steer)

        speed_class = speed_to_class(float(row['speed_cmd_mps']),
                                     self.num_speed_classes)

        return (torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()),
                torch.tensor(steer, dtype=torch.float32),
                torch.tensor(speed_class, dtype=torch.long))


def load_splits(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = Path(dataset_dir)
    frames = pd.read_parquet(dataset_dir / 'frames.parquet')
    frames = resolve_dataset_paths(frames, dataset_dir)
    if 'split' not in frames.columns:
        raise SystemExit(
            'frames.parquet has no "split" column -- run mobilebot-curate first '
            'so the train/val split is episode-wise and recorded in the data')
    train = frames[frames['split'] == 'train'].reset_index(drop=True)
    val = frames[frames['split'] == 'val'].reset_index(drop=True)
    if len(val) == 0:
        raise SystemExit('validation split is empty -- pass --val-episodes to mobilebot-curate')
    if len(val) > len(train):
        print(f'WARNING: validation split ({len(val)}) is LARGER than training '
             f'({len(train)}). Re-run mobilebot-curate holding out shorter '
             f'episodes -- aim for roughly 20% of frames in validation.')
    return train, val
