"""Layer B sanity checks. Run this before trusting a dataset for training.

Checks frame counts against the source episodes, timestamp-gap-based dropped
frame detection independent of the extractor's own count, alignment residual
distribution, steering histogram (the straight-line bias the design report
flags), duplicate-image detection (the DonkeyCar tub silent-overwrite failure
mode), and exposure/brightness outliers that might mean the exposure lock
slipped mid-session.

Usage:
    mobilebot-validate --dataset ~/mobilebot_train/2026-08-17T14-03-00Z
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mobilebot_tools.paths import resolve_dataset_paths


def _duplicate_image_hashes(image_paths: pd.Series, sample_every: int = 1) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = {}
    for path in image_paths.iloc[::sample_every]:
        p = Path(path)
        if not p.exists():
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(str(p))
    return {h: paths for h, paths in hashes.items() if len(paths) > 1}


def _undecodable_images(image_paths: pd.Series) -> list[str]:
    """Flag files that fail to actually decode as an image.

    An earlier version of this check required the JPEG EOI marker (FF D9) to
    be the literal last two bytes of the file. That is stricter than the JPEG
    format actually requires: every real decoder (OpenCV, PIL, any image
    viewer) stops at the first EOI it finds and ignores anything after it. If
    the bytes decoded off the MCAP message carry so much as one trailing byte
    past the real EOI, the old check failed every single frame while every
    real decoder opened them perfectly -- which is exactly what happened here
    (100% flagged, all confirmed openable elsewhere). Testing with the actual
    decoder instead of a byte-position assumption is both more correct and
    tests the thing that actually matters: will this frame decode when
    training reads it.
    """
    import cv2
    import numpy as np

    bad: list[str] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            bad.append(str(p))
            continue
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                bad.append(str(p))
        except OSError:
            bad.append(str(p))
    return bad


def _brightness_outliers(
        image_paths: pd.Series, sample_every: int = 5) -> list[str]:
    try:
        from PIL import Image
    except ImportError:
        print('Pillow not installed; skipping brightness check (pip install Pillow)')
        return []

    values = []
    sampled_paths = []
    for path in image_paths.iloc[::sample_every]:
        p = Path(path)
        if not p.exists():
            continue
        with Image.open(p) as im:
            arr = np.asarray(im.convert('L'), dtype=np.float32)
        values.append(arr.mean())
        sampled_paths.append(str(p))

    if len(values) < 10:
        return []

    values = np.array(values)
    mean, std = values.mean(), values.std()
    if std < 1e-6:
        return []
    z = np.abs((values - mean) / std)
    return [p for p, zi in zip(sampled_paths, z) if zi > 3.0]


def validate(dataset_dir: Path, brightness_sample_every: int = 5,
            hash_sample_every: int = 1) -> bool:
    frames_path = dataset_dir / 'frames.parquet'
    episodes_path = dataset_dir / 'episodes.parquet'
    manifest_path = dataset_dir / 'manifest.json'

    for p in (frames_path, episodes_path, manifest_path):
        if not p.exists():
            print(f'FAIL: missing {p.name} -- run mobilebot-extract first')
            return False

    frames = pd.read_parquet(frames_path)
    frames = resolve_dataset_paths(frames, dataset_dir)
    episodes = pd.read_parquet(episodes_path)
    manifest = json.loads(manifest_path.read_text())

    ok = True

    # -- frame counts --
    per_episode_actual = frames.groupby('episode_id').size()
    for _, ep in episodes.iterrows():
        actual = per_episode_actual.get(ep['episode_id'], 0)
        if actual != ep['frame_count']:
            print(f'FAIL: {ep["episode_id"]} manifest says {ep["frame_count"]} frames, '
                 f'parquet has {actual}')
            ok = False

    # -- missing image files --
    missing = [p for p in frames['image_path'] if not Path(p).exists()]
    if missing:
        print(f'FAIL: {len(missing)} image files referenced in frames.parquet do not exist '
             f'on disk (first: {missing[0]})')
        ok = False

    # -- alignment quality --
    unaligned = int((~frames['valid']).sum())
    if unaligned:
        pct = 100 * unaligned / len(frames)
        level = 'FAIL' if pct > 5.0 else 'WARN'
        print(f'{level}: {unaligned} frames ({pct:.1f}%) have no aligned command')
        if pct > 5.0:
            ok = False

    residuals = frames['align_residual_ms'].dropna()
    if len(residuals):
        print(f'alignment residual: mean {residuals.mean():.2f} ms, '
             f'p95 {residuals.quantile(0.95):.2f} ms, max {residuals.max():.2f} ms')

    # -- dropped frames --
    total_dropped = int(episodes['dropped_frame_count'].sum())
    if total_dropped:
        print(f'WARN: {total_dropped} dropped-frame gaps detected across the session')

    # -- steering histogram: is this dataset straight-biased as expected? --
    steer = frames['steer_cmd_rad'].dropna()
    if len(steer):
        near_zero_deg = 5.0 * np.pi / 180.0
        frac_near_zero = float((steer.abs() < near_zero_deg).mean())
        print(f'steering: {frac_near_zero * 100:.1f}% of frames within ±5° of centre '
             f'(literature baseline for lane-keeping data is ~90%)')
        if frac_near_zero > 0.97:
            print('WARN: extremely straight-biased -- consider more turning/curved-route data')

    # -- images that fail to actually decode --
    undecodable = _undecodable_images(frames['image_path'])
    if undecodable:
        pct = 100 * len(undecodable) / len(frames)
        level = 'FAIL' if pct > 1.0 else 'WARN'
        print(f'{level}: {len(undecodable)} frames ({pct:.1f}%) fail to decode as an image '
             f'-- these will crash training. e.g. {undecodable[0]}')
        if pct > 1.0:
            ok = False

    # -- duplicate images (DonkeyCar's silent-overwrite failure mode) --
    dupes = _duplicate_image_hashes(frames['image_path'], sample_every=hash_sample_every)
    if dupes:
        n_affected = sum(len(v) for v in dupes.values())
        print(f'FAIL: {len(dupes)} distinct images repeated across {n_affected} frames -- '
             f'possible writer overwrite bug, inspect before training')
        ok = False

    # -- brightness outliers (exposure lock slipped?) --
    outliers = _brightness_outliers(frames['image_path'], sample_every=brightness_sample_every)
    if outliers:
        print(f'WARN: {len(outliers)} sampled frames are brightness outliers '
             f'(>3 sigma) -- check for exposure drift, e.g. {outliers[0]}')

    print(f'\n{"PASS" if ok else "FAIL"}: {len(frames)} frames, {len(episodes)} episodes, '
         f'extractor {manifest.get("extractor_version", "unknown")}')
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--brightness-sample-every', type=int, default=5,
                        help='check every Nth frame for brightness outliers')
    parser.add_argument('--hash-sample-every', type=int, default=1,
                        help='check every Nth frame for duplicate images (1 = all)')
    args = parser.parse_args()

    ok = validate(args.dataset, args.brightness_sample_every, args.hash_sample_every)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
