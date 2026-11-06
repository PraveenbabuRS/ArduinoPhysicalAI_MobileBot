"""Per-episode and dataset-wide summary. The quick health check before you
bother opening a notebook.

Usage:
    mobilebot-stats --dataset ~/mobilebot_train/2026-08-17T14-03-00Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _histogram_line(values_deg: pd.Series, edges_deg: list[float]) -> str:
    """Histogram of steering in DEGREES against DEGREE bin edges.

    Both arguments must be in degrees. An earlier version converted the edges
    to radians while the values were already degrees, which silently dropped
    every sample outside +-0.436 deg and reported a meaningless distribution.
    Out-of-range samples are now counted explicitly rather than vanishing.
    """
    counts, _ = np.histogram(values_deg, bins=edges_deg)
    below = int((values_deg < edges_deg[0]).sum())
    above = int((values_deg > edges_deg[-1]).sum())
    total = len(values_deg)
    if total == 0:
        return '(no data)'

    parts = []
    if below:
        parts.append(f'(<{edges_deg[0]:+.0f})={100 * below / total:4.1f}%')
    for i, c in enumerate(counts):
        pct = 100 * c / total
        parts.append(f'[{edges_deg[i]:+.0f},{edges_deg[i+1]:+.0f})={pct:4.1f}%')
    if above:
        parts.append(f'(>{edges_deg[-1]:+.0f})={100 * above / total:4.1f}%')
    return '  '.join(parts)


def stats(dataset_dir: Path) -> None:
    frames = pd.read_parquet(dataset_dir / 'frames.parquet')
    episodes = pd.read_parquet(dataset_dir / 'episodes.parquet')
    manifest_path = dataset_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    valid = frames[frames['valid']] if 'valid' in frames else frames

    print(f'=== {dataset_dir} ===')
    print(f'extractor version : {manifest.get("extractor_version", "unknown")}')
    print(f'episodes          : {len(episodes)}')
    print(f'frames (total)    : {len(frames)}')
    print(f'frames (valid)    : {len(valid)}')
    if 'split' in frames:
        print(f'  train           : {(frames["split"] == "train").sum()}')
        print(f'  val             : {(frames["split"] == "val").sum()}')

    if len(valid) and 'steer_cmd_rad' in valid:
        steer = valid['steer_cmd_rad'].dropna()
        steer_deg = np.degrees(steer)
        print(f'\nsteering (deg): mean {steer_deg.mean():+.2f}  std {steer_deg.std():.2f}  '
             f'p05 {steer_deg.quantile(0.05):+.2f}  p95 {steer_deg.quantile(0.95):+.2f}')
        print('steering histogram: ' + _histogram_line(
            steer_deg, [-25, -15, -5, 5, 15, 25]))

    if len(valid) and 'speed_cmd_mps' in valid:
        speed = valid['speed_cmd_mps'].dropna()
        print(f'\nspeed (m/s): mean {speed.mean():+.2f}  std {speed.std():.2f}  '
             f'unique setpoints seen: {sorted(speed.round(2).unique().tolist())}')

    if len(valid) and 'control_source' in valid:
        print('\ncontrol source breakdown:')
        print(valid['control_source'].value_counts().to_string())

    print('\nper-episode:')
    cols = [c for c in ('episode_id', 'frame_count', 'dropped_frame_count',
                        'unaligned_frame_count', 'mean_align_residual_ms',
                        'accepted') if c in episodes]
    print(episodes[cols].to_string(index=False))

    total_dropped = int(episodes['dropped_frame_count'].sum()) if 'dropped_frame_count' in episodes else 0
    if total_dropped:
        drop_pct = 100 * total_dropped / (len(frames) + total_dropped)
        print(f'\ndropped frames: {total_dropped} ({drop_pct:.1f}% of nominal capture)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    args = parser.parse_args()
    stats(args.dataset)


if __name__ == '__main__':
    main()
