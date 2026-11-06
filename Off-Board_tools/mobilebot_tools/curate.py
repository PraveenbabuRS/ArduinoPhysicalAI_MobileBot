"""Apply curation decisions and produce a route/episode-wise train/val split.

Reads frames.parquet and curation.json (written by mobilebot-view), applies
per-frame overrides, applies episode-level accept/reject, and writes a
filtered, split dataset. Never mutates the extracted dataset in place -- this
writes a new directory, so a re-curation is always cheap and Layer B is never
touched.

Why route-wise, not random, split: at 20 Hz consecutive frames are near-
duplicates. A random split leaks information across train/val and produces a
validation loss that looks good and predicts nothing about on-road behaviour
(Codevilla et al. measured only 0.39 correlation even in the friendlier case).
Holding out whole episodes is the minimum bar; holding out whole conditions
(a route driven at a different time of day) is better when you have enough
sessions to afford it.

Usage:
    mobilebot-curate --dataset ~/mobilebot_train/2026-08-17T14-03-00Z \\
                      --out ~/mobilebot_train/2026-08-17T14-03-00Z_curated \\
                      --val-episodes episode_0003_10-15-02 episode_0007_10-42-11
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _apply_frame_overrides(frames: pd.DataFrame, curation: dict) -> pd.DataFrame:
    frames = frames.copy()
    for key, decision in curation.items():
        episode_id, frame_index = key.rsplit(':', 1)
        mask = (frames['episode_id'] == episode_id) & \
               (frames['frame_index'] == int(frame_index))
        frames.loc[mask, 'valid'] = decision['valid']
        if decision.get('reject_reason'):
            frames.loc[mask, 'reject_reason'] = decision['reject_reason']
    return frames


def _apply_episode_rejects(frames: pd.DataFrame, episodes: pd.DataFrame,
                           reject_episodes: set[str]) -> pd.DataFrame:
    if not reject_episodes:
        return frames
    frames = frames.copy()
    mask = frames['episode_id'].isin(reject_episodes)
    frames.loc[mask, 'valid'] = False
    frames.loc[mask, 'reject_reason'] = 'episode rejected in curation'
    return frames


def curate(dataset_dir: Path, out_dir: Path, val_episodes: list[str],
          reject_episodes: list[str]) -> None:
    frames = pd.read_parquet(dataset_dir / 'frames.parquet')
    episodes = pd.read_parquet(dataset_dir / 'episodes.parquet')
    manifest = json.loads((dataset_dir / 'manifest.json').read_text())

    curation_path = dataset_dir / 'curation.json'
    curation = json.loads(curation_path.read_text()) if curation_path.exists() else {}

    all_episode_ids = set(frames['episode_id'].unique())
    val_set = set(val_episodes)
    reject_set = set(reject_episodes)

    unknown_val = val_set - all_episode_ids
    unknown_reject = reject_set - all_episode_ids
    if unknown_val:
        raise SystemExit(f'--val-episodes not found in dataset: {sorted(unknown_val)}')
    if unknown_reject:
        raise SystemExit(f'--reject-episodes not found in dataset: {sorted(unknown_reject)}')

    frames = _apply_frame_overrides(frames, curation)
    frames = _apply_episode_rejects(frames, episodes, reject_set)

    frames['split'] = 'train'
    frames.loc[frames['episode_id'].isin(val_set), 'split'] = 'val'

    kept = frames[frames['valid']].copy()
    dropped = len(frames) - len(kept)

    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_dir / 'frames.parquet', index=False)
    episodes.to_parquet(out_dir / 'episodes.parquet', index=False)

    curation_manifest = {
        **manifest,
        'curated_from': str(dataset_dir),
        'val_episodes': sorted(val_set),
        'reject_episodes': sorted(reject_set),
        'frame_count_before_curation': len(frames),
        'frame_count_after_curation': len(kept),
        'frames_dropped': dropped,
        'train_frame_count': int((kept['split'] == 'train').sum()),
        'val_frame_count': int((kept['split'] == 'val').sum()),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(curation_manifest, indent=2, default=str))

    print(f'curated: {len(kept)}/{len(frames)} frames kept ({dropped} dropped)')
    print(f'  train: {curation_manifest["train_frame_count"]} frames, '
         f'{len(all_episode_ids - val_set - reject_set)} episodes')
    print(f'  val:   {curation_manifest["val_frame_count"]} frames, '
         f'{len(val_set)} episodes')
    if not val_set:
        print('WARNING: no --val-episodes given -- this dataset has no held-out '
             'route and cannot report a meaningful validation metric')
    print(f'-> {out_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--val-episodes', nargs='*', default=[],
                        help='episode_id values to hold out entirely for validation')
    parser.add_argument('--reject-episodes', nargs='*', default=[],
                        help='episode_id values to drop entirely (crashes, bad runs)')
    args = parser.parse_args()

    curate(args.dataset, args.out, args.val_episodes, args.reject_episodes)


if __name__ == '__main__':
    main()
