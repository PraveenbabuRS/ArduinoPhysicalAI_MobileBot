"""Merge several per-session extracted datasets (each already produced by
mobilebot-extract) into one combined dataset that mobilebot-curate,
mobilebot-validate, mobilebot-stats, and training can all consume unchanged.

Why this is a separate step rather than pointing curate at multiple dirs:
episode IDs are only unique WITHIN a session ("episode_0001" exists in every
session). Concatenating raw parquet files as-is would make --reject-episodes
and --val-episodes ambiguous -- rejecting "episode_0001" would silently apply
to all three sessions' episode_0001 at once. This tool prefixes every
episode_id with its source session so every episode has one unambiguous name
across the whole merged dataset.

It also resolves every image_path to a real, verified-to-exist absolute path
at merge time (each source session has its own images_root; a merged dataset
can't rely on a single shared one), and refuses to merge sessions recorded
under different preprocessing contracts -- mixing crop/resize contracts
silently corrupts training with no error anywhere else in the pipeline.

Usage:
    mobilebot-merge --datasets ~/ExtractedData/2026-08-22T14-01-10Z \\
                              ~/ExtractedData/2026-08-22T14-37-30Z \\
                              ~/ExtractedData/2026-08-22T14-48-52Z \\
                    --out ~/ExtractedData/merged_2026-08-22

    # then curate exactly as with a single session, using the prefixed IDs:
    mobilebot-curate --dataset ~/ExtractedData/merged_2026-08-22 \\
                     --out ~/ExtractedData/merged_2026-08-22_curated \\
                     --reject-episodes "2026-08-22T14-01-10Z__episode_0003" \\
                     --val-episodes   "2026-08-22T14-37-30Z__episode_0002"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mobilebot_tools.paths import resolve_dataset_paths


def _session_preprocessing_version(manifest: dict) -> str | None:
    # Layer C metadata is nested under session_metadata in the per-session
    # manifest; fall back to a top-level key for forward compatibility.
    return (manifest.get('session_metadata', {}).get('preprocessing_version')
           or manifest.get('preprocessing_version'))


def merge(dataset_dirs: list[Path], out_dir: Path) -> None:
    all_frames = []
    all_episodes = []
    session_summaries = []
    preprocessing_versions: dict[str, list[str]] = {}

    for d in dataset_dirs:
        d = Path(d)
        frames_path, episodes_path, manifest_path = (
            d / 'frames.parquet', d / 'episodes.parquet', d / 'manifest.json')
        for p in (frames_path, episodes_path, manifest_path):
            if not p.exists():
                raise SystemExit(f'{p} missing -- is {d} a valid mobilebot-extract output?')

        frames = pd.read_parquet(frames_path)
        episodes = pd.read_parquet(episodes_path)
        manifest = json.loads(manifest_path.read_text())

        # Resolve to real absolute paths NOW, per-session, while we still know
        # this session's own images_root. The merged dataset has no single
        # shared images_root, so this must happen before concatenation.
        frames = resolve_dataset_paths(frames, d)

        session_tag = d.name
        frames['source_session'] = session_tag
        episodes['source_session'] = session_tag
        frames['episode_id'] = session_tag + '__' + frames['episode_id'].astype(str)
        episodes['episode_id'] = session_tag + '__' + episodes['episode_id'].astype(str)

        pv = _session_preprocessing_version(manifest)
        preprocessing_versions.setdefault(str(pv), []).append(session_tag)

        n_valid = int(frames['valid'].sum()) if 'valid' in frames else len(frames)
        session_summaries.append({
            'session': session_tag, 'source_dir': str(d),
            'episode_count': len(episodes), 'frame_count': len(frames),
            'valid_frame_count': n_valid,
        })
        print(f'  {session_tag}: {len(episodes)} episodes, {len(frames)} frames '
             f'({n_valid} valid), preprocessing_version={pv}')

        all_frames.append(frames)
        all_episodes.append(episodes)

    if len(preprocessing_versions) > 1:
        lines = [f'  {pv}: {", ".join(sessions)}' for pv, sessions in preprocessing_versions.items()]
        raise SystemExit(
            'REFUSING TO MERGE: sessions were recorded under DIFFERENT preprocessing '
            'contracts:\n' + '\n'.join(lines) +
            '\nMixing crop/resize contracts silently corrupts training with no error '
            'anywhere downstream. Re-extract the mismatched session(s) under the current '
            'contract, or merge only the sessions that already agree.')

    merged_frames = pd.concat(all_frames, ignore_index=True)
    merged_episodes = pd.concat(all_episodes, ignore_index=True)

    dupe_ids = merged_episodes['episode_id'][merged_episodes['episode_id'].duplicated()]
    if len(dupe_ids):
        raise SystemExit(f'INTERNAL ERROR: duplicate episode_id after prefixing: '
                        f'{dupe_ids.unique().tolist()} -- do two input dirs share a name?')

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_frames.to_parquet(out_dir / 'frames.parquet', index=False)
    merged_episodes.to_parquet(out_dir / 'episodes.parquet', index=False)

    manifest = {
        'merged_from': [str(d) for d in dataset_dirs],
        'session_count': len(dataset_dirs),
        'frame_count': len(merged_frames),
        'episode_count': len(merged_episodes),
        'preprocessing_version': next(iter(preprocessing_versions)),
        'sessions': session_summaries,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))

    print(f'\nmerged {len(dataset_dirs)} sessions -> {len(merged_episodes)} episodes, '
         f'{len(merged_frames)} frames')
    print(f'-> {out_dir}')
    print('\nEpisode IDs are now session-prefixed, e.g.:')
    for eid in merged_episodes['episode_id'].head(3):
        print(f'  {eid}')
    print('\nUse these full prefixed IDs with --reject-episodes / --val-episodes in '
         'mobilebot-curate.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', required=True, nargs='+', type=Path,
                   help='two or more extracted session directories to merge')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()

    if len(args.datasets) < 2:
        raise SystemExit('need at least 2 --datasets to merge (use the session directly '
                        'with mobilebot-curate if there is only one)')

    print(f'merging {len(args.datasets)} sessions:')
    merge(args.datasets, args.out)


if __name__ == '__main__':
    main()
