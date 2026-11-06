"""Resolve frames.parquet's image_path column against wherever the dataset
actually lives right now.

Why this exists: an absolute path baked in at extraction time breaks the
moment the dataset directory is copied, renamed, or used from a different
machine -- exactly what happened when a path like
'ExtractedData/2026-08-18T16-37-21Z/images/episode_0009/000392.jpg' was still
resolvable from the directory `mobilebot-extract` was originally run from, but
not from wherever `mobilebot-train` happened to be invoked from later. Every
tool that touches images now calls resolve_dataset_paths() exactly once, right
after loading frames.parquet, so a broken path is a single clear error at
startup -- not a cryptic failure inside DataLoader worker process 3, 40
minutes into a run, after 39 successful epochs already ate the GPU time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _candidate_paths(stored: str, dataset_dir: Path, images_root: Path | None) -> list[Path]:
    candidates = []
    p = Path(stored)
    if p.is_absolute():
        candidates.append(p)
    if images_root is not None:
        candidates.append(images_root / stored)
    candidates.append(dataset_dir / stored)

    # Compatibility fallback for datasets written before images_root existed
    # in the manifest, which could carry an arbitrary leading prefix (e.g. a
    # relative --out path from wherever `extract` happened to be run). Find
    # the 'images' path component and re-anchor from there -- that segment's
    # internal structure (images/episode_NNNN/NNNNNN.jpg) has been stable
    # since the recorder was first written, even when the prefix before it
    # was not.
    parts = p.parts
    if 'images' in parts:
        tail = Path(*parts[parts.index('images'):])
        if images_root is not None:
            candidates.append(images_root / tail)
        candidates.append(dataset_dir / tail)

    return candidates


def resolve_dataset_paths(frames: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """Returns a copy of `frames` with image_path rewritten to a fully
    resolved, absolute, verified-to-exist path for every row.

    Raises SystemExit with a diagnostic (not a bare exception) if any path
    cannot be found, since this is meant to be the fail-fast checkpoint before
    a long training run, not something to debug via a stack trace.
    """
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / 'manifest.json'
    images_root = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('images_root'):
            images_root = Path(manifest['images_root'])

    frames = frames.copy()
    resolved: list[str] = []
    failures: list[tuple[str, list[str]]] = []

    for stored in frames['image_path']:
        found = None
        tried = []
        for candidate in _candidate_paths(stored, dataset_dir, images_root):
            tried.append(str(candidate))
            if candidate.exists():
                found = candidate
                break
        if found is None:
            failures.append((stored, tried))
            resolved.append(stored)  # placeholder; we raise below regardless
        else:
            resolved.append(str(found))

    if failures:
        lines = [f'Could not resolve {len(failures)}/{len(frames)} image paths '
                f'in {dataset_dir}. First few failures:']
        for stored, tried in failures[:5]:
            lines.append(f'  stored: {stored}')
            for t in tried:
                lines.append(f'    tried: {t}')
        lines.append('')
        if images_root is None:
            lines.append(
                "manifest.json has no 'images_root' -- this dataset was extracted "
                'before that field existed. Re-run mobilebot-extract on the source '
                'session (the .mcap bags are untouched, so this is safe and cheap) '
                'to regenerate frames.parquet with portable paths, then re-run '
                'mobilebot-curate.')
        else:
            lines.append(f"manifest.json images_root: {images_root} -- "
                         'does that directory still exist and contain the images?')
        raise SystemExit('\n'.join(lines))

    frames['image_path'] = resolved
    return frames
