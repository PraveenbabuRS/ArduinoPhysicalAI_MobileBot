"""Scrub through one episode with a steering overlay, mark bad ranges.

Keys while running:
    d / a       next / previous frame
    D / A       jump 10 frames forward / back
    space       toggle current frame valid/invalid
    r           enter a reject reason for the current frame (typed in terminal)
    s           save curation decisions to curation.json in the dataset dir
    q           save and quit

Curation is written as a separate file, never by mutating frames.parquet, so
it is reversible and survives a Layer B rebuild after re-extraction.

Usage:
    mobilebot-view --dataset ~/mobilebot_train/2026-08-17T14-03-00Z --episode episode_0001_14-05-02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mobilebot_tools.paths import resolve_dataset_paths

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'mobilebot-view needs opencv-python-headless: '
        'pip install "mobilebot-tools[view]"') from exc


def _load_curation(dataset_dir: Path) -> dict:
    path = dataset_dir / 'curation.json'
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_curation(dataset_dir: Path, curation: dict) -> None:
    (dataset_dir / 'curation.json').write_text(json.dumps(curation, indent=2))
    print(f'saved curation.json ({len(curation)} overridden frames)')


def _draw_overlay(image: np.ndarray, row: pd.Series, valid: bool, frame_no: int, total: int) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = image.copy()

    # Steering needle: centred, deflects left/right proportional to angle.
    cx, cy = w // 2, h - 20
    length = 60
    angle = float(row['steer_cmd_rad']) if pd.notna(row['steer_cmd_rad']) else 0.0
    ex = int(cx + length * np.sin(angle))
    ey = int(cy - length * np.cos(angle))
    colour = (0, 220, 0) if valid else (0, 0, 220)
    cv2.line(canvas, (cx, cy), (ex, ey), colour, 3)
    cv2.circle(canvas, (cx, cy), 4, colour, -1)

    speed = row['speed_cmd_mps'] if pd.notna(row['speed_cmd_mps']) else 0.0
    text = (f'{frame_no + 1}/{total}  steer={np.degrees(angle):+.1f} deg  '
           f'speed={speed:+.2f} m/s  {"VALID" if valid else "REJECTED"}')
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def run(dataset_dir: Path, episode_id: str) -> None:
    frames = pd.read_parquet(dataset_dir / 'frames.parquet')
    frames = resolve_dataset_paths(frames, dataset_dir)
    ep = frames[frames['episode_id'] == episode_id].sort_values('frame_index').reset_index(drop=True)
    if ep.empty:
        raise SystemExit(f'no frames found for episode_id={episode_id}')

    curation = _load_curation(dataset_dir)
    i = 0
    n = len(ep)

    window = f'mobilebot-view: {episode_id}'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        row = ep.iloc[i]
        key_frame = f'{episode_id}:{int(row["frame_index"])}'
        valid = curation.get(key_frame, {}).get('valid', bool(row['valid']))

        image = cv2.imread(row['image_path'])
        if image is None:
            image = np.zeros((120, 160, 3), dtype=np.uint8)
        canvas = _draw_overlay(image, row, valid, i, n)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            _save_curation(dataset_dir, curation)
            break
        elif key == ord('s'):
            _save_curation(dataset_dir, curation)
        elif key == ord('d'):
            i = min(i + 1, n - 1)
        elif key == ord('a'):
            i = max(i - 1, 0)
        elif key == ord('D'):
            i = min(i + 10, n - 1)
        elif key == ord('A'):
            i = max(i - 10, 0)
        elif key == ord(' '):
            curation[key_frame] = {'valid': not valid, 'reject_reason': ''}
        elif key == ord('r'):
            reason = input(f'reject reason for frame {i} ({key_frame}): ')
            curation[key_frame] = {'valid': False, 'reject_reason': reason}

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--episode', required=True,
                        help='episode_id, e.g. episode_0001_14-05-02')
    args = parser.parse_args()
    run(args.dataset, args.episode)


if __name__ == '__main__':
    main()
