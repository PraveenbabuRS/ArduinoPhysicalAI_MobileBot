"""Layer A -> Layer B extraction.

Deterministic and idempotent: re-running against the same session directory
with the same code and parameters reproduces the same Layer B output byte for
byte (image bytes are referenced, not re-encoded). This is what makes it safe
to delete and rebuild the whole training set whenever the crop or resize
target changes.

Usage:
    mobilebot-extract --session ~/mobilebot_data/2026-08-17T14-03-00Z \\
                       --out ~/mobilebot_train/2026-08-17T14-03-00Z
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .mcap_io import read_episode, list_episodes, DecodedMessage

# Two full control periods (50 Hz => 20 ms) each side is the outer bound for
# calling an image/command pair "aligned"; wider than that and the label no
# longer describes what the frame shows.
DEFAULT_ALIGN_TOLERANCE_NS = 25_000_000
NOMINAL_FRAME_PERIOD_NS = 50_000_000  # 20 Hz target recording rate
DROP_GAP_FACTOR = 1.5


def _extractor_version() -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=Path(__file__).parent, capture_output=True,
                             text=True, timeout=2.0)
        return out.stdout.strip() if out.returncode == 0 else 'unknown'
    except Exception:  # noqa: BLE001
        return 'unknown'


class NearestIndex:
    """Nearest-neighbour lookup by timestamp over a sorted message list."""

    def __init__(self, messages: list[DecodedMessage]) -> None:
        self._messages = sorted(messages, key=lambda m: m.log_time_ns)
        self._times = [m.log_time_ns for m in self._messages]

    def nearest(self, t_ns: int, tolerance_ns: int) -> tuple[DecodedMessage | None, int]:
        if not self._times:
            return None, tolerance_ns + 1
        i = bisect.bisect_left(self._times, t_ns)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(self._times)]
        if not candidates:
            return None, tolerance_ns + 1
        best = min(candidates, key=lambda j: abs(self._times[j] - t_ns))
        residual = abs(self._times[best] - t_ns)
        if residual > tolerance_ns:
            return None, residual
        return self._messages[best], residual


@dataclass
class EpisodeSummary:
    episode_id: str
    session_id: str
    frame_count: int
    dropped_frame_count: int
    duration_s: float
    mean_align_residual_ms: float
    max_align_residual_ms: float
    unaligned_frame_count: int
    steer_p05_rad: float
    steer_p50_rad: float
    steer_p95_rad: float
    accepted: bool
    reject_reason: str


def extract_episode(bag_dir: Path, images_out: Path, dataset_root: Path,
                    align_tolerance_ns: int) -> tuple[list[dict], EpisodeSummary, dict]:
    by_topic: dict[str, list[DecodedMessage]] = {}
    for m in read_episode(bag_dir):
        by_topic.setdefault(m.topic, []).append(m)

    session_meta_msgs = by_topic.get('session/metadata', [])
    session_meta = _msg_to_dict(session_meta_msgs[0].msg) if session_meta_msgs else {}

    images = sorted(by_topic.get('image_raw/compressed', []), key=lambda m: m.log_time_ns)
    cmd_index = NearestIndex(by_topic.get('cmd_drive', []))
    teleop_index = NearestIndex(by_topic.get('teleop/cmd', []))
    mcu_index = NearestIndex(by_topic.get('mcu/status', []))
    imu_index = NearestIndex(by_topic.get('imu/data', []))

    images_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    residuals: list[float] = []
    unaligned = 0
    dropped_frames = 0

    for idx, img in enumerate(images):
        if idx > 0:
            gap = img.log_time_ns - images[idx - 1].log_time_ns
            if gap > NOMINAL_FRAME_PERIOD_NS * DROP_GAP_FACTOR:
                dropped_frames += round(gap / NOMINAL_FRAME_PERIOD_NS) - 1

        cmd, cmd_residual = cmd_index.nearest(img.log_time_ns, align_tolerance_ns)
        teleop, _ = teleop_index.nearest(img.log_time_ns, align_tolerance_ns)
        mcu, _ = mcu_index.nearest(img.log_time_ns, align_tolerance_ns)
        imu, _ = imu_index.nearest(img.log_time_ns, align_tolerance_ns)

        valid = cmd is not None
        if not valid:
            unaligned += 1
        else:
            residuals.append(cmd_residual / 1e6)

        image_filename = f'{idx:06d}.jpg'
        image_path = (images_out / image_filename).resolve()
        _write_compressed_image(img.msg, image_path)

        rows.append({
            'episode_id': bag_dir.name,
            'frame_index': idx,
            'global_index': None,  # filled in by the caller across episodes
            't_capture_ns': img.log_time_ns,
            't_command_ns': cmd.log_time_ns if cmd else None,
            'align_residual_ms': (cmd_residual / 1e6) if cmd else None,
            # Stored RELATIVE to the dataset's own directory (see
            # manifest.json's images_root), not absolute. An absolute path
            # baked in here breaks the instant this dataset is copied, moved,
            # or used from a different machine -- resolve_dataset_paths() in
            # paths.py is what turns this back into a real path at load time.
            'image_path': str(image_path.relative_to(dataset_root.resolve())),
            'steer_cmd_rad': _get(cmd, 'drive.steering_angle'),
            'speed_cmd_mps': _get(cmd, 'drive.speed'),
            'steer_raw_rad': _get(teleop, 'drive.steering_angle'),
            'speed_raw_mps': _get(teleop, 'drive.speed'),
            'deadman': _get(teleop, 'deadman'),
            'steer_hold': _get(teleop, 'steer_hold'),
            'control_source': 'human',  # ARMED_AUTO not reachable yet; revisit
            'mcu_state': _get(mcu, 'state'),
            'servo_deg': _get(mcu, 'servo_deg'),
            'imu_ax': _get(imu, 'linear_acceleration.x'),
            'imu_ay': _get(imu, 'linear_acceleration.y'),
            'imu_az': _get(imu, 'linear_acceleration.z'),
            'imu_gx': _get(imu, 'angular_velocity.x'),
            'imu_gy': _get(imu, 'angular_velocity.y'),
            'imu_gz': _get(imu, 'angular_velocity.z'),
            'valid': valid,
            'reject_reason': '' if valid else 'no command within alignment tolerance',
        })

    duration_s = (images[-1].log_time_ns - images[0].log_time_ns) / 1e9 if images else 0.0
    steer_series = pd.Series([r['steer_cmd_rad'] for r in rows if r['steer_cmd_rad'] is not None])

    summary = EpisodeSummary(
        episode_id=bag_dir.name,
        session_id=session_meta.get('session_id', ''),
        frame_count=len(rows),
        dropped_frame_count=dropped_frames,
        duration_s=duration_s,
        mean_align_residual_ms=float(pd.Series(residuals).mean()) if residuals else float('nan'),
        max_align_residual_ms=float(pd.Series(residuals).max()) if residuals else float('nan'),
        unaligned_frame_count=unaligned,
        steer_p05_rad=float(steer_series.quantile(0.05)) if len(steer_series) else float('nan'),
        steer_p50_rad=float(steer_series.quantile(0.50)) if len(steer_series) else float('nan'),
        steer_p95_rad=float(steer_series.quantile(0.95)) if len(steer_series) else float('nan'),
        accepted=True,   # extraction never rejects; that is a curation decision
        reject_reason='',
    )

    return rows, summary, session_meta


def _get(msg_wrapper, dotted_path: str):
    if msg_wrapper is None:
        return None
    obj = msg_wrapper.msg
    for part in dotted_path.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _msg_to_dict(msg) -> dict:
    # mcap_ros2 decodes into a generic dynamic type; __slots__/dir() based
    # introspection keeps this independent of the exact decoder class.
    result = {}
    for name in dir(msg):
        if name.startswith('_'):
            continue
        value = getattr(msg, name)
        if callable(value):
            continue
        result[name] = value
    return result


def _write_compressed_image(msg, out_path: Path) -> None:
    # CompressedImage.data is already JPEG bytes off the wire -- write as-is,
    # never re-encode. Re-encoding here would silently change the training
    # images every time the extractor runs.
    data = bytes(msg.data)
    out_path.write_bytes(data)


def extract_session(session_dir: Path, out_dir: Path, align_tolerance_ns: int) -> None:
    episodes = list_episodes(session_dir)
    if not episodes:
        raise SystemExit(f'no episode_* directories found under {session_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / 'images'

    all_rows: list[dict] = []
    all_summaries: list[dict] = []
    session_meta: dict = {}
    global_index = 0

    for ep_dir in episodes:
        rows, summary, ep_session_meta = extract_episode(
            ep_dir, images_root / ep_dir.name, out_dir, align_tolerance_ns)
        for r in rows:
            r['global_index'] = global_index
            global_index += 1
        all_rows.extend(rows)
        all_summaries.append(asdict(summary))
        if not session_meta and ep_session_meta:
            session_meta = ep_session_meta

        print(f'  {ep_dir.name}: {summary.frame_count} frames, '
              f'{summary.unaligned_frame_count} unaligned, '
              f'{summary.dropped_frame_count} dropped, '
              f'align residual mean/max {summary.mean_align_residual_ms:.1f}/'
              f'{summary.max_align_residual_ms:.1f} ms')

    frames_df = pd.DataFrame(all_rows)
    episodes_df = pd.DataFrame(all_summaries)

    frames_df.to_parquet(out_dir / 'frames.parquet', index=False)
    episodes_df.to_parquet(out_dir / 'episodes.parquet', index=False)

    manifest = {
        'extractor_version': _extractor_version(),
        'align_tolerance_ns': align_tolerance_ns,
        'source_session_dir': str(session_dir),
        # The anchor paths.py joins stored relative image_path values against.
        # This is out_dir itself (the dataset root), not the images/
        # subfolder -- stored paths already include 'images/episode_.../...'.
        'images_root': str(out_dir.resolve()),
        'episode_count': len(episodes),
        'frame_count': len(frames_df),
        'session_metadata': session_meta,
        'frames_content_hash': hashlib.sha256(
            frames_df.to_csv(index=False).encode()).hexdigest(),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))

    print(f'\nextracted {len(frames_df)} frames from {len(episodes)} episodes -> {out_dir}')
    if frames_df['valid'].eq(False).any():
        n_bad = int((~frames_df['valid']).sum())
        print(f'WARNING: {n_bad} frames have no aligned command and are marked invalid')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path,
                        help='session directory containing episode_* subdirs')
    parser.add_argument('--out', required=True, type=Path,
                        help='output directory for the Layer B dataset')
    parser.add_argument('--align-tolerance-ms', type=float, default=25.0,
                        help='max image/command time gap to call them aligned')
    args = parser.parse_args()

    extract_session(args.session, args.out, int(args.align_tolerance_ms * 1e6))


if __name__ == '__main__':
    main()
