"""MCAP reading helpers shared by extract, validate, and view.

Deliberately independent of ROS: reads message schemas embedded in the MCAP
file itself via mcap-ros2-support, so this package runs on a plain laptop
Python environment with no ROS install.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


@dataclass
class DecodedMessage:
    topic: str
    ros_type: str
    log_time_ns: int
    publish_time_ns: int
    msg: object  # decoded ROS 2 message, attribute access matches the .msg fields


def read_episode(bag_dir: Path) -> Iterator[DecodedMessage]:
    """Yield every message in one episode's MCAP file, in log-time order.

    `bag_dir` is the directory rosbag2 created (it contains the .mcap file and
    metadata.yaml); this locates the actual .mcap file inside it.
    """
    mcap_path = _find_mcap_file(bag_dir)
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages():
            yield DecodedMessage(
                topic=channel.topic,
                ros_type=schema.name if schema else '',
                log_time_ns=message.log_time,
                publish_time_ns=message.publish_time,
                msg=decoded,
            )


def _find_mcap_file(bag_dir: Path) -> Path:
    if bag_dir.is_file() and bag_dir.suffix == '.mcap':
        return bag_dir
    candidates = sorted(bag_dir.glob('*.mcap'))
    if not candidates:
        raise FileNotFoundError(f'no .mcap file found under {bag_dir}')
    return candidates[0]


def list_episodes(session_dir: Path) -> list[Path]:
    """Episode directories in a session, sorted by their index prefix."""
    return sorted(
        p for p in session_dir.iterdir()
        if p.is_dir() and p.name.startswith('episode_')
    )


def episode_id(bag_dir: Path) -> str:
    return bag_dir.name
