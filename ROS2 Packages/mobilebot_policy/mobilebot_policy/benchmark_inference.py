"""Pure inference-cost benchmark. No ROS, no camera, no rclpy.

policy_node's own rolling stats measure the full pipeline (decode + preprocess
+ inference) as it actually runs on the robot -- that is the number that
matters operationally. This script isolates just the ONNX Runtime forward
pass, which is useful for separating "the model itself is too slow" from
"JPEG decode is the bottleneck" when the full-pipeline number disappoints.

Usage:
    benchmark_inference --model ~/runs/v1/policy.onnx --iterations 200

    # against real frames instead of synthetic noise, more representative
    # of actual branch-prediction/cache behaviour on real image statistics:
    benchmark_inference --model ~/runs/v1/policy.onnx --dataset ~/curated_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Same reasoning as policy_node.py: an unconstrained thread pool measured 3.4x
# SLOWER in the full graph than a pinned one, due to contention with the other
# ROS nodes. --threads still lets you sweep and confirm for your own graph.
import cv2
cv2.setNumThreads(1)


def _load_real_frames(dataset_dir: Path, n: int, input_shape: tuple) -> list:
    import pandas as pd
    # Deliberately does not import mobilebot_tools -- this package must not
    # depend on the training-side pip package. Decode/crop/resize is done
    # inline with cv2 directly, mirroring preprocess.py's contract exactly.
    import cv2

    from mobilebot_policy.preprocess import DEFAULT, preprocess_bgr, to_nchw

    frames = pd.read_parquet(dataset_dir / 'frames.parquet')
    sample = frames.sample(min(n, len(frames)), random_state=0)

    tensors = []
    for path in sample['image_path']:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = preprocess_bgr(img, DEFAULT)
        tensors.append(to_nchw(rgb).astype(np.float32))
    if not tensors:
        raise SystemExit(f'no readable images found under {dataset_dir}')
    return tensors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--dataset', type=Path, default=None,
                   help='curated dataset dir; use real frames instead of random noise')
    p.add_argument('--threads', type=int, default=1,
                   help='ONNX Runtime intra-op thread count. Default 1: measured 3.4x '
                        'faster than the ORT default when competing with the rest of '
                        'the ROS graph for the same 4 cores. Try 2 and 4 explicitly '
                        'to confirm for your own setup -- and re-sweep with the full '
                        'graph running, not just standalone, since standalone numbers '
                        'do not reflect real contention.')
    p.add_argument('--inter-threads', type=int, default=1)
    args = p.parse_args()

    import onnxruntime as ort

    sidecar = json.loads(args.model.with_suffix('.json').read_text())
    input_shape = tuple(sidecar['input_shape'])
    print(f'model: {args.model}')
    print(f'input shape: {input_shape}')
    print(f'preprocessing_version: {sidecar["preprocessing_version"]}')

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session_options.inter_op_num_threads = args.inter_threads
    session = ort.InferenceSession(str(args.model), sess_options=session_options,
                                   providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    if args.dataset is not None:
        tensors = _load_real_frames(args.dataset, args.iterations, input_shape)
        print(f'using {len(tensors)} real frames from {args.dataset}')
    else:
        rng = np.random.default_rng(0)
        tensors = [rng.random(input_shape, dtype=np.float32) for _ in range(args.iterations)]
        print(f'using {len(tensors)} synthetic random frames (no --dataset given)')

    def _sample() -> np.ndarray:
        return tensors[np.random.randint(len(tensors))]

    for _ in range(args.warmup):
        session.run(None, {input_name: _sample()})

    latencies = []
    for _ in range(args.iterations):
        x = _sample()
        started = time.perf_counter()
        session.run(None, {input_name: x})
        latencies.append(time.perf_counter() - started)

    latencies = np.array(latencies) * 1000.0  # ms
    print(f'\n{args.iterations} iterations (after {args.warmup} warmup):')
    print(f'  min    {latencies.min():.2f} ms')
    print(f'  mean   {latencies.mean():.2f} ms')
    print(f'  p50    {np.percentile(latencies, 50):.2f} ms')
    print(f'  p95    {np.percentile(latencies, 95):.2f} ms')
    print(f'  max    {latencies.max():.2f} ms')
    print(f'  implied max rate: {1000.0 / latencies.mean():.1f} Hz (model-only, no decode/ROS)')

    target_hz = 20.0
    if latencies.mean() > 1000.0 / target_hz:
        print(f'\nWARNING: mean latency exceeds the {1000.0 / target_hz:.0f} ms budget '
             f'for {target_hz:.0f} Hz. This is model-only cost -- the full pipeline '
             f'(JPEG decode + preprocess + this) will be slower still. Check '
             f'policy_node\'s own stats log for the end-to-end number.')


if __name__ == '__main__':
    main()
