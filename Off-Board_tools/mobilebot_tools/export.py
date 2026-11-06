"""Export a trained checkpoint to ONNX, then verify the export numerically.

The verification step is not optional. A silent transpose, a normalisation
mismatch, or an op that converted subtly differently produces a model that
loads cleanly, runs at the right speed, and drives into a hedge -- with
nothing in any log to explain it. Comparing PyTorch and ONNX Runtime outputs on
real frames costs seconds and closes that whole failure class.

Opset is pinned. Embedded runtime versions drift, and an opset mismatch can
silently change op semantics rather than failing loudly.

Usage:
    mobilebot-export --checkpoint ~/.../runs/v1/best.pt --out ~/.../runs/v1/policy.onnx \\
                      --verify-dataset ~/.../session_curated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mobilebot_tools.model import PilotNet, ExportWrapper
from mobilebot_tools.preprocess import DEFAULT as PREPROC, preprocess_file

OPSET = 17


def export(checkpoint: Path, out_path: Path, verify_dataset: Path | None,
          n_verify: int = 32) -> None:
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    num_classes = ckpt['num_speed_classes']

    model = PilotNet(num_speed_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    wrapper = ExportWrapper(model).eval()
    dummy = torch.zeros(1, 3, PREPROC.resize_height, PREPROC.resize_width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=['image'],
        output_names=['steering_rad', 'speed_class'],
        opset_version=OPSET,
        dynamic_axes=None,  # fixed 1x3x120x160 -- embedded runtimes prefer static
    )
    print(f'exported (opset {OPSET}) -> {out_path}')

    sidecar = {
        'preprocessing_version': ckpt['preprocessing_version'],
        'input_shape': [1, 3, PREPROC.resize_height, PREPROC.resize_width],
        'input_layout': 'NCHW, RGB, float32, [0,1]',
        'crop': {'x': PREPROC.crop_x, 'y': PREPROC.crop_y,
                 'width': PREPROC.crop_width, 'height': PREPROC.crop_height},
        'outputs': {'steering_rad': 'float32 scalar',
                    'speed_class': f'int32 index into {num_classes} classes'},
        'num_speed_classes': num_classes,
        'opset': OPSET,
        'torch_version': torch.__version__,
        'trained_epoch': ckpt.get('epoch'),
        'val_metrics': ckpt.get('metrics'),
    }
    sidecar_path = out_path.with_suffix('.json')
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f'metadata -> {sidecar_path}')

    if verify_dataset is None:
        print('WARNING: skipped numeric verification (--verify-dataset not given). '
             'Do not deploy an unverified export.')
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print('WARNING: onnxruntime not installed; cannot verify. '
             '`pip install onnxruntime` then re-run.')
        return

    import pandas as pd
    frames = pd.read_parquet(verify_dataset / 'frames.parquet')
    sample = frames.sample(min(n_verify, len(frames)), random_state=0)

    session = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    max_steer_diff = 0.0
    class_mismatches = 0

    for _, row in sample.iterrows():
        img = preprocess_file(row['image_path'], PREPROC)
        x = np.transpose(img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        with torch.no_grad():
            t_steer, t_class = wrapper(torch.from_numpy(x))

        o_steer, o_class = session.run(None, {'image': x})

        max_steer_diff = max(max_steer_diff,
                             abs(float(t_steer.item()) - float(o_steer.reshape(-1)[0])))
        if int(t_class.item()) != int(np.asarray(o_class).reshape(-1)[0]):
            class_mismatches += 1

    print(f'\nverification over {len(sample)} real frames:')
    print(f'  max steering difference : {np.degrees(max_steer_diff):.6f} deg')
    print(f'  speed class mismatches  : {class_mismatches}/{len(sample)}')

    if max_steer_diff > 1e-4 or class_mismatches > 0:
        print('  FAIL: PyTorch and ONNX disagree. Do NOT deploy this export.')
    else:
        print('  PASS: outputs match within tolerance.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--verify-dataset', type=Path, default=None,
                   help='curated dataset dir; real frames are used to verify the export')
    p.add_argument('--n-verify', type=int, default=32)
    args = p.parse_args()
    export(args.checkpoint, args.out, args.verify_dataset, args.n_verify)


if __name__ == '__main__':
    main()
