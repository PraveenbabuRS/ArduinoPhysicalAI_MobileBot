"""Inference backend behind a Strategy interface.

Two implementations for now:
    OnnxBackend  -- ONNX Runtime CPU execution provider, runs policy.onnx as
                    exported, no conversion step.
    DummyBackend -- returns zero steering / idle unconditionally. Useful for
                    exercising the ROS plumbing (topics, rates, arbiter
                    interaction) without a camera or a model at all.

A TFLite+XNNPACK backend is the natural next implementation if ONNX Runtime's
measured latency doesn't clear the control-loop budget -- it plugs in here
without touching policy_node.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class InferenceResult:
    steering_rad: float
    speed_class: int
    inference_seconds: float


class InferenceBackend(Protocol):
    preprocessing_version: str
    num_speed_classes: int

    def infer(self, nchw_image: np.ndarray) -> InferenceResult: ...


class OnnxBackend:

    def __init__(self, model_path: Path, intra_op_threads: int = 1,
                inter_op_threads: int = 1) -> None:
        import onnxruntime as ort

        sidecar_path = model_path.with_suffix('.json')
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f'{sidecar_path} not found -- policy.onnx must ship with the '
                'metadata sidecar written by mobilebot-export. Without it there '
                'is no way to verify the preprocessing contract matches.')
        sidecar = json.loads(sidecar_path.read_text())

        self.preprocessing_version: str = sidecar['preprocessing_version']
        self.num_speed_classes: int = sidecar['num_speed_classes']
        self._input_shape = tuple(sidecar['input_shape'])

        # ONNX Runtime's default pool size and behaviour under contention with
        # the rest of the ROS graph is not yet well characterised: isolated
        # sweeps favour 2 threads over 1 (14ms vs 26ms -- parallelism helps
        # when cores are free), but the previous full-graph improvement to
        # ~20Hz came from disabling data_recorder, not from thread pinning,
        # which was never actually tested under real contention. Sweep with
        # benchmark_inference --threads N while the FULL graph (recorder
        # included) is running to find the true contended optimum -- it can
        # differ from the isolated one.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_threads
        session_options.inter_op_num_threads = inter_op_threads

        # CPU EP explicitly -- this is the QRB2210, there is no GPU EP path
        # for ONNX Runtime on this hardware.
        self._session = ort.InferenceSession(
            str(model_path), sess_options=session_options,
            providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, nchw_image: np.ndarray) -> InferenceResult:
        if nchw_image.shape != self._input_shape:
            raise ValueError(
                f'input shape {nchw_image.shape} does not match the shape '
                f'this model was exported for {self._input_shape}')

        started = time.perf_counter()
        steering, speed_class = self._session.run(
            None, {self._input_name: nchw_image.astype(np.float32)})
        elapsed = time.perf_counter() - started

        return InferenceResult(
            steering_rad=float(np.asarray(steering).reshape(-1)[0]),
            speed_class=int(np.asarray(speed_class).reshape(-1)[0]),
            inference_seconds=elapsed,
        )


class DummyBackend:
    """Zero steering, idle speed, unconditionally. Validates the control path
    (topics, rates, arbiter mode transitions) before trusting a real model, or
    isolates ROS/DDS overhead from model inference cost when benchmarking.
    """

    def __init__(self, num_speed_classes: int = 2,
                preprocessing_version: str = 'dummy-no-contract') -> None:
        self.preprocessing_version = preprocessing_version
        self.num_speed_classes = num_speed_classes

    def infer(self, nchw_image: np.ndarray) -> InferenceResult:
        started = time.perf_counter()
        elapsed = time.perf_counter() - started
        return InferenceResult(steering_rad=0.0, speed_class=0, inference_seconds=elapsed)


def build_backend(kind: str, model_path: Path | None,
                  intra_op_threads: int = 1, inter_op_threads: int = 1) -> InferenceBackend:
    if kind == 'onnx':
        if model_path is None:
            raise ValueError('model_path is required for the onnx backend')
        return OnnxBackend(model_path, intra_op_threads, inter_op_threads)
    if kind == 'dummy':
        return DummyBackend()
    raise ValueError(f'unknown backend kind: {kind!r} (expected "onnx" or "dummy")')
