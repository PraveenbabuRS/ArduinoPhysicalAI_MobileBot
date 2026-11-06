"""Dual-head PilotNet-scale CNN.

Sized deliberately small. The QRB2210 has no NPU -- inference runs on the
Cortex-A53 via LiteRT+XNNPACK -- and the control loop needs 20-30 Hz. The
DeepPicar measurement anchors this: a ~250k-parameter PilotNet-class net ran
18.5 ms/frame on a Cortex-A53 at 1.2 GHz with a 2018-era runtime. This chip is
2.0 GHz with a far better runtime, so this architecture has real margin. A
MobileNet-class backbone would not.

Two heads on a shared conv trunk:
    steering : linear regression, 1 output, radians
    speed    : classification over discrete setpoint classes

Speed is classified rather than regressed because the teleop throttle is a
stepped setpoint, not a continuum -- the label genuinely is categorical, and
classification is the more stable choice for it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PilotNet(nn.Module):

    def __init__(self, num_speed_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_speed_classes = num_speed_classes

        # Input: (N, 3, 120, 160)
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ELU(),
            nn.Conv2d(24, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ELU(),
        )

        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 3, 120, 160)).numel()

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_flat, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
        )

        self.steering_head = nn.Linear(50, 1)
        self.speed_head = nn.Linear(50, num_speed_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(self.features(x))
        return self.steering_head(z).squeeze(-1), self.speed_head(z)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ExportWrapper(nn.Module):
    """Inference-time wrapper: emits steering (rad) and the speed class index.

    Argmax is folded into the graph so the on-robot node does not have to
    reimplement it -- one fewer place for train/deploy divergence to hide.
    """

    def __init__(self, model: PilotNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steering, speed_logits = self.model(x)
        return steering, torch.argmax(speed_logits, dim=-1).to(torch.int32)
