"""Train the dual-head driving policy.

Usage:
    mobilebot-train --dataset ~/.../session_curated --out ~/.../runs/v1

    # continue training the same model on a new session's data later:
    mobilebot-train --dataset ~/.../session2_curated --out ~/.../runs/v1 \
        --resume-from ~/.../runs/v1/last.pt --epochs 40

This is a thin wrapper over training_core.run_training(); the dashboard
(mobilebot-dashboard) calls the identical function, so the two never drift
into two different training loops.

Loss choices worth knowing before you tune them:

* **Steering uses turn-weighted L1, not MSE.** Published comparison on this
  exact task found MSE gave the worst steering accuracy of the three, because
  the near-zero-dominated distribution plus squared error drags predictions
  toward straight; MAE was better; a turn-emphasising loss beat both. Samples
  are weighted by 1 + k|steer| so corners count for more than straights.

* **Speed uses class-weighted cross-entropy.** Idle is a minority class by
  construction. Unweighted, the network learns to never predict a stop and
  still scores well on accuracy.

* **Validation reports driving-oriented metrics, not just loss.** Offline loss
  correlates weakly with on-road behaviour (measured correlation ~0.39 in the
  literature). Thresholded steering error and direction agreement correlate
  better -- but nothing offline substitutes for a supervised closed-loop run.

* **Training stops when it stops improving, not after a fixed epoch count.**
  --epochs is a ceiling per invocation, not a target; --stop-patience ends the
  run early once steering MAE hasn't improved in that many epochs, so you are
  not waiting out a flat LR floor for the sake of a round number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mobilebot_tools.plotting import save_plot
from mobilebot_tools.training_core import TrainConfig, run_training


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--resume-from', type=Path, default=None,
                   help='checkpoint (last.pt or best.pt) to continue training from')
    p.add_argument('--speed-classes', type=int, default=2, choices=[2, 3])
    p.add_argument('--epochs', type=int, default=60,
                   help='epochs to run THIS invocation (added on top of any '
                        'already completed when resuming), not a lifetime total')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='ignored when resuming -- the optimiser state is restored instead')
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--straight-deg', type=float, default=5.0,
                   help='half-width of the "straight" steering bin, degrees')
    p.add_argument('--straight-keep', type=float, default=0.35,
                   help='fraction of straight-driving frames to retain')
    p.add_argument('--turn-weight', type=float, default=4.0,
                   help='k in the 1+k|steer| loss weighting')
    p.add_argument('--steer-loss-weight', type=float, default=1.0,
                   help='relative weight of steering loss vs speed loss')
    p.add_argument('--lr-patience', type=int, default=5,
                   help='epochs with no improvement before halving LR')
    p.add_argument('--stop-patience', type=int, default=25,
                   help='epochs with no improvement before stopping this run early')
    p.add_argument('--no-plot', action='store_true', help='skip writing plots.png')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = TrainConfig(
        dataset=args.dataset, out=args.out, resume_from=args.resume_from,
        speed_classes=args.speed_classes, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, min_lr=args.min_lr,
        workers=args.workers, seed=args.seed, straight_deg=args.straight_deg,
        straight_keep=args.straight_keep, turn_weight=args.turn_weight,
        steer_loss_weight=args.steer_loss_weight, lr_patience=args.lr_patience,
        stop_patience=args.stop_patience,
    )

    print(f'device: {"cuda" if torch.cuda.is_available() else "cpu"}')
    result = run_training(cfg)

    print(f'\nbest val steering MAE: {result["best_metric"]:.2f} deg')
    print(f'-> {args.out / "best.pt"}  (best checkpoint)')
    print(f'-> {args.out / "last.pt"}  (most recent -- resume from this one)')

    if not args.no_plot and result['history']:
        plot_path = args.out / 'plots.png'
        save_plot(result['history'], plot_path)
        print(f'-> {plot_path}')

    print('\nReminder: offline metrics screen candidates, they do not certify a '
         'policy. Confirm behaviour in a supervised closed-loop run before '
         'trusting any of these numbers.')


if __name__ == '__main__':
    main()
