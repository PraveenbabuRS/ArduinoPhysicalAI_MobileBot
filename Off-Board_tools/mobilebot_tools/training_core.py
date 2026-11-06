"""The actual training loop, extracted from the CLI so the dashboard can drive
it too. mobilebot-train and the Streamlit dashboard both call run_training();
neither reimplements it. Two entry points into one training loop, not two
training loops that can drift apart.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mobilebot_tools.dataset import (DrivingDataset, balance_steering,
                                     class_names, load_splits)
from mobilebot_tools.model import PilotNet
from mobilebot_tools.preprocess import CONTRACT_VERSION, DEFAULT as PREPROC


@dataclass
class TrainConfig:
    dataset: Path
    out: Path
    speed_classes: int = 2
    epochs: int = 60                 # epochs to run THIS invocation, not a lifetime total
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-6
    workers: int = 4
    seed: int = 0
    straight_deg: float = 5.0
    straight_keep: float = 0.35
    turn_weight: float = 4.0
    steer_loss_weight: float = 1.0
    lr_patience: int = 5             # epochs of no improvement before halving LR
    stop_patience: int = 25          # epochs of no improvement before stopping early
    resume_from: Optional[Path] = None


EpochCallback = Callable[[dict], None]


def turn_weighted_l1(pred: torch.Tensor, target: torch.Tensor, k: float) -> torch.Tensor:
    weights = 1.0 + k * target.abs()
    return (weights * (pred - target).abs()).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
            num_classes: int) -> dict:
    model.eval()
    steer_err, steer_true, steer_pred = [], [], []
    speed_correct = 0
    per_class_correct = np.zeros(num_classes, dtype=int)
    per_class_total = np.zeros(num_classes, dtype=int)
    total = 0

    for imgs, steer, speed_cls in loader:
        imgs, steer, speed_cls = imgs.to(device), steer.to(device), speed_cls.to(device)
        pred_steer, speed_logits = model(imgs)
        pred_cls = speed_logits.argmax(dim=-1)

        steer_err.append((pred_steer - steer).abs().cpu().numpy())
        steer_true.append(steer.cpu().numpy())
        steer_pred.append(pred_steer.cpu().numpy())

        speed_correct += (pred_cls == speed_cls).sum().item()
        for c in range(num_classes):
            mask = speed_cls == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (pred_cls[mask] == c).sum().item()
        total += len(steer)

    steer_err = np.concatenate(steer_err)
    steer_true = np.concatenate(steer_true)
    steer_pred = np.concatenate(steer_pred)

    moving = np.abs(steer_true) > np.radians(2.0)
    direction_ok = (np.sign(steer_pred[moving]) == np.sign(steer_true[moving]))

    return {
        'steer_mae_deg': float(np.degrees(steer_err.mean())),
        'steer_rmse_deg': float(np.degrees(np.sqrt((steer_err ** 2).mean()))),
        'steer_within_3deg_pct': float(100 * (np.degrees(steer_err) < 3.0).mean()),
        'steer_within_5deg_pct': float(100 * (np.degrees(steer_err) < 5.0).mean()),
        'steer_direction_agreement_pct': float(100 * direction_ok.mean())
            if moving.any() else float('nan'),
        'speed_accuracy_pct': float(100 * speed_correct / max(total, 1)),
        'speed_per_class_recall_pct': {
            class_names(num_classes)[c]:
                float(100 * per_class_correct[c] / max(per_class_total[c], 1))
            for c in range(num_classes)
        },
    }


def _load_checkpoint_for_resume(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if 'optimizer_state' not in ckpt:
        raise SystemExit(
            f'{path} has no optimizer_state -- it was saved by an older version '
            'of this script and cannot be resumed from. Start fresh, or load '
            'just its weights into a new run manually.')
    return ckpt


def run_training(cfg: TrainConfig, on_epoch: Optional[EpochCallback] = None) -> dict:
    """Runs cfg.epochs additional epochs (not a lifetime total -- if resuming,
    this many MORE epochs are run on top of whatever the checkpoint already
    completed). Returns {'history': [...], 'best_metric': float, 'stopped_early': bool}.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_frames, val_frames = load_splits(Path(cfg.dataset))
    balanced, balance_report = balance_steering(
        train_frames, straight_deg=cfg.straight_deg,
        keep_fraction=cfg.straight_keep, seed=cfg.seed)

    train_ds = DrivingDataset(balanced, cfg.speed_classes, augment=True,
                              cfg=PREPROC, seed=cfg.seed)
    val_ds = DrivingDataset(val_frames, cfg.speed_classes, augment=False, cfg=PREPROC)

    weights = train_ds.class_weights().to(device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers)

    model = PilotNet(num_speed_classes=cfg.speed_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    speed_loss_fn = nn.CrossEntropyLoss(weight=weights)

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_metric = float('inf')
    history: list[dict] = []
    epochs_since_improvement = 0

    if cfg.resume_from is not None:
        ckpt = _load_checkpoint_for_resume(Path(cfg.resume_from), device)
        model.load_state_dict(ckpt['model_state'])
        optimiser.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch']
        best_metric = ckpt.get('best_metric', ckpt['metrics']['steer_mae_deg'])
        history_path = out_dir / 'history.json'
        if history_path.exists():
            history = json.loads(history_path.read_text())
        print(f'resumed from {cfg.resume_from} at epoch {start_epoch}, '
             f'best steer MAE so far {best_metric:.2f} deg')

    def _print(rec: dict) -> None:
        print(f'[{rec["epoch"]:4d}] loss {rec["train_loss"]:.4f}  '
             f'val steer MAE {rec["steer_mae_deg"]:.2f} deg  '
             f'<3deg {rec["steer_within_3deg_pct"]:.1f}%  '
             f'dir {rec["steer_direction_agreement_pct"]:.1f}%  '
             f'speed acc {rec["speed_accuracy_pct"]:.1f}%  '
             f'lr {rec["lr"]:.2e}  ({rec["seconds"]:.1f}s)')

    stopped_early = False
    stopped_by_request = False
    target_epoch = start_epoch + cfg.epochs

    for epoch in range(start_epoch + 1, target_epoch + 1):
        model.train()
        epoch_loss = epoch_steer = epoch_speed = 0.0
        started = time.time()

        for imgs, steer, speed_cls in train_loader:
            imgs, steer, speed_cls = imgs.to(device), steer.to(device), speed_cls.to(device)
            pred_steer, speed_logits = model(imgs)
            steer_loss = turn_weighted_l1(pred_steer, steer, k=cfg.turn_weight)
            speed_loss = speed_loss_fn(speed_logits, speed_cls)
            loss = cfg.steer_loss_weight * steer_loss + speed_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            epoch_steer += steer_loss.item()
            epoch_speed += speed_loss.item()

        n_batches = max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device, cfg.speed_classes)
        scheduler.step(metrics['steer_mae_deg'])

        record = {
            'epoch': epoch,
            'train_loss': epoch_loss / n_batches,
            'train_steer_loss': epoch_steer / n_batches,
            'train_speed_loss': epoch_speed / n_batches,
            'lr': optimiser.param_groups[0]['lr'],
            'seconds': time.time() - started,
            **metrics,
        }
        history.append(record)
        _print(record)

        improved = metrics['steer_mae_deg'] < best_metric
        if improved:
            best_metric = metrics['steer_mae_deg']
            epochs_since_improvement = 0
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimiser.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'num_speed_classes': cfg.speed_classes,
                'preprocessing_version': CONTRACT_VERSION,
                'epoch': epoch,
                'best_metric': best_metric,
                'metrics': metrics,
            }, out_dir / 'best.pt')
        else:
            epochs_since_improvement += 1

        # last.pt always reflects the most recent epoch, improved or not --
        # resuming from best.pt would silently discard any regression the
        # model was in the middle of recovering from.
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimiser.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'num_speed_classes': cfg.speed_classes,
            'preprocessing_version': CONTRACT_VERSION,
            'epoch': epoch,
            'best_metric': best_metric,
            'metrics': metrics,
        }, out_dir / 'last.pt')

        (out_dir / 'history.json').write_text(json.dumps(history, indent=2))

        if on_epoch is not None:
            try:
                on_epoch(record)
            except KeyboardInterrupt:
                # This epoch is already saved (best.pt/last.pt/history.json
                # above are durable); only the epochs after this one are
                # skipped. Falls through to the normal end-of-function
                # bookkeeping below rather than aborting the whole call.
                stopped_by_request = True
                break

        if epochs_since_improvement >= cfg.stop_patience:
            print(f'\nno improvement in {cfg.stop_patience} epochs -- stopping early '
                 f'at epoch {epoch} (best was epoch {epoch - epochs_since_improvement}, '
                 f'{best_metric:.2f} deg). Collecting more/more-varied data will help '
                 f'more than further epochs at this point.')
            stopped_early = True
            break

    cfg_dict = asdict(cfg)
    cfg_dict['dataset'] = str(cfg_dict['dataset'])
    cfg_dict['out'] = str(cfg_dict['out'])
    cfg_dict['resume_from'] = str(cfg_dict['resume_from']) if cfg.resume_from else None
    cfg_dict['best_val_steer_mae_deg'] = best_metric
    cfg_dict['final_epoch'] = history[-1]['epoch'] if history else start_epoch
    cfg_dict['stopped_early'] = stopped_early
    cfg_dict['stopped_by_request'] = stopped_by_request
    (out_dir / 'config.json').write_text(json.dumps(cfg_dict, indent=2))

    best_epoch = min(history, key=lambda r: r['steer_mae_deg'])['epoch'] if history else None
    if history and best_epoch is not None and best_epoch < history[-1]['epoch'] * 0.1 \
            and len(history) > 20 and not stopped_early and not stopped_by_request:
        print(f'\nNOTE: best epoch ({best_epoch}) is very early relative to the '
             f'{len(history)} epochs run. This usually means the LR schedule '
             f'floored out rather than the model genuinely plateauing -- check '
             f'the lr column in history.json.')

    return {'history': history, 'best_metric': best_metric, 'stopped_early': stopped_early,
           'stopped_by_request': stopped_by_request, 'balance_report': balance_report.describe()}
