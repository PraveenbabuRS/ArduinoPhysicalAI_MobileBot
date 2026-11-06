"""Render training curves from history.json. Shared by the CLI (saves a PNG)
and the dashboard (renders the same figure live).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(history: list[dict]):
    if not history:
        raise ValueError('empty history -- nothing to plot')

    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('Training progress')

    ax = axes[0, 0]
    ax.plot(epochs, [r['steer_mae_deg'] for r in history], label='val steer MAE (deg)')
    ax.plot(epochs, [r['steer_rmse_deg'] for r in history], label='val steer RMSE (deg)',
           alpha=0.6)
    best_epoch = min(history, key=lambda r: r['steer_mae_deg'])['epoch']
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.5,
              label=f'best (epoch {best_epoch})')
    ax.set_xlabel('epoch'); ax.set_ylabel('degrees'); ax.legend(fontsize=8)
    ax.set_title('Steering error')

    ax = axes[0, 1]
    ax.plot(epochs, [r['steer_within_3deg_pct'] for r in history], label='within 3 deg')
    ax.plot(epochs, [r['steer_direction_agreement_pct'] for r in history],
           label='direction agreement')
    ax.plot(epochs, [r['speed_accuracy_pct'] for r in history], label='speed accuracy')
    ax.set_xlabel('epoch'); ax.set_ylabel('%'); ax.set_ylim(0, 100); ax.legend(fontsize=8)
    ax.set_title('Driving-oriented metrics')

    ax = axes[1, 0]
    ax.plot(epochs, [r['train_loss'] for r in history], label='total')
    ax.plot(epochs, [r['train_steer_loss'] for r in history], label='steering', alpha=0.7)
    ax.plot(epochs, [r['train_speed_loss'] for r in history], label='speed', alpha=0.7)
    ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend(fontsize=8)
    ax.set_title('Training loss')

    ax = axes[1, 1]
    ax.plot(epochs, [r['lr'] for r in history])
    ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel('learning rate (log)')
    ax.set_title('LR schedule -- a flat floor here means the model stopped '
                "learning, not that it's done learning")

    fig.tight_layout()
    return fig


def save_plot(history: list[dict], out_path: Path) -> None:
    fig = plot_history(history)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
