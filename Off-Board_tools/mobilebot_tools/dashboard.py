"""Interactive training dashboard.

Run with:
    streamlit run $(python -c "import mobilebot_tools.dashboard as d; print(d.__file__)")

or, more simply, from the host_tools directory:
    streamlit run mobilebot_tools/dashboard.py

Configure the dataset, output directory, and hyperparameters in the sidebar,
optionally point --resume-from at an existing last.pt to continue training the
same model on a new session's data, then press Start. Training runs on a
background thread; this script polls a queue and redraws the charts as epochs
complete. Closing the browser tab does not stop training -- use the Stop
button, or Ctrl-C the terminal running `streamlit run`.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import streamlit as st

from mobilebot_tools.plotting import plot_history
from mobilebot_tools.training_core import TrainConfig, run_training

st.set_page_config(page_title='mobilebot training', layout='wide')


def _training_thread(cfg: TrainConfig, q: queue.Queue, stop_event: threading.Event) -> None:
    def on_epoch(record: dict) -> None:
        q.put(('epoch', record))
        if stop_event.is_set():
            raise KeyboardInterrupt('stopped from dashboard')

    try:
        result = run_training(cfg, on_epoch=on_epoch)
        q.put(('done', result))
    except Exception as exc:  # noqa: BLE001 -- surface it in the UI, don't just die silently
        q.put(('error', str(exc)))


def _init_state() -> None:
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('running', False)
    st.session_state.setdefault('thread', None)
    st.session_state.setdefault('queue', None)
    st.session_state.setdefault('stop_event', None)
    st.session_state.setdefault('final_message', None)


_init_state()

st.title('mobilebot -- policy training')

with st.sidebar:
    st.header('Dataset')
    dataset = st.text_input('Curated dataset directory', value=str(Path.home() / 'Documents' / 'Arduino_Challenge_Data' / 'ExtractedData' / 'Curated'))
    out_dir = st.text_input('Output directory (run name)', value=str(Path.home()/ 'Documents' / 'Arduino_Challenge_Data/' / 'runs' / 'v1'))
    resume_from = st.text_input(
        'Resume from checkpoint (optional)', value='',
        help='Path to last.pt from a previous run. Leave empty to start fresh. '
             'Use this to keep training the same model as new sessions come in.')

    st.header('Hyperparameters')
    speed_classes = st.selectbox('Speed classes', [2, 3], index=0)
    epochs = st.number_input('Epochs (this run)', min_value=1, value=60, step=10,
                            help='A ceiling, not a target -- early stopping will '
                                 'end the run sooner if it stops improving.')
    batch_size = st.number_input('Batch size', min_value=1, value=64, step=8)
    lr = st.number_input('Learning rate', min_value=1e-6, value=1e-3, format='%.1e',
                        help='Ignored when resuming; optimiser state is restored instead.')
    stop_patience = st.number_input('Early-stop patience (epochs)', min_value=1, value=25)
    lr_patience = st.number_input('LR-reduce patience (epochs)', min_value=1, value=5)

    with st.expander('Loss & balancing (advanced)'):
        straight_deg = st.number_input('Straight-bin half-width (deg)', value=5.0)
        straight_keep = st.slider('Fraction of straight frames kept', 0.05, 1.0, 0.35)
        turn_weight = st.number_input('Turn-weight k (1+k|steer|)', value=4.0)
        steer_loss_weight = st.number_input('Steering loss weight', value=1.0)

    st.divider()
    col1, col2 = st.columns(2)
    start_clicked = col1.button('Start / Resume', type='primary',
                               disabled=st.session_state.running)
    stop_clicked = col2.button('Stop', disabled=not st.session_state.running)

if stop_clicked and st.session_state.stop_event is not None:
    st.session_state.stop_event.set()
    st.warning('Stop requested -- will halt after the current epoch finishes.')

if start_clicked and not st.session_state.running:
    dataset_path = Path(dataset)
    if not (dataset_path / 'frames.parquet').exists():
        st.error(f'{dataset_path} has no frames.parquet -- check the path.')
    else:
        cfg = TrainConfig(
            dataset=dataset_path, out=Path(out_dir),
            resume_from=Path(resume_from) if resume_from.strip() else None,
            speed_classes=int(speed_classes), epochs=int(epochs),
            batch_size=int(batch_size), lr=float(lr),
            straight_deg=float(straight_deg), straight_keep=float(straight_keep),
            turn_weight=float(turn_weight), steer_loss_weight=float(steer_loss_weight),
            lr_patience=int(lr_patience), stop_patience=int(stop_patience),
        )
        st.session_state.history = []
        st.session_state.final_message = None
        st.session_state.queue = queue.Queue()
        st.session_state.stop_event = threading.Event()
        st.session_state.thread = threading.Thread(
            target=_training_thread,
            args=(cfg, st.session_state.queue, st.session_state.stop_event),
            daemon=True)
        st.session_state.running = True
        st.session_state.thread.start()

status = st.empty()
metrics_row = st.empty()
chart_placeholder = st.empty()


def _render(history: list[dict]) -> None:
    if not history:
        status.info('No training run yet. Configure the sidebar and press Start.')
        return
    latest = history[-1]
    cols = metrics_row.columns(5)
    cols[0].metric('Epoch', latest['epoch'])
    cols[1].metric('Val steer MAE', f'{latest["steer_mae_deg"]:.2f} deg')
    cols[2].metric('Direction agreement', f'{latest["steer_direction_agreement_pct"]:.1f}%')
    cols[3].metric('Speed accuracy', f'{latest["speed_accuracy_pct"]:.1f}%')
    cols[4].metric('Learning rate', f'{latest["lr"]:.1e}')
    chart_placeholder.pyplot(plot_history(history))


_render(st.session_state.history)

if st.session_state.running:
    status.info('Training in progress...')
    q = st.session_state.queue
    # Drain the queue and redraw roughly twice a second. This blocks the script
    # for the duration of the run, which is the simplest reliable pattern for a
    # live-updating Streamlit view without a separate process.
    while st.session_state.running:
        drained = False
        try:
            while True:
                kind, payload = q.get_nowait()
                drained = True
                if kind == 'epoch':
                    st.session_state.history.append(payload)
                elif kind == 'done':
                    st.session_state.running = False
                    if payload['stopped_by_request']:
                        st.session_state.final_message = (
                            f'Stopped by user at epoch {payload["history"][-1]["epoch"]}. '
                            f'Best val steering MAE so far: {payload["best_metric"]:.2f} deg')
                    else:
                        st.session_state.final_message = (
                            f'Finished. Best val steering MAE: {payload["best_metric"]:.2f} deg'
                            + (' (stopped early -- plateaued)' if payload['stopped_early'] else ''))
                elif kind == 'error':
                    st.session_state.running = False
                    st.session_state.final_message = f'Training error: {payload}'
        except queue.Empty:
            pass

        if drained:
            _render(st.session_state.history)
        time.sleep(0.5)

    if st.session_state.final_message:
        status.success(st.session_state.final_message)
    st.rerun()  # one clean final redraw with buttons re-enabled
