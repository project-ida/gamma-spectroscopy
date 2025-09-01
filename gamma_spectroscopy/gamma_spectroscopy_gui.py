import argparse
import csv
import ctypes
from math import floor
import sys
from pathlib import Path
import time
from datetime import datetime
from collections import deque

import numpy as np
from pkg_resources import resource_filename

from PyQt5 import uic, QtWidgets, QtCore
import pyqtgraph as pg

from gamma_spectroscopy.picoscope_5000a import PicoScope5000A, INPUT_RANGES
from gamma_spectroscopy.fake_picoscope import FakePicoScope

from gamma_spectroscopy.root_exporter import RootWriter, next_run_id
from gamma_spectroscopy.settings_persistence import attach_settings_store

GUIDE_COLORS = {
    'red': (255, 0, 0, 63),
    'green': (0, 255, 0, 63),
    'blue': (0, 0, 255, 63),
    'purple': (255, 0, 255, 63),
}

# Custom symbol for use in (mostly) histogram plot
histogram_symbol = pg.QtGui.QPainterPath()
histogram_symbol.moveTo(0, -.5)
histogram_symbol.lineTo(0, .5)

PLOT_OPTIONS = {
    'lines': {'A': {'pen': {'color': 'w', 'width': 2.}},
              'B': {'pen': {'color': (255, 200, 0), 'width': 2.}},
             },
    'marks': {'A': {'pen': None, 'symbol': histogram_symbol,
                    'symbolPen': 'w', 'symbolSize': 2},
              'B': {'pen': None, 'symbol': histogram_symbol,
                    'symbolPen': (255, 200, 0), 'symbolSize': 2},
             },
}

def create_callback(signal, scope):
    @ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
    def my_callback(handle, status, parameters):
        try:
            scope._last_ready_mono_ns = time.monotonic_ns()  # <-- monotonic
        except Exception:
            pass
        signal.emit()
    return my_callback


class UserInterface(QtWidgets.QMainWindow):

    POLARITY = ['Positive', 'Negative']
    POLARITY_SIGN = [1, -1]

    COUPLING = ['AC', 'DC']

    start_run_signal = QtCore.pyqtSignal()
    new_data_signal = QtCore.pyqtSignal()
    plot_data_signal = QtCore.pyqtSignal(dict)

    run_timer = QtCore.QTimer(interval=1000)

    num_events = 0

    _is_running = False
    _trigger_channel = 'A'
    _coupling = 'AC'
    _is_trigger_enabled = False
    _is_upper_threshold_enabled = False
    _pulse_polarity = 'Positive'
    _polarity_sign = 1
    _is_baseline_correction_enabled = True
    _show_guides = False
    _show_marks = False
    _plot_options = PLOT_OPTIONS['lines']

    _range = 0
    _offset_level = 0.
    _offset = 0.
    _threshold = 0.
    _upper_threshold = 1.
    _timebase = 0
    _pre_trigger_window = 0.
    _post_trigger_window = 0.
    _pre_samples = 0
    _post_samples = 0
    _num_samples = 0

    _t_start_run = 0
    _run_time = 0
    _t_prev_run_time = 0

    _write_output = False
    _output_path = Path.home() / 'Documents'
    _run_number = 0
    _output_filename = None
    _output_file = None

    def __init__(self, use_fake=False, root_export_folder=None, root_max_mb=None):

        super().__init__()

        self._pulseheights = {'A': [], 'B': []}
        self._baselines = {'A': [], 'B': []}

        self._buf_cfg = None  # (num_samples, num_captures)

        if use_fake:
            self.scope = FakePicoScope()
        else:
            self.scope = PicoScope5000A()

        self.init_ui()
        
        # Persisted settings (loads after the UI shows)
        self._settings_store = attach_settings_store(self, org="YourLab", app="GammaSpectroscopy")

        # Root writers (open once after settings have loaded, so channel checkboxes are correct)
        self._root_folder = root_export_folder
        self._root_max_mb = root_max_mb
        self._writers = {}

        # Rolling CPS over the last few seconds
        self._rate_hist = deque()     # holds (timestamp, count_increment)
        self._rate_window_sec = 5.0   # “current” = last 5 seconds

        # Absolute time epoch (ns) for this run; set when Run starts
        self._run_epoch_ns = 0
        self._prev_block_ready_mono_ns = None  

        if self._root_folder:
        
            self.label_status.setText(f"ROOT export → {self._root_folder}")      
        
            def _open_writers():
                rid = next_run_id(self._root_folder)
                run_raw_dir = Path(self._root_folder) / f"run{rid:05d}" / "RAW"
                run_raw_dir.mkdir(parents=True, exist_ok=True)

                # Save run-level info for settings.txt
                self._run_id = rid
                self._run_dir = run_raw_dir.parent  # path to .../runNNNNN
                self._settings_written = False

                if self.ch_A_enabled_box.isChecked():
                    self._writers[0] = RootWriter(run_raw_dir, "A", rid, self._root_max_mb)
                if self.ch_B_enabled_box.isChecked():
                    self._writers[1] = RootWriter(run_raw_dir, "B", rid, self._root_max_mb)

            QtCore.QTimer.singleShot(0, _open_writers)

    def closeEvent(self, event):
        self._is_running = False
        self.scope.stop()
        for w in self._writers.values():
            try:
                w.close()
            except:
                pass

    def init_ui(self):
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOption('antialias', True)

        ui_path = resource_filename('gamma_spectroscopy',
                                    'gamma_spectroscopy_gui.ui')
        layout = uic.loadUi(ui_path, self)

        # Menubar
        menubar = QtWidgets.QMenuBar()

        export_spectrum_action = QtWidgets.QAction('&Export spectrum', self)
        export_spectrum_action.setShortcut('Ctrl+S')
        export_spectrum_action.triggered.connect(self.export_spectrum_dialog)

        write_output_action = QtWidgets.QAction('&Write output files', self)
        write_output_action.setShortcut('Ctrl+O')
        write_output_action.triggered.connect(self.write_output_dialog)

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(export_spectrum_action)
        file_menu.addAction(write_output_action)

        layout.setMenuBar(menubar)

        statusbar = QtWidgets.QStatusBar()
        self.label_status = QtWidgets.QLabel("")
        statusbar.addWidget(self.label_status)

        layout.setStatusBar(statusbar)

        self.start_run_signal.connect(self.start_scope_run)
        self.start_run_signal.connect(self._update_run_label)

        self.new_data_signal.connect(self.fetch_data)
        self.callback = create_callback(self.new_data_signal, self.scope)

        self.plot_data_signal.connect(self.plot_data)

        self.range_box.addItems(INPUT_RANGES.values())
        self.range_box.currentIndexChanged.connect(self.set_range)
        self.range_box.setCurrentIndex(5)
        self.polarity_box.addItems(self.POLARITY)
        self.polarity_box.currentIndexChanged.connect(self.set_polarity)
        self._pulse_polarity = self.POLARITY[0]
        self.coupling_box.addItems(self.COUPLING)
        self.coupling_box.currentIndexChanged.connect(self.set_coupling)
        self.offset_box.valueChanged.connect(self.set_offset)
        self.threshold_box.valueChanged.connect(self.set_threshold)
        self.upper_threshold_box.valueChanged.connect(self.set_upper_threshold)
        self.trigger_box.stateChanged.connect(self.set_trigger_state)
        self.upper_trigger_box.stateChanged.connect(
            self.set_upper_trigger_state)
        self.trigger_channel_box.currentTextChanged.connect(self.set_trigger)
        self.timebase_box.valueChanged.connect(self.set_timebase)
        self.pre_trigger_box.valueChanged.connect(self.set_pre_trigger_window)
        self.post_trigger_box.valueChanged.connect(
            self.set_post_trigger_window)
        self.baseline_correction_box.stateChanged.connect(
            self.set_baseline_correction_state)

        self.lld_box.valueChanged.connect(self.update_spectrum_plot)
        self.uld_box.valueChanged.connect(self.update_spectrum_plot)
        self.num_bins_box.valueChanged.connect(self.update_spectrum_plot)

        self.clear_run_button.clicked.connect(self.clear_run)
        self.single_button.clicked.connect(self.start_scope_run)
        self.run_stop_button.clicked.connect(self.toggle_run_stop)

        self.reset_event_axes_button.clicked.connect(self.reset_event_axes)
        self.reset_spectrum_axes_button.clicked.connect(
            self.reset_spectrum_axes)
        self.toggle_guides_button1.clicked.connect(self.toggle_guides)
        self.toggle_guides_button2.clicked.connect(self.toggle_guides)
        self.toggle_markslines_button1.clicked.connect(
            self.toggle_show_marks_or_lines)
        self.toggle_markslines_button2.clicked.connect(
            self.toggle_show_marks_or_lines)

        self.ch_A_enabled_box.stateChanged.connect(self.set_channel)
        self.ch_B_enabled_box.stateChanged.connect(self.set_channel)

        self.run_timer.timeout.connect(self._update_run_time_label)

        # --- settings snapshot on changes (debounced) ---
        self._info_snapshot_timer = QtCore.QTimer(self)
        self._info_snapshot_timer.setSingleShot(True)
        self._info_snapshot_timer.setInterval(1000)  # 1 s debounce
        self._info_snapshot_timer.timeout.connect(self._snapshot_info)

        def _hook(w, sig):
            getattr(w, sig).connect(self._snapshot_info_throttled)

        for w, sig in [
            (self.range_box, 'currentIndexChanged'),
            (self.offset_box, 'valueChanged'),
            (self.threshold_box, 'valueChanged'),
            (self.upper_threshold_box, 'valueChanged'),
            (self.trigger_box, 'stateChanged'),
            (self.upper_trigger_box, 'stateChanged'),
            (self.trigger_channel_box, 'currentTextChanged'),
            (self.timebase_box, 'valueChanged'),
            (self.pre_trigger_box, 'valueChanged'),
            (self.post_trigger_box, 'valueChanged'),
            (self.baseline_correction_box, 'stateChanged'),
            (self.polarity_box, 'currentIndexChanged'),
            (self.coupling_box, 'currentIndexChanged'),
            (self.ch_A_enabled_box, 'stateChanged'),
            (self.ch_B_enabled_box, 'stateChanged'),
        ]:
            _hook(w, sig)

        self.init_event_plot()
        self.init_spectrum_plot()

        self._emit_value_changed_signal(self.offset_box)
        self._emit_value_changed_signal(self.threshold_box)
        self._emit_value_changed_signal(self.timebase_box)
        self._emit_value_changed_signal(self.pre_trigger_box)
        self._emit_value_changed_signal(self.post_trigger_box)

        self.show()

    def _emit_value_changed_signal(self, widget):
        widget.valueChanged.emit(widget.value())

    def _current_cps(self) -> float:
        """Return rolling counts/sec over the last self._rate_window_sec."""
        now = time.time()
        # Drop old samples
        while self._rate_hist and (now - self._rate_hist[0][0] > self._rate_window_sec):
            self._rate_hist.popleft()
        if not self._rate_hist:
            return 0.0
        # Effective window is from oldest kept sample to now (≤ _rate_window_sec)
        effective_window = max(1e-9, now - self._rate_hist[0][0])
        total_counts = sum(n for _, n in self._rate_hist)
        return total_counts / effective_window

    def _snapshot_info_throttled(self):
        """Debounce settings snapshots so sliders don't spam files."""
        if not self._is_running or not getattr(self, "_run_dir", None):
            return
        self._info_snapshot_timer.stop()
        self._info_snapshot_timer.start()

    def _snapshot_info(self):
        """Write a settings/info snapshot into the current run folder."""
        if not self._is_running or not getattr(self, "_run_dir", None):
            return
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")  # human-readable, filename-safe
        self.write_info_file(info_filename=Path(self._run_dir) / f"info{ts}.txt")

    @QtCore.pyqtSlot()
    def toggle_run_stop(self):
        if not self._is_running:
            self.start_run()
        else:
            self.stop_run()

    def start_run(self):
        if not self.is_run_time_completed():
            self._is_running = True
            self._t_start_run = time.time()

            self._rate_hist.clear()

            # Absolute epoch for per-event timestamps (ns)
            self._run_epoch_wall_ns = time.time_ns()
            self._run_epoch_mono_ns = time.monotonic_ns()

            # Reset per-run timestamp helpers
            self._prev_block_ready_mono_ns = None
            setattr(self, "_ts_monotonic_offset_ns", 0)
            setattr(self, "_ts_last_ns", None)

            # Create settings.txt once per run folder
            self._maybe_write_run_settings()
            self.write_info_file(info_filename=Path(self._run_dir) / "info.txt")

            self._run_time = 0
            self._update_run_time_label()
            self.run_timer.start()
            self.start_run_signal.emit()
            self.run_stop_button.setText("Stop")
            self.single_button.setDisabled(True)
            self._update_run_label()
            if self._write_output:
                self.open_output_file()
                writer = csv.writer(self._output_file)
                writer.writerow(('time_A','pulse_height_A',
                                 'time_B','pulse_height_B'))

    def _maybe_write_run_settings(self):
        """Create runNNNNN/settings.txt with raw ns and human-readable time."""
        try:
            if not getattr(self, "_run_dir", None):
                return
            if getattr(self, "_settings_written", False):
                return
            ns = int(self._run_epoch_wall_ns)
            human = datetime.fromtimestamp(ns / 1e9).isoformat(sep=" ", timespec="microseconds")
            path = Path(self._run_dir) / "settings.txt"
            # Write only if it doesn't exist (keeps the first epoch recorded)
            if not path.exists():
                path.write_text(f"{ns}\n{human}\n", encoding="utf-8")
            self._settings_written = True
        except Exception:
            # Non-fatal: don't interrupt acquisition if disk write fails
            pass

    def stop_run(self):
        self._is_running = False
        self._update_run_time_label()
        self.scope.stop()
        self.run_timer.stop()
        self._run_time = time.time() - self._t_start_run
        self._t_prev_run_time += self._run_time
        self.run_stop_button.setText("Run")
        self.single_button.setDisabled(False)     
        
        if self._write_output:
            self.write_info_file()
            self.close_output_file()

    @QtCore.pyqtSlot()
    def start_scope_run(self):
        num_captures = self.num_captures_box.value()
        enA = self.ch_A_enabled_box.isChecked()
        enB = self.ch_B_enabled_box.isChecked()
        cfg = (self._num_samples, num_captures, enA, enB)
        if self._buf_cfg != cfg:
            self.scope.set_up_buffers(self._num_samples, num_captures)
            self._buf_cfg = cfg
        self.scope.start_run(self._pre_samples, self._post_samples,
                            self._timebase, num_captures, callback=self.callback)

    @QtCore.pyqtSlot(int)
    def set_range(self, range_idx):
        ranges = list(INPUT_RANGES.keys())
        self._range = ranges[range_idx]
        self.set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_offset(self, offset_level):
        self._offset_level = offset_level
        self.set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_threshold(self, threshold):
        self._threshold = threshold
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_upper_threshold(self, threshold):
        self._upper_threshold = threshold
        self.scope.stop()

    @QtCore.pyqtSlot(int)
    def set_trigger_state(self, state):
        self._is_trigger_enabled = state
        self.set_trigger()

    @QtCore.pyqtSlot(int)
    def set_upper_trigger_state(self, state):
        if not self._trigger_channel == 'A OR B':
            self._is_upper_threshold_enabled = state
            self.scope.stop()

    @QtCore.pyqtSlot(int)
    def set_polarity(self, idx):
        self._pulse_polarity = self.POLARITY[idx]
        self._polarity_sign = self.POLARITY_SIGN[idx]
        self.set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(int)
    def set_coupling(self, idx):
        self._coupling = self.COUPLING[idx]
        self.set_channel()

    @QtCore.pyqtSlot(int)
    def set_baseline_correction_state(self, state):
        self._is_baseline_correction_enabled = state

    def set_channel(self):
        self._offset = np.interp(self._offset_level, [-100, 100],
                                [-self._range, self._range])
        enA = self.ch_A_enabled_box.isChecked()
        enB = self.ch_B_enabled_box.isChecked()
        self.scope.set_channel('A', self._coupling, self._range,
                            self._polarity_sign * self._offset, is_enabled=enA)
        self.scope.set_channel('B', self._coupling, self._range,
                            self._polarity_sign * self._offset, is_enabled=enB)
        self.event_plot.setYRange(-self._range - self._offset,
                                self._range - self._offset)
        self._buf_cfg = None           # <— force re-setup next run
        self.scope.stop()

    def set_trigger(self):
        edge = 'RISING' if self._pulse_polarity == 'Positive' else 'FALLING'
        if self.trigger_channel_box.currentText() == 'A OR B':
            self._trigger_channel = 'A OR B'
            self.scope.set_trigger_A_OR_B(self._polarity_sign * self._threshold, edge,
                                        is_enabled=bool(self._is_trigger_enabled))
            self._is_upper_threshold_enabled = False
            self.upper_trigger_box.setChecked(False)
            self.upper_trigger_box.setCheckable(False)
        else:
            # get last letter of trigger channel box ('Channel A' -> 'A')
            channel = self.trigger_channel_box.currentText()[-1]
            self._trigger_channel = channel
            self.scope.set_trigger(channel,
                                   self._polarity_sign * self._threshold,
                                   edge, is_enabled=self._is_trigger_enabled)
            self.upper_trigger_box.setCheckable(True)
        if self._show_guides:
            self.draw_spectrum_plot_guides()
        self.scope.stop()

    @QtCore.pyqtSlot(int)
    def set_timebase(self, timebase):
        self._timebase = timebase
        dt = self.scope.get_interval_from_timebase(timebase)
        self.sampling_time_label.setText(f"{dt / 1e3:.3f} μs")
        self._update_num_samples()

    @QtCore.pyqtSlot(float)
    def set_pre_trigger_window(self, pre_trigger_window):
        self._pre_trigger_window = pre_trigger_window * 1e3
        self._update_num_samples()

    @QtCore.pyqtSlot(float)
    def set_post_trigger_window(self, post_trigger_window):
        self._post_trigger_window = post_trigger_window * 1e3
        self._update_num_samples()

    def _update_num_samples(self):
        pre_samples, post_samples = self._calculate_num_samples()
        num_samples = pre_samples + post_samples
        self.num_samples_label.setText(str(num_samples))

        self._pre_samples = pre_samples
        self._post_samples = post_samples
        self._num_samples = num_samples

        self.scope.stop()

        self._buf_cfg = None  # force re-setup on next start_scope_run

    def _calculate_num_samples(self):
        dt = self.scope.get_interval_from_timebase(self._timebase)
        pre_samples = floor(self._pre_trigger_window / dt)
        post_samples = floor(self._post_trigger_window / dt) + 1
        return pre_samples, post_samples

    def _update_run_time_label(self):
        run_time = round(self._t_prev_run_time
                         + time.time() - self._t_start_run)
        self.run_time_label.setText(f"{run_time} s")

        cps = self._current_cps()
        self.num_events_label.setText(f"({self.num_events} events, ~{cps:.1f} cps)")

        # Force repaint for fast response on user input
        self.run_time_label.repaint()
        self.num_events_label.repaint()

    def _update_run_label(self):
        self.run_number_label.setText(f"{self._run_number}")
        self.run_number_label.repaint()

    def _update_status_bar(self):
        if self._write_output:
            status_message = f'Output directory: {str(self._output_path)}'
        else:
            status_message = ''
        self.label_status.setText(status_message)

    @QtCore.pyqtSlot()
    def toggle_guides(self):
        self._show_guides = not self._show_guides

    @QtCore.pyqtSlot()
    def toggle_show_marks_or_lines(self):
        self._show_marks = not self._show_marks
        if self._show_marks:
            self._plot_options = PLOT_OPTIONS['marks']
        else:
            self._plot_options = PLOT_OPTIONS['lines']

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, ab = self.scope.get_data()
        if ab is None:
            return
        A, B = ab

        trig_offsets_ns = self.scope.get_last_trigger_offsets_ns()
        block_ready_mono_ns = self.scope.get_last_block_ready_mono_ns()

        # --- NEW: take and advance the anchor here (prevents races with plot_data)
        prev_block_ready_mono_ns = getattr(self, "_prev_block_ready_mono_ns", None)
        self._prev_block_ready_mono_ns = block_ready_mono_ns

        if (A is not None) or (B is not None):
            nA = 0 if A is None else len(A)
            nB = 0 if B is None else len(B)
            self.num_events += max(nA, nB)

            now = time.time()
            self._rate_hist.append((now, max(nA, nB)))

            self.plot_data_signal.emit({
                'x': t, 'A': A, 'B': B,
                'trig_offsets_ns': trig_offsets_ns,
                'block_ready_mono_ns': block_ready_mono_ns,
                'prev_block_ready_mono_ns': prev_block_ready_mono_ns,  # NEW
            })

        if self._is_running:
            if self.is_run_time_completed():
                self.stop_run()
            else:
                self.start_run_signal.emit()

    def is_run_time_completed(self):
        run_time = self._t_prev_run_time
        if self._is_running:
            run_time += time.time() - self._t_start_run
        return run_time >= self.run_duration_box.value()

    @QtCore.pyqtSlot()
    def clear_run(self):
        self._t_prev_run_time = 0
        self._t_start_run = time.time()
        self.num_events = 0
        self._rate_hist.clear()
        self._pulseheights = {'A': [], 'B': []}
        self._baselines = {'A': [], 'B': []}
        self._update_run_time_label()
        self.init_spectrum_plot()

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        x, A, B = data['x'], data['A'], data['B']
        trig_offsets_ns = data.get('trig_offsets_ns')
        block_ready_mono_ns = data.get('block_ready_mono_ns')
        prev_block_ready_mono_ns = data.get('prev_block_ready_mono_ns')

        # Keep an unfiltered copy of trigger offsets for the whole block
        trig_offsets_full = None if trig_offsets_ns is None else np.asarray(trig_offsets_ns, dtype=np.int64)

        # Coerce to empty arrays when channel is disabled
        nsamp = self._num_samples
        A = A if (A is not None) else np.empty((0, nsamp), dtype=float)
        B = B if (B is not None) else np.empty((0, nsamp), dtype=float)

        # --- Per-capture features
        def feats(arr):
            if arr.size == 0:
                return np.empty(0), np.empty(0), np.empty(0)
            arr = arr * self._polarity_sign
            n_bl = int(self._pre_samples * 0.8)
            bl = arr[:, :n_bl].mean(axis=1) if (self._is_baseline_correction_enabled and n_bl > 0) else np.zeros(arr.shape[0])
            ph = (arr.max(axis=1) - bl) * 1e3
            ts = x[np.argmax(arr, axis=1)]
            return ts, bl, ph

        tsA, blA, phA = feats(A)
        tsB, blB, phB = feats(B)

        # Optional CSV 
        if self._write_output and not self._output_file.closed:
            writer = csv.writer(self._output_file)
            n = max(len(tsA), len(tsB))
            for i in range(n):
                a = (tsA[i], phA[i]) if i < len(tsA) else ("", "")
                b = (tsB[i], phB[i]) if i < len(tsB) else ("", "")
                writer.writerow((a[0], a[1], b[0], b[1]))

        # ULD cut (only when a specific channel is selected)
        cond = None
        if self._is_upper_threshold_enabled and self._trigger_channel in ('A', 'B'):
            use_A = (self._trigger_channel == 'A')
            ph_sel = phA if use_A else phB
            if ph_sel.size:
                cond = (ph_sel <= self._upper_threshold * 1e3)

                if A.size: A = A[cond]
                if B.size: B = B[cond]

                if tsA.size: tsA, blA, phA = tsA[cond], blA[cond], phA[cond]
                if tsB.size: tsB, blB, phB = tsB[cond], blB[cond], phB[cond]

                # NOTE: do NOT filter trig_offsets_full here; we slice absolute times later

        # --- ROOT export (run-relative timestamps)
        if self._writers and (A.size or B.size):
            fs = self._range
            if fs:
                scale = 32767.0 / fs
                dt_ns = self.scope.get_interval_from_timebase(self._timebase)

                # reconstruct absolute monotonic trigger times for *this* block
                # Rapid-block trigger offsets are segment-local; use host time to spread.
                abs_trig_mono = None
                if block_ready_mono_ns is not None:
                    dt_ns = self.scope.get_interval_from_timebase(self._timebase)
                    post_ns = int(round(self._post_samples * dt_ns))
                    t_last = int(block_ready_mono_ns) - post_ns  # trigger time of LAST segment in this block

                    # how many captures are we actually emitting from this block (after ULD cut)?
                    ncap_full = max(A.shape[0], B.shape[0])

                    if ncap_full:
                        if self.num_captures_box.value() > 1:
                            # Spread captures between previous block's last trigger and this block's last trigger
                            prev_ready = prev_block_ready_mono_ns
                            # replace the old linspace(...) in plot_data() where we build times_full
                            if prev_ready is None:
                                # First block of the run: no anchor → just pack them just before t_last
                                # (keeps order and uniqueness)
                                times_full = np.arange(t_last - (ncap_full - 1), t_last + 1, dtype=np.int64)
                            else:
                                start = int(prev_ready) - post_ns      # = t_prev_last
                                end   = t_last
                                if end <= start:
                                    # extremely rare: jitter or zero gap → create a 1-ns strictly increasing ladder
                                    start = end - ncap_full
                                # half-open spacing: (start, end]
                                times_full = np.linspace(start, end, ncap_full + 1, dtype=np.int64)[1:]
                        else:
                            # Single capture per block → true time for that single capture
                            times_full = np.full(ncap_full, t_last, dtype=np.int64)

                        # Apply selection mask (ULD cut) if present
                        abs_trig_mono = times_full if cond is None else times_full[cond]

                    # remember for the next block
                    #self._prev_block_ready_mono_ns = int(block_ready_mono_ns)

                # STITCH: keep timestamps continuous across blocks / settings changes
                def _stitch_run_relative(abs_ns_arr):
                    if abs_ns_arr is None:
                        return None
                    rr = np.asarray(abs_ns_arr, dtype=np.int64) - np.int64(self._run_epoch_mono_ns)

                    off  = getattr(self, "_ts_monotonic_offset_ns", 0)
                    last = getattr(self, "_ts_last_ns", None)

                    if last is not None and rr.size and (rr[0] + off) < last:
                        off += (last - (rr[0] + off))

                    rr = rr + off

                    # NEW: enforce strictly increasing inside this batch
                    for i in range(1, rr.size):
                        if rr[i] <= rr[i - 1]:
                            rr[i] = rr[i - 1] + 1  # bump by 1 ns

                    setattr(self, "_ts_monotonic_offset_ns", int(off))
                    if rr.size:
                        setattr(self, "_ts_last_ns", int(rr[-1]))
                    return rr

                def emit(arr, ph, ts_peak_sec, ch_idx, trig_abs_ns):
                    w = self._writers.get(ch_idx)
                    if w is None or arr.size == 0:
                        return
                    n = min(arr.shape[0], len(ph), len(ts_peak_sec))
                    if n <= 0 or trig_abs_ns is None:
                        return
                    run_rel_ns = _stitch_run_relative(trig_abs_ns)
                    if run_rel_ns is None:
                        return
                    samples = np.clip(np.rint((arr[:n] / self._polarity_sign) * scale), -32768, 32767).astype(np.int16)
                    t_ns = np.asarray(run_rel_ns[:n], dtype=np.int64)
                    t_ns = np.maximum(t_ns, 0).astype(np.uint64)
                    e_u16 = np.clip(np.rint(ph[:n]), 0, 65535).astype(np.uint16)
                    w.add_many(
                        samples_i16=samples[:n],
                        ts_ns=t_ns[:n],
                        energy_u16=e_u16[:n],
                    )

                if abs_trig_mono is not None:
                    emit(A, phA, tsA, 0, abs_trig_mono)  # A writer may be None → harmless
                    emit(B, phB, tsB, 1, abs_trig_mono)

        # --- Accumulate spectrum and refresh UI
        self._baselines['A'].extend(blA)
        self._baselines['B'].extend(blB)
        self._pulseheights['A'].extend(phA)
        self._pulseheights['B'].extend(phB)

        # Plot the last available capture(s)
        if A.shape[0] or B.shape[0]:
            lastA = A[-1] if A.shape[0] else np.zeros(nsamp)
            lastB = B[-1] if B.shape[0] else np.zeros(nsamp)
            ph_last = np.array([
                phA[-1] if phA.size else 0.0,
                phB[-1] if phB.size else 0.0,
            ])
            bl_last = np.array([
                blA[-1] if blA.size else 0.0,
                blB[-1] if blB.size else 0.0,
            ])
            self.update_event_plot(x, lastA, lastB, ph_last, bl_last)
            self.update_spectrum_plot()


    def init_event_plot(self):
        self.event_plot.clear()
        self.event_plot.setLabels(title='Scintillator event',
                                  bottom='Time [us]', left='Signal [V]')

    @QtCore.pyqtSlot()
    def reset_event_axes(self):
        self.event_plot.enableAutoRange(axis=pg.ViewBox.XAxis)
        self.event_plot.setYRange(-self._range - self._offset,
                                  self._range - self._offset)

    def update_event_plot(self, x, A, B, pulseheights, baselines):
        self.event_plot.clear()
        if self.ch_A_enabled_box.isChecked():
            self.event_plot.plot(x * 1e6, A * self._polarity_sign, **self._plot_options['A'])
        if self.ch_B_enabled_box.isChecked():
            self.event_plot.plot(x * 1e6, B * self._polarity_sign, **self._plot_options['B'])

        if self._show_guides:
            self.draw_event_plot_guides(x, baselines, pulseheights)

    def draw_event_plot_guides(self, x, baselines, pulseheights):
        phA, phB = pulseheights
        blA, blB = baselines
        plot = self.event_plot

        # mark baselines and pulseheights
        if self.ch_A_enabled_box.isChecked():
            self.draw_guide(plot, blA, 'blue')
            self.draw_guide(plot, phA / 1e3, 'purple')
        if self.ch_B_enabled_box.isChecked():
            self.draw_guide(plot, blB, 'blue')
            self.draw_guide(plot, phB / 1e3, 'purple')

        # mark trigger instant
        try:
            # right after updating settings, pre_samples may exceed old event
            self.draw_guide(plot, x[self._pre_samples] * 1e6, 'green',
                            'vertical')
        except IndexError:
            pass

        # mark trigger thresholds
        if self._is_trigger_enabled:
            self.draw_guide(plot, self._threshold, 'green')
        if self._is_upper_threshold_enabled \
           and not self._trigger_channel == 'A OR B':
            self.draw_guide(plot, self._upper_threshold, 'green')

    def draw_guide(self, plot, pos, color, orientation='horizontal', width=2.):
        if orientation == 'vertical':
            angle = 90
        else:
            angle = 0
        color = GUIDE_COLORS[color]
        plot.addItem(pg.InfiniteLine(
            pos=pos, angle=angle,
            pen={'color': color, 'width': width}))

    def init_spectrum_plot(self):
        self.spectrum_plot.clear()
        self.spectrum_plot.setLabels(title='Spectrum',
                                     bottom='Pulseheight [mV]', left='Counts')

    @QtCore.pyqtSlot()
    def reset_spectrum_axes(self):
        self.spectrum_plot.enableAutoRange()

    def update_spectrum_plot(self):
        if len(self._baselines['A']) > 0 or len(self._baselines['B']) > 0:
            self.spectrum_plot.clear()
            x, bins, channel_counts = self.make_spectrum()
            for counts, channel in zip(channel_counts, ['A', 'B']):
                if counts is not None:
                    self.spectrum_plot.plot(x, counts, **self._plot_options[channel])
            if self._show_guides:
                self.draw_spectrum_plot_guides()

    def make_spectrum(self):
        #xrange = 2 * self._range * 1e3
        xrange = (self._range - self._offset) * 1e3
        xmin = .01 * self.lld_box.value() * xrange
        xmax = .01 * self.uld_box.value() * xrange
        if xmax < xmin:
            xmax = xmin
        bins = np.linspace(xmin, xmax, self.num_bins_box.value())
        x = (bins[:-1] + bins[1:]) / 2
        channel_counts = []

        for channel in 'A', 'B':
            box = getattr(self, f'ch_{channel}_enabled_box')
            if box.isChecked():
                n, _ = np.histogram(self._pulseheights[channel], bins=bins)
                channel_counts.append(n)
            else:
                channel_counts.append(None)

        return x, bins, channel_counts

    def draw_spectrum_plot_guides(self):
        plot = self.spectrum_plot

        if self._is_trigger_enabled:
            self.draw_guide(plot, self._threshold * 1e3, 'green', 'vertical')

        clip_level = (self._range - self._offset)
        self.draw_guide(plot, clip_level * 1e3, 'red', 'vertical')

        if self._is_upper_threshold_enabled and self._trigger_channel != 'A OR B':
            self.draw_guide(plot, self._upper_threshold * 1e3, 'green', 'vertical')

        if not self._is_baseline_correction_enabled:
            return

        if self.ch_A_enabled_box.isChecked() and len(self._baselines['A']) > 0:
            min_blA = np.percentile(self._baselines['A'], 5)
            self.draw_guide(plot, (self._threshold - min_blA) * 1e3, 'purple', 'vertical')

        if self.ch_B_enabled_box.isChecked() and len(self._baselines['B']) > 0:
            min_blB = np.percentile(self._baselines['B'], 5)
            self.draw_guide(plot, (self._threshold - min_blB) * 1e3, 'purple', 'vertical')

    def export_spectrum_dialog(self):
        """Dialog for exporting a data file."""

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Save spectrum", directory="spectrum.csv")
        if not file_path:
            # Cancel was pressed, no file was selected
            return

        x, _, channel_counts = self.make_spectrum()
        channel_counts = [u if u is not None else [0] * len(x) for
                          u in channel_counts]

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('pulseheight', 'counts_ch_A', 'counts_ch_B'))
            for row in zip(x, *channel_counts):
                writer.writerow(row)

    def write_output_dialog(self):
        self._write_output = True
        file_path = QtWidgets.QFileDialog.getExistingDirectory(self,
            caption="Choose directory for output files",
            directory=str(self._output_path.absolute()))
        self._output_path = Path(file_path)
        self._update_status_bar()
        #print('Output directory: {}'.format(self._output_path))

    def open_output_file(self):
        for x in Path(self._output_path).glob('*.csv'):
            if x.is_file() and x.name[0:3] == 'Run':
                self._run_number = int(x.name[3:7]) + 1

        self._output_filename =  self._output_path / 'Run{0:04d}.csv'\
                                                       .format(self._run_number)

        try:
             self._output_file = open(self._output_filename, 'w',
                                      newline='')
             return 1
        except IOError:
            print('Error: Unable to open: {}'\
                  .format(self._output_filename))
            return 0

    def close_output_file(self):
        try:
            self._output_file.close()
            return 1
        except IOError:
            print('Error: Unable to close: {}'\
                  .format(self._output_filename))
            return 0

    def write_info_file(self, info_filename: Path | None = None):
        if info_filename is None:
            # Original behavior (CSV companion .info)
            info_filename = self._output_filename.with_suffix('.info')

        try:
            info_file = open(info_filename, 'w', newline='', encoding="utf-8")
        except IOError:
            print(f'Error: Unable to open: {info_filename}\n')
            return

        info_file.write(f'Start time: {time.ctime(self._t_start_run)}\n')
        info_file.write(f'Run time: {self._run_time:.1f} s\n')
        info_file.write(f'Coupling: {self._coupling}\n')
        info_file.write(f'Baseline correction: {self._is_baseline_correction_enabled}\n')
        if self._is_trigger_enabled:
            info_file.write(f'Trigger channel: {self._trigger_channel}\n')
            info_file.write(f'Threshold: {self._threshold:.3f} V\n')
        else:
            info_file.write('Untriggered\n')
        info_file.write('Pre-trigger window: {0:.2f} μs\n'\
                        .format(self._pre_trigger_window/1e3))
        info_file.write('Post-trigger window {0:.2f} μs\n'\
                        .format(self._post_trigger_window/1e3))
        info_file.write(f'Samples per capture: {self._num_samples}\n')
        info_file.write(f'Captures per block: {self.num_captures_box.value()}\n')
        info_file.close()

def main():
    global qtapp

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--root-export-folder", type=str, default=None,
                        help="Folder to write ROOT files (one per active channel).")
    parser.add_argument("--root-max-mb", type=int, default=None,
                        help="Split ROOT when file size (MB) is reached.")

    parser.add_argument('--fake', action='store_true',
                        help="Use fake hardware")
    args = parser.parse_args()

    qtapp = QtWidgets.QApplication(sys.argv)

    ui = UserInterface(
        use_fake=args.fake,
        root_export_folder=args.root_export_folder,
        root_max_mb=args.root_max_mb,
    )    
    sys.exit(qtapp.exec_())


if __name__ == '__main__':
    main()
