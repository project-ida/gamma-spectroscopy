# gamma_spectroscopy/settings_persistence.py
from __future__ import annotations
from PyQt5 import QtCore

class _SettingsStore:
    def __init__(self, ui, org="YourLab", app="GammaSpectroscopy"):
        self.ui = ui
        self.s = QtCore.QSettings(org, app)
        # Keep original closeEvent so we can chain to it
        self._orig_close_event = getattr(ui, "closeEvent", None)

        # Auto-load after the UI finishes showing, auto-save on close
        QtCore.QTimer.singleShot(0, self.load)
        ui.closeEvent = self._closeEvent_with_save  # monkey-patch minimal change

    # --------- public ----------
    def load(self):
        ui, s = self.ui, self.s

        # Helpers to set values without spamming signals/hardware
        def set_combo_index(combo, idx):
            if idx is None: return
            with QtCore.QSignalBlocker(combo):
                idx = int(idx)
                if 0 <= idx < combo.count():
                    combo.setCurrentIndex(idx)

        def set_combo_text(combo, text):
            if not text: return
            with QtCore.QSignalBlocker(combo):
                ix = combo.findText(text)
                if ix >= 0:
                    combo.setCurrentIndex(ix)

        def set_check(box, val):
            if val is None: return
            with QtCore.QSignalBlocker(box):
                box.setChecked(bool(val))

        def set_spin(spin, val):
            if val is None: return
            with QtCore.QSignalBlocker(spin):
                spin.setValue(type(spin.value())(val))

        # Combos
        set_combo_index(ui.range_box,       s.value("range_idx", type=int))
        set_combo_index(ui.polarity_box,    s.value("polarity_idx", type=int))
        set_combo_index(ui.coupling_box,    s.value("coupling_idx", type=int))
        set_combo_text (ui.trigger_channel_box, s.value("trigger_channel_text", type=str))

        # Checks
        set_check(ui.trigger_box,              s.value("trigger_enabled", type=bool))
        set_check(ui.upper_trigger_box,        s.value("upper_trigger_enabled", type=bool))
        set_check(ui.baseline_correction_box,  s.value("baseline_correction", type=bool))
        set_check(ui.ch_A_enabled_box,         s.value("ch_A_enabled", type=bool))
        set_check(ui.ch_B_enabled_box,         s.value("ch_B_enabled", type=bool))

        # Spins / doubles
        set_spin(ui.offset_box,          s.value("offset_level",   type=float))
        set_spin(ui.threshold_box,       s.value("threshold",      type=float))
        set_spin(ui.upper_threshold_box, s.value("upper_threshold",type=float))
        set_spin(ui.timebase_box,        s.value("timebase",       type=int))
        set_spin(ui.pre_trigger_box,     s.value("pre_trigger",    type=float))
        set_spin(ui.post_trigger_box,    s.value("post_trigger",   type=float))
        set_spin(ui.lld_box,             s.value("lld",            type=float))
        set_spin(ui.uld_box,             s.value("uld",            type=float))
        set_spin(ui.num_bins_box,        s.value("num_bins",       type=int))
        set_spin(ui.run_duration_box,    s.value("run_duration",   type=float))
        # NEW: number of captures
        if hasattr(ui, "num_captures_box"):
            set_spin(ui.num_captures_box, s.value("num_captures", type=int))

        # Re-apply to hardware/state once (no signals fired during load)

        # 1) Channel config first
        ui.set_range(ui.range_box.currentIndex())
        ui.set_coupling(ui.coupling_box.currentIndex())
        ui.set_polarity(ui.polarity_box.currentIndex())

        # 2) Timing / sampling
        ui.set_timebase(ui.timebase_box.value())
        ui.set_pre_trigger_window(ui.pre_trigger_box.value())
        ui.set_post_trigger_window(ui.post_trigger_box.value())

        # 3) Analog levels
        ui.set_offset(ui.offset_box.value())          # updates _offset and calls set_channel()
        ui.set_threshold(ui.threshold_box.value())
        ui.set_upper_threshold(ui.upper_threshold_box.value())
        ui.set_baseline_correction_state(int(ui.baseline_correction_box.isChecked()))

        # 4) Trigger enable flags (these slots set internal flags)
        ui.set_trigger_state(int(ui.trigger_box.isChecked()))
        ui.set_upper_trigger_state(int(ui.upper_trigger_box.isChecked()))

        # 5) Finalize trigger (uses channel + flags + thresholds)
        ui.set_trigger()

        # 6) View toggles (persist _show_guides and _show_marks)
        show_guides_saved = s.value("show_guides", type=bool)
        if show_guides_saved is not None and bool(show_guides_saved) != bool(getattr(ui, "_show_guides", False)):
            ui.toggle_guides()

        show_marks_saved = s.value("show_marks", type=bool)
        if show_marks_saved is not None and bool(show_marks_saved) != bool(getattr(ui, "_show_marks", False)):
            ui.toggle_show_marks_or_lines()

    def save(self):
        ui, s = self.ui, self.s
        # Combos
        s.setValue("range_idx",            ui.range_box.currentIndex())
        s.setValue("polarity_idx",         ui.polarity_box.currentIndex())
        s.setValue("coupling_idx",         ui.coupling_box.currentIndex())
        s.setValue("trigger_channel_text", ui.trigger_channel_box.currentText())
        # Checks
        s.setValue("trigger_enabled",       bool(ui.trigger_box.isChecked()))
        s.setValue("upper_trigger_enabled", bool(ui.upper_trigger_box.isChecked()))
        s.setValue("baseline_correction",   bool(ui.baseline_correction_box.isChecked()))
        s.setValue("ch_A_enabled",          bool(ui.ch_A_enabled_box.isChecked()))
        s.setValue("ch_B_enabled",          bool(ui.ch_B_enabled_box.isChecked()))
        # Spins / doubles
        s.setValue("offset_level",   float(ui.offset_box.value()))
        s.setValue("threshold",      float(ui.threshold_box.value()))
        s.setValue("upper_threshold",float(ui.upper_threshold_box.value()))
        s.setValue("timebase",       int(ui.timebase_box.value()))
        s.setValue("pre_trigger",    float(ui.pre_trigger_box.value()))
        s.setValue("post_trigger",   float(ui.post_trigger_box.value()))
        s.setValue("lld",            float(ui.lld_box.value()))
        s.setValue("uld",            float(ui.uld_box.value()))
        s.setValue("num_bins",       int(ui.num_bins_box.value()))
        s.setValue("run_duration",   float(ui.run_duration_box.value()))
        # NEW: number of captures
        if hasattr(ui, "num_captures_box"):
            s.setValue("num_captures", int(ui.num_captures_box.value()))
        # NEW: view toggles
        s.setValue("show_guides", bool(getattr(ui, "_show_guides", False)))
        s.setValue("show_marks",  bool(getattr(ui, "_show_marks",  False)))

    # --------- internal ----------
    def _closeEvent_with_save(self, evt):
        try:
            self.save()
        finally:
            if self._orig_close_event:
                self._orig_close_event(evt)

def attach_settings_store(ui, *, org="YourLab", app="GammaSpectroscopy"):
    """One-call attachment: schedules auto-load and auto-save-on-close."""
    return _SettingsStore(ui, org=org, app=app)

