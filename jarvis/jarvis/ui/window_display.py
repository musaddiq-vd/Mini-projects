"""Display control: themes, voice window commands, move/glide, mini mode and the desktop widget."""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QWidget

from jarvis.config import save_settings
from jarvis.ui import theme
from jarvis.ui.hud import app_icon
from jarvis.ui.mini import MiniWindow
from jarvis.ui.theme import DANGER, OK, WARN


class DisplayMixin:
    """How and where Jarvis is shown."""

    # -- voice control of this window ------------------------------------------------------
    def _shown_window(self):
        return self.mini if self.mini and self.mini.isVisible() else self

    def ui_command(self, action: str, theme_name: str = "", opts: dict | None = None) -> None:
        s = self.app.settings
        opts = opts or {}
        if action in ("widget_on", "widget_off"):
            s.desktop_widget = action == "widget_on"
            corner = opts.get("place", "")
            if corner in ("top_right", "top_left", "bottom_right", "bottom_left"):
                s.widget_corner = corner
            save_settings(s)
            self.apply_desktop_widget()
            return
        if action in ("minimize", "maximize", "move", "glide", "place", "bigger", "smaller", "stop"):
            self._window_motion(action, opts)
            return
        if action in ("night_mode", "day_mode", "auto_mode"):
            s.theme_mode = {"night_mode": "dark", "day_mode": "light", "auto_mode": "auto"}[action]
        elif action == "theme" and theme_name in theme.THEMES:
            s.theme_name = theme_name
            s.theme_color = theme.THEMES[theme_name][0]  # the theme's own accent
        elif action == "wallpaper_theme":
            from jarvis.ui.wallcolors import from_wallpaper

            colors = from_wallpaper()
            if colors is None:
                self.toast("🎨", "Couldn't read the wallpaper's colours.", WARN, 4)
                return
            s.theme_color, s.theme_second = colors
        elif action == "avatar" and opts.get("avatar"):
            from jarvis.ui.avatars import AVATARS

            if opts["avatar"] in AVATARS:
                s.avatar = opts["avatar"]
                s.orb_face = True
        elif action == "mini_mode":
            if not (self.mini and self.mini.isVisible()):
                self.toggle_mini()
            return
        elif action == "full_window":
            self._show_window()
            return
        elif action == "show_help":
            if not (getattr(self, "help", None) and self.help.isVisible()):
                self.toggle_help()
            return
        elif action == "show_achievements":
            if not (getattr(self, "badges", None) and self.badges.isVisible()):
                self.toggle_badges()
            return
        elif action == "dashboard":
            tab = theme_name if theme_name in ("week", "habits", "money", "goals") else None
            if not (getattr(self, "dashboard", None) and self.dashboard.isVisible()) or tab:
                self.toggle_dashboard(tab or "week")
            return
        elif action == "clear_chat":
            self.chat.clear()
            return
        else:
            return
        save_settings(s)
        self._apply_theme()
        self.orb.flash()

    def _window_motion(self, action: str, opts: dict) -> None:
        """Voice control of Jarvis' own window (or the mini orb when it's showing)."""
        if not hasattr(self, "mover"):
            from jarvis.ui.mover import WindowMover

            self.mover = WindowMover(self._shown_window)
        if action == "stop":
            self.mover.stop()
            return
        target = self._shown_window()
        if action == "minimize":
            self.mover.stop()
            if target is self:
                self.showMinimized()
            else:
                self.mini.hide()  # the orb goes away; Ctrl+Alt+J or the tray brings Jarvis back
            return
        if action == "maximize":
            self._show_window()
            self.showMaximized()
            return
        if target is self and self.isMinimized():
            self.showNormal()
        if action == "move":
            self.mover.nudge(opts.get("direction", ""), opts.get("amount") or "little")
        elif action == "glide":
            self.mover.glide(opts.get("direction", ""), opts.get("speed") or "normal")
        elif action == "place":
            self.mover.place(opts.get("place", ""))
        elif action in ("bigger", "smaller"):
            if self.mover.resize(action == "bigger") == "fixed":
                self.toast("🔍", "The mini orb has one size - say 'full window' for the big one.", self.accent, 3)
        if target is self.mini:
            from PyQt6.QtCore import QSettings

            QTimer.singleShot(600, lambda: QSettings("MusaddiqKhan", "Jarvis").setValue("mini_pos", self.mini.pos()))

    # -- day / night ------------------------------------------------------------------
    def cycle_theme_mode(self) -> None:
        s = self.app.settings
        s.theme_mode = {"dark": "light", "light": "auto"}.get(s.theme_mode, "dark")
        save_settings(s)
        self._apply_theme()
        label = {"dark": "Night mode", "light": "Day mode", "auto": "Auto: day by day, night by night"}[s.theme_mode]
        self.toast("🌓", label, self.accent, 2.5)

    def _check_auto_theme(self) -> None:
        """In auto mode, switch when the sun 'sets' or 'rises'; follow a new wallpaper's colours if asked."""
        if self.app.settings.theme_mode == "auto" and theme.resolve_mode("auto") != self._applied_mode:
            self._apply_theme()
        self._follow_wallpaper()

    _wallpaper_sig = ""

    def _follow_wallpaper(self, force: bool = False) -> bool:
        """Take the wallpaper's colours when it changed (only with 'follow my wallpaper' on)."""
        s = self.app.settings
        if not s.follow_wallpaper:
            return False
        from jarvis.ui import wallcolors

        sig = wallcolors.signature(wallcolors.wallpaper_path())
        if not sig or (sig == self._wallpaper_sig and not force):
            return False
        self._wallpaper_sig = sig
        colors = wallcolors.from_wallpaper()
        if colors is None or (s.theme_color, s.theme_second) == colors:
            return False
        s.theme_color, s.theme_second = colors
        save_settings(s)
        self._apply_theme()
        return True

    _applied_mode = "dark"

    # -- mini mode and desktop widget ----------------------------------------------
    def toggle_mini(self) -> None:
        if self.mini and self.mini.isVisible():
            self._show_window()
            return
        if self.mini is None:
            from jarvis.ui.mini import next_up

            self.mini = MiniWindow(self.accent, self.app.settings.orb_face, self._show_window, [
                ("Wake / Sleep", self._toggle_sleep), ("Mute / Unmute", self.mute_btn.toggle),
                ("Quit", self.close)], send=self._send, upcoming=lambda: next_up(self.app))
            for who, text in self._recent_for_mini():
                self.mini.add_message(who, text)
            self.mini.orb.set_state(self._last_state or "connecting")
            self.mini.orb.set_avatar(self.app.settings.avatar)
            self.orb.orbs.append(self.mini.orb)
        self.mini.orb._timer.start(16)
        self.hide()
        self.mini.show_at_saved_place()

    def apply_desktop_widget(self) -> None:
        """Show, move or hide the desktop widget to match the settings."""
        s = self.app.settings
        if not s.desktop_widget:
            if self.desk is not None:
                self.desk.hide_from_desktop()
            return
        if self.desk is None:
            from jarvis.ui.desktop_widget import DesktopWidget

            self.desk = DesktopWidget(self.app, self.accent, s.orb_face, s.widget_corner, s.widget_screen)
            self.desk.orb.set_state(self._last_state or "connecting")
            self.desk.orb.set_avatar(self.app.settings.avatar)
            self.orb.orbs.append(self.desk.orb)
        self.desk.set_accent(self.accent)
        self.desk.place(s.widget_corner, s.widget_screen)
        self.desk.show_on_desktop()

    desk = None

    def _recent_for_mini(self) -> list[tuple[str, str]]:
        history = self.app.history
        if not history:
            return []
        return [(i.get("role", ""), i.get("text", "")) for i in history.recent(6)
                if i.get("role") in ("user", "assistant")]

    def _leave_mini(self) -> None:
        if self.mini and self.mini.isVisible():
            self.mini.hide()
            self.mini.orb._timer.stop()  # no drawing while hidden

    def _show_window(self) -> None:
        self._leave_mini()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _apply_theme(self) -> None:
        s = self.app.settings
        self.accent = theme.accent_for(s.theme_name, s.theme_color)
        self._applied_mode = theme.resolve_mode(s.theme_mode)
        QApplication.instance().setStyleSheet(theme.apply(s.theme_name, s.theme_mode, self.accent, s.theme_second))
        if hasattr(self, "mode_btn"):
            self.mode_btn.setText({"dark": "🌙", "light": "☀", "auto": "🌓"}.get(s.theme_mode, "🌙"))
            self.mode_btn.setToolTip(f"Theme: {s.theme_mode} - click to switch (dark → light → auto)")
        self.orb.set_accent(self.accent)
        self.orb.set_face(self.app.settings.orb_face)
        self.orb.set_avatar(self.app.settings.avatar)
        if self.mini:
            self.mini.set_accent(self.accent)
        if self.desk is not None:
            self.desk.set_accent(self.accent)
        self.stats.accent = self.accent
        self.background.set_accent(self.accent)
        for glow in self._glows:
            glow.set_color(self.accent)
        name = self.app.settings.assistant_name
        self.title.setText(name.upper() + ("  ·  TUTOR" if self.practicing else ""))
        self.setWindowTitle(name)
        icon = app_icon(self.accent)
        self.setWindowIcon(icon)
        if self.tray:
            self.tray.setIcon(icon)
            self.tray.setToolTip(name)
        self._set_state_style(self._last_state or "connecting")

    def _set_state_style(self, state: str) -> None:
        color = {"thinking": WARN, "muted": DANGER, "sleeping": theme.MUTED, "connecting": theme.MUTED,
                 "offline": theme.MUTED, "speaking": OK}.get(state, theme.readable(self.accent))
        self.state_label.setStyleSheet(f"color:{color};")
        self.dot.color = QColor(color)

    def _update_clock(self) -> None:
        self.clock.setText(datetime.now().strftime("%A · %d %B · %I:%M:%S %p"))

    # -- privacy: you always see when Jarvis looks ----------------------------------------
    def flash_capture(self, what: str) -> None:
        """A camera-shutter flash over the window and a notice, every time the screen or camera is captured."""
        from PyQt6.QtWidgets import QGraphicsOpacityEffect

        flash = QWidget(self.centralWidget())
        flash.setStyleSheet("background: white;")
        flash.setGeometry(self.centralWidget().rect())
        flash.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        effect = QGraphicsOpacityEffect(flash)
        flash.setGraphicsEffect(effect)
        flash.show()
        flash.raise_()
        anim = QPropertyAnimation(effect, b"opacity", flash)
        anim.setDuration(420)
        anim.setStartValue(0.55)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(flash.deleteLater)
        anim.start()
        icon, text = ("📷", "Jarvis took a photo with your camera") if what == "camera" else \
            ("👁", "Jarvis looked at your screen")
        self.toast(icon, text, theme.ACCENT, 3)
        if self.tray and not self.isVisible():
            self.tray.showMessage(self.app.settings.assistant_name, text)
