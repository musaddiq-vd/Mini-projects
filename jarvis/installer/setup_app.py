"""Jarvis Setup: install, update and uninstall Jarvis for the current user (no admin rights needed).

Runs on the official, signed Python runtime that ships in the package, so Windows Smart App Control allows it.

    Install Jarvis.cmd                                  the setup window
    pythonw setup_app.py --silent [--dir D] [--no-desktop] [--no-menu] [--autostart]
    pythonw setup_app.py --uninstall [--silent] [--delete-data]

Installs to %LOCALAPPDATA%\\Programs\\Jarvis, adds Start Menu / desktop shortcuts, optional start with Windows,
and an entry in Settings > Apps. Updating keeps your settings and data: they live in %USERPROFILE%\\.jarvis,
never in the program folder.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

APP = "Jarvis"
PUBLISHER = "Musaddiq Khan"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
NO_WINDOW = 0x08000000
DETACHED = 0x00000008


def uninstall_key() -> str:
    # tests use their own key so they never touch a real install
    return os.environ.get("JARVIS_SETUP_REGKEY", r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Jarvis")


def run_value() -> str:
    return os.environ.get("JARVIS_SETUP_RUNVALUE", APP)


def package_root() -> Path:
    """The folder that holds runtime\\ and app\\ (the unzipped package, or an installed copy)."""
    return Path(__file__).resolve().parents[2]


def version() -> str:
    sys.path.insert(0, str(package_root() / "app"))
    try:
        from jarvis import __version__

        return __version__
    except Exception:
        return "?"


def default_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Programs" / APP


def start_menu() -> Path:
    test = os.environ.get("JARVIS_SETUP_SHORTCUTS")
    if test:
        return Path(test) / "start"
    return Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def desktop() -> Path:
    test = os.environ.get("JARVIS_SETUP_SHORTCUTS")
    if test:
        return Path(test) / "desktop"
    try:  # the real Desktop folder (it can be moved, e.g. into OneDrive)
        import ctypes

        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0:
            return Path(buf.value)
    except (AttributeError, OSError):
        pass
    return Path.home() / "Desktop"


def paths(target: Path) -> dict[str, Path]:
    return {"pythonw": target / "runtime" / "pythonw.exe", "launcher": target / "app" / "jarvis_launcher.pyw",
            "icon": target / "app" / "assets" / "jarvis.ico", "setup": target / "app" / "installer" / "setup_app.py",
            "workdir": target / "app"}


# -- Windows pieces ----------------------------------------------------------------------------
def make_shortcut(link: Path, target: Path, args: str, icon: Path, workdir: Path) -> None:
    """A .lnk through the Windows Script Host. Paths travel in environment variables, never in the script."""
    link.parent.mkdir(parents=True, exist_ok=True)
    script = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:J_LINK);$s.TargetPath=$env:J_TARGET;"
              "$s.Arguments=$env:J_ARGS;$s.WorkingDirectory=$env:J_DIR;$s.IconLocation=$env:J_ICON;"
              "$s.Description='Jarvis voice assistant';$s.Save()")
    env = {**os.environ, "J_LINK": str(link), "J_TARGET": str(target), "J_ARGS": args, "J_DIR": str(workdir),
           "J_ICON": str(icon)}
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], env=env,
                   creationflags=NO_WINDOW, capture_output=True, timeout=60, check=False)
    if not link.exists():
        raise RuntimeError(f"Couldn't create the shortcut {link.name}.")


def running_in(folder: Path) -> list:
    """Jarvis processes started from this folder (its own python)."""
    import psutil

    folder = folder.resolve()
    me = os.getpid()
    found = []
    for p in psutil.process_iter(["pid", "exe"]):
        try:
            exe = p.info.get("exe")
            if exe and p.info["pid"] != me and folder in Path(exe).resolve().parents:
                found.append(p)
        except (psutil.Error, OSError, ValueError):
            continue
    return found


def stop_running(folder: Path, wait: float = 8.0) -> bool:
    import psutil

    procs = running_in(folder)
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=wait)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    return not running_in(folder)


def registry_set(path: str, values: dict) -> None:
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path) as key:
        for name, value in values.items():
            kind = winreg.REG_DWORD if isinstance(value, int) else winreg.REG_SZ
            winreg.SetValueEx(key, name, 0, kind, value)


def registry_get(path: str, name: str) -> str | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            return str(winreg.QueryValueEx(key, name)[0])
    except OSError:
        return None


def registry_delete(path: str, value: str | None = None) -> None:
    import winreg

    try:
        if value is None:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
        else:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, value)
    except OSError:
        pass


def installed_dir() -> Path | None:
    where = registry_get(uninstall_key(), "InstallLocation")
    return Path(where) if where else None


def folder_size_kb(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) // 1024


# -- install -----------------------------------------------------------------------------------
def install(target: Path, desktop_icon: bool = True, menu_icon: bool = True, autostart: bool = False,
            progress=lambda pct, text: None, source: Path | None = None) -> Path:
    """Copy the package into `target` (an older version is swapped out safely). Returns pythonw.exe."""
    source = (source or package_root()).resolve()
    target = Path(target).resolve()
    if not (source / "runtime" / "pythonw.exe").is_file() or not (source / "app" / "jarvis").is_dir():
        raise RuntimeError("This setup folder is incomplete. Extract the whole zip and run Install Jarvis.cmd again.")
    if target == source or source in target.parents:
        raise RuntimeError("Choose a different folder - this is where the setup files are.")
    if target.exists() and any(target.iterdir()) and not (target / "runtime").is_dir():
        raise RuntimeError(f"{target} already has other files in it. Choose an empty folder or a new one.")
    progress(3, "Closing Jarvis if it is running…")
    if target.exists() and not stop_running(target):
        raise RuntimeError("Jarvis is still running. Close it and try again.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + f".new-{int(time.time())}")
    shutil.rmtree(staging, ignore_errors=True)
    files = [f for f in source.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    try:
        for i, f in enumerate(files):
            dest = staging / f.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(f, dest)  # same drive: instant, and no second copy of 250 MB
            except OSError:
                shutil.copy2(f, dest)
            if i % 150 == 0:
                progress(5 + int(80 * i / max(1, len(files))), "Copying files…")
        progress(87, "Replacing the old version…")
        old = None
        if target.exists():
            old = target.with_name(target.name + f".old-{int(time.time())}")
            target.rename(old)  # swap folders, so a failure can be undone
        try:
            staging.rename(target)
        except OSError:
            if old is not None:
                old.rename(target)
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    p = paths(target)
    launch_args = f'"{p["launcher"]}"'
    progress(90, "Adding shortcuts…")
    for wanted, link in ((menu_icon, start_menu() / f"{APP}.lnk"), (desktop_icon, desktop() / f"{APP}.lnk")):
        if wanted:
            make_shortcut(link, p["pythonw"], launch_args, p["icon"], p["workdir"])
        else:
            link.unlink(missing_ok=True)
    if autostart:
        registry_set(RUN_KEY, {run_value(): f'"{p["pythonw"]}" {launch_args}'})
    else:  # unticked on an update: remove our old entry (never someone else's)
        old_run = registry_get(RUN_KEY, run_value())
        if old_run and str(target).lower() in old_run.lower():
            registry_delete(RUN_KEY, run_value())
    progress(96, "Registering…")
    registry_set(uninstall_key(), {
        "DisplayName": APP, "DisplayVersion": version(), "Publisher": PUBLISHER, "InstallLocation": str(target),
        "DisplayIcon": str(p["icon"]),
        "UninstallString": f'"{p["pythonw"]}" "{p["setup"]}" --uninstall',
        "QuietUninstallString": f'"{p["pythonw"]}" "{p["setup"]}" --uninstall --silent',
        "EstimatedSize": folder_size_kb(target), "NoModify": 1, "NoRepair": 1,
        "URLInfoAbout": "https://github.com/musaddiq-vd/Mini-projects",
    })
    progress(100, "Done!")
    return p["pythonw"]


def launch(target: Path) -> None:
    p = paths(target)
    subprocess.Popen([str(p["pythonw"]), str(p["launcher"])], cwd=str(p["workdir"]), close_fds=True,
                     creationflags=DETACHED)


# -- uninstall ---------------------------------------------------------------------------------
def uninstall(target: Path, delete_data: bool = False) -> None:
    target = Path(target).resolve()
    if not (target / "runtime").is_dir() or not (target / "app").is_dir():
        raise RuntimeError(f"Jarvis isn't installed in {target}.")
    if not stop_running(target):
        raise RuntimeError("Jarvis is still running. Close it and try again.")
    (start_menu() / f"{APP}.lnk").unlink(missing_ok=True)
    (desktop() / f"{APP}.lnk").unlink(missing_ok=True)
    run = registry_get(RUN_KEY, run_value())
    if run and str(target).lower() in run.lower():
        registry_delete(RUN_KEY, run_value())
    registry_delete(uninstall_key())
    if delete_data:
        shutil.rmtree(Path(os.environ.get("JARVIS_HOME") or Path.home() / ".jarvis"), ignore_errors=True)
    me = Path(sys.executable).resolve()
    if target in me.parents:
        # this setup runs on the installed python, which can't delete its own folder: remove it right after we exit
        env = {**os.environ, "J_REMOVE": str(target)}  # the path is data, never part of the command
        subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                          "Start-Sleep -Seconds 3; Remove-Item -LiteralPath $env:J_REMOVE -Recurse -Force"],
                         env=env, creationflags=NO_WINDOW, close_fds=True)
    else:
        shutil.rmtree(target, ignore_errors=True)


# -- windows -----------------------------------------------------------------------------------
STYLE = """
QWidget { background: #0a0f14; color: #d8e6ef; font-family: 'Segoe UI'; font-size: 10pt; }
QLabel#title { color: #00d4ff; font-size: 20pt; font-weight: 700; }
QLabel#muted { color: #6f8594; font-size: 9pt; }
QLineEdit { background: #16212b; border: 1px solid #22303c; border-radius: 6px; padding: 6px; }
QPushButton { background: #16212b; border: 1px solid #22303c; border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { border-color: #00d4ff; }
QPushButton#primary { background: #00d4ff; color: #06121a; font-weight: 700; padding: 8px 26px; border: none; }
QPushButton#primary:disabled { background: #22303c; color: #6f8594; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #4a6070; border-radius: 4px; background: #16212b; }
QCheckBox::indicator:checked { background: #00d4ff; border-color: #00d4ff; }
QProgressBar { background: #16212b; border: none; border-radius: 4px; height: 8px; text-align: center; }
QProgressBar::chunk { background: #00d4ff; border-radius: 4px; }
"""


def gui_install() -> int:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QIcon, QPixmap
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
        QPushButton, QVBoxLayout, QWidget,
    )

    qt = QApplication.instance() or QApplication(sys.argv)
    qt.setStyleSheet(STYLE)
    assets = package_root() / "app" / "assets"
    win = QWidget()
    win.setWindowTitle(f"{APP} Setup")
    win.setWindowIcon(QIcon(str(assets / "jarvis.ico")))
    win.setFixedWidth(560)
    v = QVBoxLayout(win)
    v.setContentsMargins(26, 22, 26, 22)
    head = QHBoxLayout()
    logo = QLabel()
    logo.setPixmap(QPixmap(str(assets / "jarvis.png")).scaled(76, 76, Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation))
    head.addWidget(logo)
    existing = installed_dir()
    texts = QVBoxLayout()
    texts.addWidget(QLabel(f"{APP} {version()}", objectName="title"))
    texts.addWidget(QLabel("Update to this version - your settings and data are kept." if existing else
                           "Your real-time voice AI assistant for Windows."))
    head.addLayout(texts, 1)
    v.addLayout(head)
    v.addSpacing(10)
    v.addWidget(QLabel("Install folder", objectName="muted"))
    row = QHBoxLayout()
    path = QLineEdit(str(existing or default_dir()))
    row.addWidget(path, 1)
    browse = QPushButton("Browse…")
    browse.clicked.connect(lambda: (lambda d: d and path.setText(str(Path(d) / APP)))(
        QFileDialog.getExistingDirectory(win, "Install folder", path.text())))
    row.addWidget(browse)
    v.addLayout(row)
    boxes = {}
    for key, text, on in (("desktop", "Desktop shortcut", True), ("menu", "Start Menu shortcut", True),
                          ("autostart", "Start with Windows", False), ("launch", f"Open {APP} when setup finishes", True)):
        boxes[key] = QCheckBox(text)
        boxes[key].setChecked(on)
        v.addWidget(boxes[key])
    bar = QProgressBar()
    bar.setTextVisible(False)
    v.addSpacing(8)
    v.addWidget(bar)
    status = QLabel("No admin rights needed. Nothing is sent anywhere during setup.", objectName="muted")
    v.addWidget(status)
    go = QPushButton("Update" if existing else "Install", objectName="primary")
    go.setCursor(Qt.CursorShape.PointingHandCursor)
    v.addWidget(go, 0, Qt.AlignmentFlag.AlignRight)
    events: list = []

    def work():
        try:
            target = Path(path.text().strip())
            install(target, boxes["desktop"].isChecked(), boxes["menu"].isChecked(), boxes["autostart"].isChecked(),
                    lambda pct, text: events.append(("progress", pct, text)))
            events.append(("done", target, ""))
        except Exception as exc:
            events.append(("error", str(exc), ""))

    def pump():
        while events:
            kind, a, b = events.pop(0)
            if kind == "progress":
                bar.setValue(a)
                status.setText(b)
            elif kind == "error":
                go.setEnabled(True)
                status.setText("")
                QMessageBox.warning(win, f"{APP} Setup", a)
                return
            elif kind == "done":
                if boxes["launch"].isChecked():
                    launch(a)
                QMessageBox.information(win, f"{APP} Setup", f"{APP} is installed. Say \"{APP}\" any time!")
                win.close()
                return
        QTimer.singleShot(80, pump)

    def start():
        if not path.text().strip():
            return
        go.setEnabled(False)
        threading.Thread(target=work, daemon=True).start()
        QTimer.singleShot(80, pump)

    go.clicked.connect(start)
    win.show()
    return qt.exec()


def gui_uninstall(target: Path) -> int:
    from PyQt6.QtWidgets import QApplication, QMessageBox

    qt = QApplication.instance() or QApplication(sys.argv)
    qt.setStyleSheet(STYLE)
    yes, no = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
    if QMessageBox.question(None, f"Uninstall {APP}", f"Remove {APP} from this computer?", yes | no, no) != yes:
        return 0
    delete_data = QMessageBox.question(
        None, f"Uninstall {APP}", "Also delete your Jarvis data (memory, chat history, notes, settings)?\n\n"
                                  "Choose No to keep it for a future install.", yes | no, no) == yes
    try:
        uninstall(target, delete_data)
        QMessageBox.information(None, f"Uninstall {APP}", f"{APP} was removed.")
    except Exception as exc:
        QMessageBox.warning(None, f"Uninstall {APP}", str(exc))
    return 0


def main(argv: list[str]) -> int:
    silent = "--silent" in argv
    target = None
    if "--dir" in argv and argv.index("--dir") + 1 < len(argv):
        target = Path(argv[argv.index("--dir") + 1])
    if "--uninstall" in argv:
        target = target or installed_dir() or package_root()
        if silent:
            uninstall(target, delete_data="--delete-data" in argv)
            return 0
        return gui_uninstall(target)
    if silent:
        install(target or installed_dir() or default_dir(), desktop_icon="--no-desktop" not in argv,
                menu_icon="--no-menu" not in argv, autostart="--autostart" in argv)
        if "--launch" in argv:
            launch(target or installed_dir() or default_dir())
        return 0
    return gui_install()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
