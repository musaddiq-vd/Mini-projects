"""Settings > Home: connect Home Assistant and add Tuya / Smart Life devices."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from jarvis.home import KINDS

COLUMNS = ("Name", "Type", "Room", "Device ID", "IP", "Local key", "Version")
HELP_HA = ("Home Assistant: open your profile in Home Assistant → Security → Long-lived access tokens → Create, "
           "and paste it here with the address (like http://homeassistant.local:8123).")
HELP_TUYA = ("Tuya / Smart Life (Wipro, Syska, Havells, Halonix ...): each device needs its ID and local key. "
             "Get them once with the free tinytuya wizard (python -m tinytuya wizard), then use Import to load "
             "its devices.json. Jarvis then talks to the devices directly on your WiFi.")


class HomeTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.s = settings
        v = QVBoxLayout(self)
        v.addWidget(QLabel("HOME ASSISTANT", objectName="section"))
        form = QFormLayout()
        self.ha_url = QLineEdit(settings.ha_url)
        self.ha_url.setPlaceholderText("http://homeassistant.local:8123")
        form.addRow("Address", self.ha_url)
        token_row = QHBoxLayout()
        self.ha_token = QLineEdit(settings.ha_token)
        self.ha_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.ha_token.setPlaceholderText("Long-lived access token")
        token_row.addWidget(self.ha_token, 1)
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self.test_ha)
        token_row.addWidget(self.test_btn)
        form.addRow("Token", token_row)
        v.addLayout(form)
        self.ha_status = QLabel(HELP_HA, objectName="muted")
        self.ha_status.setWordWrap(True)
        v.addWidget(self.ha_status)

        v.addSpacing(8)
        v.addWidget(QLabel("TUYA / SMART LIFE DEVICES", objectName="section"))
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(150)
        v.addWidget(self.table, 1)
        for d in settings.tuya_devices:
            self.add_row(d)
        row = QHBoxLayout()
        add = QPushButton("＋  Add device")
        add.clicked.connect(lambda: self.add_row({}))
        row.addWidget(add)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self.remove_selected)
        row.addWidget(remove)
        imp = QPushButton("Import devices.json…")
        imp.clicked.connect(self.import_file)
        row.addWidget(imp)
        row.addStretch(1)
        v.addLayout(row)
        self.tuya_status = QLabel(HELP_TUYA, objectName="muted")
        self.tuya_status.setWordWrap(True)
        v.addWidget(self.tuya_status)

    # -- Tuya table -----------------------------------------------------------------------------
    def add_row(self, d: dict) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        values = [d.get("name", ""), None, d.get("room", ""), d.get("id", ""), d.get("ip", ""), d.get("key", ""),
                  str(d.get("version", "3.3") or "3.3")]
        for c, val in enumerate(values):
            if c == 1:
                box = QComboBox()
                for k in KINDS[:-1]:  # scenes come only from Home Assistant
                    box.addItem(k, k)
                box.setCurrentIndex(max(0, box.findData(d.get("type", "plug"))))
                self.table.setCellWidget(r, 1, box)
            else:
                self.table.setItem(r, c, QTableWidgetItem(val))

    def remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def import_file(self, path: str | None = None) -> int:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "tinytuya devices.json", str(Path.home()), "JSON (*.json)")
        if not path:
            return 0
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.tuya_status.setText("⚠ That file isn't a devices.json from tinytuya.")
            return 0
        known = {self._cell(r, 3) for r in range(self.table.rowCount())}
        added = 0
        for d in data if isinstance(data, list) else []:
            if not isinstance(d, dict) or not d.get("id") or not d.get("key") or d["id"] in known:
                continue
            name = d.get("name") or d["id"]
            low = (name + " " + str(d.get("category", ""))).lower()
            kind = ("light" if any(w in low for w in ("light", "bulb", "lamp", "dj", "dd")) else
                    "fan" if "fan" in low else "ac" if any(w in low for w in ("ac", "air", "kt")) else "plug")
            self.add_row({"name": name, "type": kind, "id": d["id"], "key": d["key"], "ip": d.get("ip", ""),
                          "version": d.get("version", "3.3")})
            added += 1
        self.tuya_status.setText(f"✓ Imported {added} device{'s' if added != 1 else ''}. Check the types, then Save."
                                 if added else "No new devices in that file.")
        return added

    def _cell(self, r: int, c: int) -> str:
        item = self.table.item(r, c)
        return item.text().strip() if item else ""

    def tuya_devices(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            dev_id, key = self._cell(r, 3), self._cell(r, 5)
            if not dev_id or not key:
                continue  # half-filled rows are skipped
            box = self.table.cellWidget(r, 1)
            try:
                version = float(self._cell(r, 6) or 3.3)
            except ValueError:
                version = 3.3
            out.append({"name": self._cell(r, 0) or dev_id, "type": box.currentData() if box else "plug",
                        "room": self._cell(r, 2), "id": dev_id, "ip": self._cell(r, 4) or "Auto", "key": key,
                        "version": version})
        return out

    # -- Home Assistant -------------------------------------------------------------------------
    def test_ha(self) -> None:
        from jarvis.home import HomeAssistant, HomeError

        url, token = self.ha_url.text().strip(), self.ha_token.text().strip()
        if not url.startswith(("http://", "https://")) or not token:
            self.ha_status.setText("⚠ Add the address (http://…) and a token first.")
            return
        self.test_btn.setEnabled(False)
        self.ha_status.setText("Checking…")
        result: dict = {}

        def work():
            try:
                ha = HomeAssistant(url, token)
                ha.check()
                result["text"] = f"✓ Connected - {len(ha.devices())} devices found."
            except HomeError as exc:
                result["text"] = f"⚠ {exc}"
            except Exception as exc:  # anything unexpected is shown, never raised in the dialog
                result["text"] = f"⚠ {type(exc).__name__}"

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def wait():
            if thread.is_alive():
                QTimer.singleShot(150, wait)
                return
            self.test_btn.setEnabled(True)
            self.ha_status.setText(result.get("text", ""))

        QTimer.singleShot(150, wait)

    def apply_to(self, s) -> None:
        url = self.ha_url.text().strip().rstrip("/")
        s.ha_url = url if url.startswith(("http://", "https://")) else ""
        s.ha_token = self.ha_token.text().strip()
        s.tuya_devices = self.tuya_devices()


__all__ = ["HomeTab", "Qt"]
