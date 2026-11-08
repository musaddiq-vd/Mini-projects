"""Dashboard (Ctrl+D): your week, habits, money and goals - with charts and quick actions.

Charts are painted with QPainter (no extra packages). Every change goes through Jarvis's tools, so
guardrails and undo apply exactly as they do by voice.
"""

from __future__ import annotations

import html
from collections import defaultdict
from datetime import date, datetime, timedelta

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QLineEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)

from jarvis.ui import theme


# -- small building blocks ------------------------------------------------------------------------
class BarChart(QWidget):
    """Vertical bars with labels under them and values on top."""

    def __init__(self, values: list[float], labels: list[str], color: str | None = None, fmt=lambda v: f"{v:g}",
                 highlight_last: bool = True, parent=None):
        super().__init__(parent)
        self.values, self.labels, self.color, self.fmt = values, labels, color, fmt
        self.highlight_last = highlight_last
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = max(1, len(self.values))
        w, h = self.width(), self.height()
        top, bottom = 18, 18
        slot = w / n
        bar = min(34.0, slot * 0.6)
        peak = max(self.values) if self.values and max(self.values) > 0 else 1
        accent = QColor(self.color or theme.ACCENT)
        p.setFont(QFont("Segoe UI", 8))
        for i, (v, lab) in enumerate(zip(self.values, self.labels)):
            x = slot * i + (slot - bar) / 2
            full = h - top - bottom
            bh = max(2.0, full * (v / peak)) if v else 2.0
            track = QRectF(x, top, bar, full)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.P.track)
            p.drawRoundedRect(track, 6, 6)
            c = QColor(accent)
            if self.highlight_last and i < n - 1:
                c.setAlpha(150)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x, top + full - bh, bar, bh), 6, 6)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(slot * i, h - bottom + 2, slot, bottom), Qt.AlignmentFlag.AlignCenter, lab)
            if v:
                p.setPen(QColor(theme.TEXT))
                p.drawText(QRectF(slot * i, top + full - bh - 16, slot, 14), Qt.AlignmentFlag.AlignCenter, self.fmt(v))
        p.end()


class Dots(QWidget):
    """Seven dots: met (filled), missed (ring), not counted (faint)."""

    def __init__(self, week: list, parent=None):
        super().__init__(parent)
        self.week = week
        self.setFixedSize(7 * 22, 30)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(QFont("Segoe UI", 7))
        for i, (letter, _count, met) in enumerate(self.week):
            cx = i * 22 + 11
            if met is True:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(theme.OK))
            elif met is False:
                p.setPen(QColor(theme.DANGER))
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(theme.P.track)
            p.drawEllipse(QRectF(cx - 6, 2, 12, 12))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(cx - 8, 16, 16, 12), Qt.AlignmentFlag.AlignCenter, letter)
        p.end()


def _card(title: str = "") -> tuple[QFrame, QVBoxLayout]:
    card = QFrame(objectName="card")
    v = QVBoxLayout(card)
    v.setContentsMargins(14, 12, 14, 12)
    v.setSpacing(8)
    if title:
        v.addWidget(QLabel(title, objectName="h2"))
    return card, v


def _big(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {theme.readable(theme.ACCENT)}; background: transparent;")
    return lab


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            _clear(item.layout())


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


# -- data helpers ---------------------------------------------------------------------------------
def last_days(n: int = 7, today: date | None = None) -> list[date]:
    today = today or date.today()
    return [today - timedelta(days=i) for i in range(n - 1, -1, -1)]


def week_numbers(app, today: date | None = None) -> dict[str, list[float]]:
    """Per day for the last 7 days: messages, tasks done, focus minutes, money spent, habits met."""
    days = last_days(7, today)
    keys = [d.isoformat() for d in days]
    out = {k: [0.0] * 7 for k in ("messages", "tasks", "focus", "spent", "habits")}
    idx = {k: i for i, k in enumerate(keys)}
    for item in (app.history.recent(5000) if app.history else []):
        k = datetime.fromtimestamp(item.get("t", 0)).date().isoformat()
        if k in idx and item.get("role") == "user":
            out["messages"][idx[k]] += 1
    for n in app.state.get("notes", []) or []:
        if n.get("done_at"):
            k = datetime.fromtimestamp(n["done_at"]).date().isoformat()
            if k in idx:
                out["tasks"][idx[k]] += 1
    for f in app.state.get("focus_log", []) or []:
        if f.get("date") in idx:
            out["focus"][idx[f["date"]]] += f.get("minutes", 0)
    from jarvis.tools.goals import goals
    from jarvis.tools.habits import done_on, habits
    from jarvis.tools.money import expenses

    for e in expenses().load()["items"]:
        if e["date"] in idx:
            out["spent"][idx[e["date"]]] += e["amount"]
    for h in habits().load()["habits"]:
        for d in days:
            if done_on(h, d):
                out["habits"][idx[d.isoformat()]] += 1
    for g in goals().load()["goals"]:
        for s in g["steps"]:
            if s.get("done_at"):
                k = datetime.fromtimestamp(s["done_at"]).date().isoformat()
                if k in idx:
                    out["tasks"][idx[k]] += 1
    return out


# -- the panel --------------------------------------------------------------------------------------
class DashboardPanel(QFrame):
    def __init__(self, parent: QWidget, app, send):
        super().__init__(parent)
        self.setObjectName("dashPanel")
        self.app, self.send = app, send
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        head = QHBoxLayout()
        head.addWidget(QLabel("DASHBOARD", objectName="section"))
        head.addStretch(1)
        close = QPushButton("✕", objectName="small")
        close.clicked.connect(self.hide)
        head.addWidget(close)
        outer.addLayout(head)
        self.tabs = QTabWidget()
        self.pages: dict[str, QVBoxLayout] = {}
        for key, title in (("week", "📊  Week"), ("habits", "✅  Habits"), ("money", "💰  Money"), ("goals", "🎯  Goals")):
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)  # grow and scroll, never squash
            lay.setContentsMargins(4, 8, 12, 8)
            lay.setSpacing(10)
            self.pages[key] = lay
            self.tabs.addTab(_scroll(inner), title)
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        outer.addWidget(self.tabs, 1)
        self.hide()

    # -- open ------------------------------------------------------------------------------------------
    def toggle(self, tab: str | None = None) -> None:
        if self.isVisible() and tab is None:
            self.hide()
            return
        self.setStyleSheet(f"QFrame#dashPanel {{ background: {theme.BG}; border: 1px solid {theme.ACCENT}; "
                           "border-radius: 16px; }")
        if tab in self.pages:
            self.tabs.setCurrentIndex(list(self.pages).index(tab))
        self.setGeometry(self.parentWidget().rect().adjusted(8, 8, -8, -8))
        self.show()
        self.raise_()
        self.refresh()

    def _call(self, tool: str, args: dict) -> dict:
        result = self.app.registry.call(tool, args, self.app.ctx)
        if "error" in result:
            self.app.events.emit("log", text=f"⚠ {result['error'][:160]}")
        self.refresh()
        return result

    def refresh(self) -> None:
        if not self.isVisible():
            return
        key = list(self.pages)[self.tabs.currentIndex()]
        lay = self.pages[key]
        _clear(lay)
        try:
            getattr(self, f"_build_{key}")(lay)
        except Exception as exc:  # a broken data file must never take the window down
            lay.addWidget(QLabel(f"Couldn't show this page: {html.escape(str(exc))}", objectName="bad"))
        lay.addStretch(1)

    # -- week ------------------------------------------------------------------------------------------
    def _build_week(self, lay: QVBoxLayout) -> None:
        nums = week_numbers(self.app)
        labels = [d.strftime("%a") for d in last_days(7)]
        cur = getattr(self.app.settings, "currency", "₹")
        totals = QHBoxLayout()
        for title, key, fmt in (("💬 Messages", "messages", "{:.0f}"), ("✅ Tasks done", "tasks", "{:.0f}"),
                                ("🎯 Focus", "focus", "{:.0f} min"), ("💸 Spent", "spent", cur + "{:,.0f}")):
            card, v = _card()
            v.addWidget(QLabel(title, objectName="muted"))
            v.addWidget(_big(fmt.format(sum(nums[key]))))
            v.addWidget(QLabel("last 7 days", objectName="muted"))
            totals.addWidget(card)
        lay.addLayout(totals)
        grid = QGridLayout()
        grid.setSpacing(10)
        charts = [("💬 Messages per day", "messages", lambda v: f"{v:.0f}"),
                  ("✅ Tasks and steps done", "tasks", lambda v: f"{v:.0f}"),
                  ("🎯 Focus minutes", "focus", lambda v: f"{v:.0f}"),
                  ("💸 Spending", "spent", lambda v: f"{cur}{v:,.0f}")]
        for i, (title, key, fmt) in enumerate(charts):
            card, v = _card(title)
            v.addWidget(BarChart(nums[key], labels, fmt=fmt))
            grid.addWidget(card, i // 2, i % 2)
        lay.addLayout(grid)
        ach = getattr(self.app, "achievements", None)
        if ach is not None:
            card, v = _card()
            v.addWidget(QLabel(f"🔥 {ach.streak()}-day streak   ·   🏆 {len(ach.unlocked())} achievements   ·   "
                               f"✅ {sum(nums['habits']):.0f} habit check-ins this week", objectName="h2"))
            lay.addWidget(card)

    # -- habits ------------------------------------------------------------------------------------------
    def _build_habits(self, lay: QVBoxLayout) -> None:
        from jarvis.tools.habits import habits, streak, week

        add = QHBoxLayout()
        name = QLineEdit()
        name.setPlaceholderText("New habit, e.g. 'Read 10 pages'")
        target = QComboBox()
        for n in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            target.addItem(f"{n}× a day", n)
        btn = QPushButton("＋ Add habit", objectName="primary")

        def do_add():
            if name.text().strip():
                self._call("add_habit", {"name": name.text().strip(), "target": target.currentData()})

        btn.clicked.connect(do_add)
        name.returnPressed.connect(do_add)
        add.addWidget(name, 1)
        add.addWidget(target)
        add.addWidget(btn)
        lay.addLayout(add)
        items = habits().load()["habits"]
        if not items:
            lay.addWidget(QLabel("No habits yet - add one above, or say \"make gym a daily habit\".", objectName="muted"))
            return
        today = date.today().isoformat()
        for h in items:
            card, v = _card()
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{h['icon']}  <b>{html.escape(h['name'])}</b>"), 1)
            s = streak(h)
            row.addWidget(QLabel(f"🔥 {s}" if s else "", objectName="accent"))
            v.addLayout(row)
            row2 = QHBoxLayout()
            count, target_n = h["log"].get(today, 0), h.get("target", 1)
            bar = QProgressBar()
            bar.setMaximum(target_n)
            bar.setValue(min(count, target_n))
            bar.setTextVisible(False)
            row2.addWidget(bar, 1)
            row2.addWidget(QLabel(f"{count}/{target_n} today", objectName="muted"))
            row2.addWidget(Dots(week(h)))
            plus = QPushButton("＋1")
            plus.setToolTip("Mark one more for today")
            plus.clicked.connect(lambda _, n=h["name"]: self._call("check_habit", {"name": n}))
            row2.addWidget(plus)
            minus = QPushButton("−", objectName="small")
            minus.setToolTip("Undo one for today")
            minus.clicked.connect(lambda _, n=h["name"]: self._call("check_habit", {"name": n, "count": -1}))
            row2.addWidget(minus)
            row2.addWidget(_trash(lambda _, n=h["name"]: self._call("delete_habit", {"name": n})))
            v.addLayout(row2)
            lay.addWidget(card)

    # -- money ---------------------------------------------------------------------------------------------
    def _build_money(self, lay: QVBoxLayout) -> None:
        from jarvis.tools.money import CATEGORIES, ICONS, expenses, money

        ctx = self.app.ctx
        data = expenses().load()
        add = QHBoxLayout()
        amount = QLineEdit()
        amount.setPlaceholderText("Amount")
        amount.setMaximumWidth(110)
        note = QLineEdit()
        note.setPlaceholderText("What for? (e.g. chai)")
        cat = QComboBox()
        for c in CATEGORIES:
            cat.addItem(f"{ICONS[c]} {c}", c)
        btn = QPushButton("＋ Add", objectName="primary")
        error = QLabel("", objectName="bad")
        error.hide()

        def do_add():
            try:
                value = float(amount.text().replace(",", "").strip())
            except ValueError:
                error.setText("Type the amount as a number.")
                error.show()
                return
            self._call("add_expense", {"amount": value, "category": cat.currentData(), "note": note.text().strip()})

        btn.clicked.connect(do_add)
        amount.returnPressed.connect(do_add)
        note.returnPressed.connect(do_add)
        for w_ in (amount, note, cat, btn):
            add.addWidget(w_, 1 if w_ is note else 0)
        lay.addLayout(add)
        lay.addWidget(error)

        today = date.today()
        month = [e for e in data["items"] if e["date"][:7] == today.isoformat()[:7]]
        total = sum(e["amount"] for e in month)
        card, v = _card()
        v.addWidget(QLabel(f"This month · {today:%B}", objectName="muted"))
        v.addWidget(_big(money(ctx, total)))
        limit = data["budgets"].get("total")
        if limit:
            bar = QProgressBar()
            bar.setMaximum(1000)
            bar.setValue(min(1000, int(1000 * total / limit)))
            bar.setTextVisible(False)
            if total > limit:
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {theme.DANGER}; border-radius: 5px; }}")
            v.addWidget(bar)
            v.addWidget(QLabel(f"{money(ctx, total)} of {money(ctx, limit)} budget ({total / limit:.0%})",
                               objectName="muted"))
        else:
            v.addWidget(QLabel("Tip: say \"set my monthly budget to 15000\"", objectName="muted"))
        lay.addWidget(card)

        days = last_days(14)
        per_day = defaultdict(float)
        for e in data["items"]:
            per_day[e["date"]] += e["amount"]
        cur = getattr(self.app.settings, "currency", "₹")
        chart, cv = _card("📅 Last 14 days")
        cv.addWidget(BarChart([per_day[d.isoformat()] for d in days], [d.strftime("%d") for d in days],
                              fmt=lambda val: f"{cur}{val:,.0f}"))
        lay.addWidget(chart)

        by_cat = defaultdict(float)
        for e in month:
            by_cat[e["category"]] += e["amount"]
        if by_cat:
            cats, cv = _card("🗂 By category (this month)")
            top = max(by_cat.values())
            for c, amt in sorted(by_cat.items(), key=lambda kv: -kv[1]):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{ICONS.get(c, '💸')} {c}"), 0)
                bar = QProgressBar()
                bar.setMaximum(1000)
                bar.setValue(int(1000 * amt / top))
                bar.setTextVisible(False)
                row.addWidget(bar, 1)
                row.addWidget(QLabel(money(ctx, amt), objectName="h2"))
                cv.addLayout(row)
            lay.addWidget(cats)

        recent = sorted(data["items"], key=lambda e: (e["date"], e["id"]))[-8:][::-1]
        if recent:
            rc, rv = _card("🧾 Recent")
            for e in recent:
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{ICONS.get(e['category'], '💸')}  {html.escape(e['note'] or e['category'])}"
                                     f"  <span style='color:{theme.MUTED};'>· {e['date']}</span>"), 1)
                row.addWidget(QLabel(money(ctx, e["amount"]), objectName="h2"))
                row.addWidget(_trash(lambda _, i=e["id"]: self._call("delete_expense", {"id": i})))
                rv.addLayout(row)
            lay.addWidget(rc)

    # -- goals ----------------------------------------------------------------------------------------------
    def _build_goals(self, lay: QVBoxLayout) -> None:
        from jarvis.tools.goals import goals, progress

        ask = QPushButton("🎯  Set a new goal with Jarvis", objectName="primary")
        ask.clicked.connect(lambda: self.send("I want to set a new goal. Ask me what it is and help me break it "
                                              "into steps."))
        lay.addWidget(ask)
        items = [g for g in goals().load()["goals"]]
        if not items:
            lay.addWidget(QLabel("No goals yet. Tell Jarvis something you want to achieve - it will plan the steps.",
                                 objectName="muted"))
            return
        for g in sorted(items, key=lambda g: (g.get("done", False), g["deadline"])):
            p = progress(g)
            card, v = _card()
            head = QHBoxLayout()
            done_mark = "🏁 " if g.get("done") else ""
            head.addWidget(QLabel(f"{done_mark}<b>{html.escape(g['title'])}</b>"), 1)
            left = "finished" if g.get("done") else (f"{p['days_left']} days left" if p["days_left"] >= 0 else "past deadline")
            head.addWidget(QLabel(left, objectName="muted"))
            head.addWidget(_trash(lambda _, t=g["title"]: self._call("delete_goal", {"goal": t})))
            v.addLayout(head)
            if g.get("why"):
                v.addWidget(QLabel(f"💡 {html.escape(g['why'])}", objectName="muted"))
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(p["percent"])
            bar.setTextVisible(False)
            v.addWidget(bar)
            v.addWidget(QLabel(f"{p['done']}/{p['total']} steps · {p['percent']}%"
                               + (f" · ⚠ {p['overdue']} overdue" if p["overdue"] else ""), objectName="muted"))
            today = date.today().isoformat()
            for i, s in enumerate(g["steps"], 1):
                box = QCheckBox(f"{s['text']}  ·  {date.fromisoformat(s['due']):%d %b}".replace("&", "&&"))
                box.setChecked(bool(s.get("done")))
                box.setEnabled(not s.get("done") and not g.get("done"))
                if not s.get("done") and s["due"] < today:
                    box.setStyleSheet(f"color: {theme.DANGER};")
                box.toggled.connect(lambda on, t=g["title"], n=i: on and self._call(
                    "complete_goal_step", {"goal": t, "step": str(n)}))
                v.addWidget(box)
            lay.addWidget(card)


def _trash(fn) -> QPushButton:
    b = QPushButton("🗑", objectName="small")
    b.setToolTip("Delete (say 'undo' to bring it back)")
    b.clicked.connect(fn)
    return b
