"""Language practice screen: day plan, modules, quizzes, reading, listening, speaking, progress."""

from __future__ import annotations

import threading

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from jarvis.english.curriculum import LANGUAGES, LEVELS, MODULES, SKILLS, Task
from jarvis.ui import theme
from jarvis.ui.theme import DANGER, OK, WARN

SHORT = {"vocab": "Vocabulary", "grammar": "Grammar", "reading": "Reading", "listening": "Listening",
         "speak": "Speaking", "bonus": "Bonus"}
SKILL_ICONS = {"listening": "🎧", "reading": "📖", "speaking": "🗣", "grammar": "✍", "vocab": "📚"}


def _label(text: str = "", name: str = "", wrap: bool = True) -> QLabel:
    lab = QLabel(text, objectName=name) if name else QLabel(text)
    lab.setWordWrap(wrap)
    return lab


class SkillBars(QWidget):
    """Five small accuracy bars, one per skill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: dict[str, int | None] = {}
        self.setFixedHeight(40)

    def set_values(self, values: dict) -> None:
        self.values = values
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(SKILLS)
        gap = 10
        cell = (self.width() - gap * (n - 1)) / n
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for i, skill in enumerate(SKILLS):
            x = i * (cell + gap)
            v = self.values.get(skill)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(x, 0, cell, 15), Qt.AlignmentFlag.AlignLeft, f"{SKILL_ICONS[skill]} {skill.title()}")
            color = QColor(theme.readable(theme.ACCENT) if v is None or v >= 70 else WARN if v >= 40 else DANGER)
            p.setPen(color)
            p.drawText(QRectF(x, 0, cell, 15), Qt.AlignmentFlag.AlignRight, "-" if v is None else f"{v}%")
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.P.track)
            p.drawRoundedRect(QRectF(x, 22, cell, 6), 3, 3)
            if v:
                p.setBrush(color)
                p.drawRoundedRect(QRectF(x, 22, cell * v / 100, 6), 3, 3)
        p.end()


class QuizView(QWidget):
    """Multiple-choice questions one at a time, with instant feedback."""

    finished = pyqtSignal(int, int, list)  # score, total, missed questions
    answered = pyqtSignal(bool)  # right or wrong, for the orb to react

    def __init__(self, questions: list[dict], intro: str = "", parent=None):
        super().__init__(parent)
        self.questions = questions
        self.i = 0
        self.score = 0
        self.missed: list[dict] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        if intro:
            v.addWidget(_label(intro, "muted"))
        self.counter = _label("", "section")
        v.addWidget(self.counter)
        self.question = _label("", "h2")
        v.addWidget(self.question)
        self.buttons: list[QPushButton] = []
        for k in range(4):
            b = QPushButton(objectName="option")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, k=k: self.answer(k))
            v.addWidget(b)
            self.buttons.append(b)
        self.why = _label("", "muted")
        v.addWidget(self.why)
        self.next = QPushButton("Next  →", objectName="primary")
        self.next.clicked.connect(self._advance)
        self.next.hide()
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.next)
        v.addLayout(row)
        self._show()

    def _show(self) -> None:
        q = self.questions[self.i]
        self.counter.setText(f"QUESTION {self.i + 1} OF {len(self.questions)}")
        self.question.setText(q["q"])
        for k, b in enumerate(self.buttons):
            if k < len(q["options"]):
                b.setText(f"{'ABCD'[k]}.   {q['options'][k]}")
                b.setObjectName("option")
                b.setEnabled(True)
                b.show()
                b.style().unpolish(b)
                b.style().polish(b)
            else:
                b.hide()
        self.why.setText("")
        self.next.hide()

    def answer(self, k: int) -> None:
        q = self.questions[self.i]
        right = q["answer"]
        self.answered.emit(k == right)
        if k == right:
            self.score += 1
            verdict = "✓ Correct!"
        else:
            self.missed.append(q)
            verdict = f"✗ The answer is {'ABCD'[right]}."
        for j, b in enumerate(self.buttons):
            b.setEnabled(False)
            if j == right:
                b.setObjectName("optionGood")
            elif j == k:
                b.setObjectName("optionBad")
            b.style().unpolish(b)
            b.style().polish(b)
        self.why.setText(f"{verdict}  {q.get('why', '')}")
        self.next.setText("Finish  ✓" if self.i == len(self.questions) - 1 else "Next  →")
        self.next.show()

    def _advance(self) -> None:
        if self.i < len(self.questions) - 1:
            self.i += 1
            self._show()
        else:
            self.finished.emit(self.score, len(self.questions), self.missed)


class PracticePanel(QWidget):
    """The whole practice screen. `say(text)` sends a quiet instruction to the live tutor."""

    plan_ready = pyqtSignal(bool)  # emitted from the worker thread; Qt queues it onto the GUI thread

    def __init__(self, app, say, parent=None):
        super().__init__(parent)
        self.app = app
        self.store = app.english
        self.say = say  # callable(str) or None when not connected
        self.current: Task | None = None
        self._loading = False
        self.plan_ready.connect(self._loaded)
        self._build()
        self.refresh()

    # -- layout ---------------------------------------------------------------------------
    def _build(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        head = QHBoxLayout()
        self.profile_lbl = _label("", "muted", wrap=False)
        head.addWidget(self.profile_lbl)
        head.addStretch(1)
        self.day_lbl = _label("", "accent", wrap=False)
        self.streak_lbl = _label("", "h2", wrap=False)
        self.xp_lbl = _label("", "h2", wrap=False)
        for w in (self.day_lbl, self.streak_lbl, self.xp_lbl):
            head.addSpacing(10)
            head.addWidget(w)
        v.addLayout(head)

        prefs = QHBoxLayout()
        self.native = QComboBox()
        self.native.addItems(LANGUAGES)
        self.target = QComboBox()
        self.target.addItems(LANGUAGES)
        self.level = QComboBox()
        for name in LEVELS:
            self.level.addItem(name.title(), name)
        prefs.addWidget(_label("I speak", "muted", wrap=False))
        prefs.addWidget(self.native, 1)
        prefs.addWidget(_label("Learning", "muted", wrap=False))
        prefs.addWidget(self.target, 1)
        prefs.addWidget(_label("Level", "muted", wrap=False))
        prefs.addWidget(self.level, 1)
        v.addLayout(prefs)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.module_btns: dict[str, QPushButton] = {}
        for i, (mid, (icon, title, _)) in enumerate(MODULES.items()):
            b = QPushButton(f"{icon}  {title.replace('&', '&&')}", objectName="chip")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, m=mid: self._set_focus(m))
            grid.addWidget(b, i // 4, i % 4)
            self.module_btns[mid] = b
        v.addLayout(grid)

        prog = QHBoxLayout()
        self.today_lbl = _label("", "muted", wrap=False)
        prog.addWidget(self.today_lbl)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        prog.addWidget(self.bar, 1)
        v.addLayout(prog)
        self.skills = SkillBars()
        v.addWidget(self.skills)

        self.tasks_row = QHBoxLayout()
        self.tasks_row.setSpacing(6)
        v.addLayout(self.tasks_row)

        # exercise area
        self.area = QFrame(objectName="card")
        self.area_layout = QVBoxLayout(self.area)
        self.area_layout.setContentsMargins(16, 14, 16, 14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.area)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        v.addWidget(scroll, 1)

        # corrections
        self.feedback = QVBoxLayout()
        self.feedback.setSpacing(6)
        v.addLayout(self.feedback)

        p = self.store.profile
        self.native.setCurrentText(p["native"])
        self.target.setCurrentText(p["target"])
        self.level.setCurrentIndex(max(0, self.level.findData(p["level"])))
        self.native.currentTextChanged.connect(lambda t: self._set_profile(native=t))
        self.target.currentTextChanged.connect(lambda t: self._set_profile(target=t))
        self.level.currentIndexChanged.connect(lambda _: self._set_profile(level=self.level.currentData()))

    # -- data -----------------------------------------------------------------------------
    def _generator(self):
        practice = getattr(getattr(self.app.ctx, "assistant", None), "practice", None)
        if practice is not None:
            return practice.generator()
        if self.store.profile.get("target") != "English" and self.app.settings.effective_api_key:
            from jarvis.english.generator import make_generator

            return make_generator(self.app.ctx.llm)
        return None

    def _tasks(self) -> list[Task]:
        return self.store.plan(generator=self._generator())

    def _set_profile(self, **changes) -> None:
        if changes.get("native") and changes["native"] == self.store.profile["target"]:
            pass  # allowed (e.g. polishing your own language), nothing special
        self.store.set_profile(**changes)
        self.reload(tell_tutor=True)

    def _set_focus(self, module: str) -> None:
        self.store.set_profile(focus=module)
        self.reload(tell_tutor=True)

    def reload(self, tell_tutor: bool = False) -> None:
        """Rebuild today's plan (in the background: other languages are written by the AI)."""
        if self._loading:
            return
        self._loading = True
        self._clear_area()
        self.area_layout.addWidget(_label("Preparing today's exercises...", "h2"))

        def work():
            try:
                self._tasks()
            finally:
                self.plan_ready.emit(tell_tutor)

        if self.store.profile.get("target") == "English":
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def _loaded(self, tell_tutor: bool) -> None:
        self._loading = False
        self.current = None
        self.refresh()
        assistant = getattr(self.app.ctx, "assistant", None)
        if tell_tutor and assistant is not None and getattr(assistant, "mode", "") == "practice":
            assistant.restart()  # the tutor gets the new languages / level / plan

    def refresh(self, auto_open: bool = True) -> None:
        s = self.store.summary()
        p = self.store.profile
        self.day_lbl.setText(f"DAY {s['day']}")
        self.streak_lbl.setText(f"🔥 {s['streak']}")
        self.streak_lbl.setToolTip(f"Streak (best {s['best_streak']} days)")
        self.xp_lbl.setText(f"⭐ {s['xp']} XP")
        days = s["practice_days"]
        self.profile_lbl.setText(f"{p['native']} → {p['target']}  ·  {p['level'].title()}  ·  "
                                 f"{days} practice day{'s' if days != 1 else ''}")
        self.today_lbl.setText(f"Today  {s['done']}/{s['total']}")
        self.bar.setMaximum(max(1, s["total"]))
        self.bar.setValue(s["done"])
        self.skills.set_values(s["skills"])
        for mid, b in self.module_btns.items():
            b.setChecked(mid == p["focus"])
        self._fill_tasks()
        if auto_open and self.current is None and not self._loading:
            current = self.data_current()
            if current is not None:
                self.open_task(current)
            else:
                self._welcome()

    def data_current(self) -> Task | None:
        tid = self.store.data.get("current_task")
        task = self.store.task(tid) if tid else None
        if task is not None and task.id not in self.store.done_today():
            return task
        return next((t for t in self._tasks() if t.id not in self.store.done_today()), None)

    def _fill_tasks(self) -> None:
        while self.tasks_row.count():
            w = self.tasks_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        done = self.store.done_today()
        for t in self._tasks():
            r = self.store.result(t.id)
            status = f"✓ {r['score']:g}/{r['total']:g}" if r else "to do"
            short = SHORT.get(t.id.rsplit("-", 1)[-1], t.kind.title())
            b = QPushButton(f"{t.icon}\n{short}\n{status}", objectName="chip")
            b.setCheckable(True)
            b.setChecked(self.current is not None and self.current.id == t.id)
            b.setToolTip(t.title)
            b.setMinimumHeight(64)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if t.id in done:
                b.setStyleSheet(f"QPushButton {{ border: 1px solid {OK}; }}")
            b.clicked.connect(lambda _, t=t: self.open_task(t))
            self.tasks_row.addWidget(b)

    # -- exercise area ----------------------------------------------------------------------
    def _clear_area(self) -> None:
        while self.area_layout.count():
            item = self.area_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())

    def _welcome(self) -> None:
        self._clear_area()
        s = self.store.summary()
        if s["done"] >= s["total"]:
            self.area_layout.addWidget(_label("🎉  All done for today!", "h1"))
            self.area_layout.addWidget(_label("Come back tomorrow for Day {} - or just chat with your tutor to keep "
                                              "practising. Every conversation counts.".format(s["day"] + 1), "muted"))
        else:
            self.area_layout.addWidget(_label("Pick a task above to start.", "h2"))
            self.area_layout.addWidget(_label("Talk to me any time - I'll correct you gently and show tips here.",
                                              "muted"))
        self.area_layout.addStretch(1)

    def open_task(self, task: Task) -> None:
        self.current = task
        self.store.set_current(task.id)
        self._clear_area()
        head = QHBoxLayout()
        head.addWidget(_label(f"{task.icon}  {task.title}", "h1"), 1)
        r = self.store.result(task.id)
        if r:
            head.addWidget(_label(f"Done: {r['score']:g}/{r['total']:g}", "good", wrap=False))
        self.area_layout.addLayout(head)
        if task.payload.get("fallback"):
            self.area_layout.addWidget(_label(f"(Couldn't prepare this in {self.store.profile['target']} right now - "
                                              "practising the English version.)", "muted"))
        if task.kind == "mcq":
            self._quiz(task, task.payload["questions"])
        elif task.kind == "reading":
            passage = _label(task.payload["passage"], "msg")
            passage.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.area_layout.addWidget(passage)
            self.area_layout.addWidget(_label("Read it (out loud if you can), then answer:", "muted"))
            self._quiz(task, task.payload["questions"])
        elif task.kind == "listening":
            self._listening(task)
        else:
            self._speaking(task)
        self.area_layout.addStretch(1)
        self._fill_tasks()

    def _quiz(self, task: Task | None, questions: list[dict], intro: str = "") -> QuizView:
        quiz = QuizView(questions, intro)
        quiz.finished.connect(lambda score, total, missed: self._quiz_done(task, quiz, score, total, missed))
        quiz.answered.connect(lambda ok: self.app.events.emit("practice_answer", correct=ok))
        self.area_layout.addWidget(quiz)
        return quiz

    def _quiz_done(self, task: Task | None, quiz: QuizView, score: int, total: int, missed: list) -> None:
        missed_text = "; ".join(f"'{q['q']}' (answer: {q['options'][q['answer']]})" for q in missed[:3])
        if task is None:  # a quick question from the tutor
            if self.say:
                self.say(f"(system: practice) The learner answered your quiz question "
                         f"{'correctly' if score else 'incorrectly'}. React in one short sentence.")
            quiz.hide()
            self._welcome()
            return
        if self.say:
            self.say(f"(system: practice) The learner finished '{task.title}' with {score}/{total}"
                     + (f". Missed: {missed_text}" if missed else "")
                     + ". React warmly in one or two sentences; if they missed something, explain the main point simply.")
        info = self.store.record(task.id, task.skill, score, total)
        # the window passes this event back to progress(), which shows the result card
        self.app.events.emit("practice_progress", task_id=task.id, score=float(score), total=float(total), **info)

    def _result_card(self, score: float, total: float, info: dict, task: Task) -> None:
        self._clear_area()
        pct = score / total if total else 0
        mood = "🌟 Excellent!" if pct >= 0.9 else "👍 Good job!" if pct >= 0.6 else "💪 Keep going - you'll get there!"
        self.area_layout.addWidget(_label(mood, "h1"))
        self.area_layout.addWidget(_label(f"{task.title}:  {score:g}/{total:g}   ·   +{info['xp_gained']} XP   ·   "
                                          f"🔥 {info['streak']} day streak", "h2"))
        row = QHBoxLayout()
        retry = QPushButton("↻  Try again")
        retry.clicked.connect(lambda: self.open_task(task))
        row.addWidget(retry)
        nxt = next((t for t in self._tasks() if t.id not in self.store.done_today()), None)
        if nxt:
            go = QPushButton(f"Next: {nxt.title}  →", objectName="primary")
            go.clicked.connect(lambda: self.open_task(nxt))
            row.addWidget(go)
        row.addStretch(1)
        self.area_layout.addLayout(row)
        self.area_layout.addStretch(1)

    def _listening(self, task: Task) -> None:
        self.area_layout.addWidget(_label("Listen carefully - your tutor will read it once. Then answer the questions.",
                                          "muted"))
        row = QHBoxLayout()
        play = QPushButton("▶  Play", objectName="primary")
        play.setEnabled(self.say is not None)
        play.clicked.connect(lambda: self.say and self.say(
            "(system: practice listening) Read this aloud exactly, slowly and clearly, then stop and say nothing "
            f"else: {task.payload['script']}"))
        row.addWidget(play)
        transcript = _label(task.payload["script"], "msg")
        transcript.hide()
        show = QPushButton("Show transcript")
        show.clicked.connect(lambda: transcript.setVisible(not transcript.isVisible()))
        row.addWidget(show)
        row.addStretch(1)
        self.area_layout.addLayout(row)
        if self.say is None:
            self.area_layout.addWidget(_label("(Connect Jarvis with your API key to hear it - or read the transcript.)",
                                              "muted"))
        self.area_layout.addWidget(transcript)
        self._quiz(task, task.payload["questions"])

    def _speaking(self, task: Task) -> None:
        icon, title, _ = MODULES.get(task.module, ("🗣", "Speaking", "speaking"))
        card = QFrame(objectName="card")
        cv = QVBoxLayout(card)
        cv.addWidget(_label(f"{icon}  {title.upper()}", "section"))
        cv.addWidget(_label(task.payload["prompt"], "h2"))
        self.area_layout.addWidget(card)
        self.area_layout.addWidget(_label(
            "Press start, then just talk. Your tutor runs the task like a real situation, then shows corrections "
            "here and gives a score out of 10.", "muted"))
        row = QHBoxLayout()
        start = QPushButton("🎙  Start speaking task", objectName="primary")
        start.setEnabled(self.say is not None)
        start.clicked.connect(lambda: self._start_speaking(task, start))
        row.addWidget(start)
        row.addStretch(1)
        self.area_layout.addLayout(row)
        if self.say is None:
            self.area_layout.addWidget(_label("(Needs Jarvis to be connected - add your API key in Settings.)", "muted"))

    def _start_speaking(self, task: Task, button: QPushButton) -> None:
        button.setText("🎙  Listening... talk to your tutor")
        button.setEnabled(False)
        title = MODULES.get(task.module, ("", "Speaking", ""))[1]
        self.say(f"(system: practice task {task.id}) Module: {title}. Task: {task.payload['prompt']} "
                 f"Start now: set up the situation in one or two sentences and let the learner speak. Ask one follow-up, "
                 f"then give feedback, call show_correction for the main mistakes and mark_task_done('{task.id}', score).")

    # -- events from the tutor ---------------------------------------------------------------
    def tutor_mcq(self, question: str, options: list, answer: int, why: str) -> None:
        self.current = None
        self._clear_area()
        self.area_layout.addWidget(_label("⚡  Quick quiz from your tutor", "h1"))
        self._quiz(None, [{"q": question, "options": options, "answer": answer, "why": why}])
        self.area_layout.addStretch(1)

    def correction(self, said: str, better: str, tip: str) -> None:
        card = QFrame(objectName="cardGood")
        h = QVBoxLayout(card)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(2)
        if said:
            wrong = _label(f"✗  {said}", "muted")
            h.addWidget(wrong)
        h.addWidget(_label(f"✓  {better}", "h2"))
        if tip:
            h.addWidget(_label(f"💡 {tip}", "muted"))
        self.feedback.addWidget(card)
        while self.feedback.count() > 2:  # keep the latest two on screen
            old = self.feedback.takeAt(0).widget()
            if old:
                old.deleteLater()

    def progress(self, task_id: str, score: float = 0, total: float = 10, xp_gained: int = 0, streak: int = 0,
                 **_) -> None:
        """A task was saved (by a quiz here, or by the tutor after a speaking task)."""
        task = self.store.task(task_id)
        showing = self.current is not None and self.current.id == task_id
        if showing:
            self.current = None
        self.refresh(auto_open=False)
        if task is not None and (showing or self.current is None):
            self._result_card(score, total, {"xp_gained": xp_gained, "streak": streak}, task)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())
