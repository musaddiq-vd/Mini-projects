"""The practice-mode brain: the tutor's system prompt and its own small tool set.

In practice mode the live session gets ONLY these tools, so Jarvis's computer commands (open apps,
shutdown, messages...) can't run until the user switches back to agent mode with the button.

Events for the window:
    practice_mcq         question, options, answer, why        (a quiz question on screen)
    practice_correction  said, better, tip                     (a correction card)
    practice_progress    task_id, score, total, xp_gained, streak
"""

from __future__ import annotations

from datetime import date

from jarvis.english.curriculum import MODULES
from jarvis.english.store import EnglishStore
from jarvis.tools import Tool, ToolRegistry


class Tutor:
    def __init__(self, app, store: EnglishStore | None = None):
        self.app = app
        self.store = store or app.english
        self.registry = self._build_registry()

    # -- prompt -------------------------------------------------------------------------
    def prompt(self) -> str:
        p = self.store.profile
        s = self.app.settings
        tasks = self.store.plan(generator=self.generator())
        done = self.store.done_today()
        lines = "\n".join(f"  - {t.id}: {t.title} [{'done' if t.id in done else 'to do'}]" for t in tasks)
        name = s.user_name or "the learner"
        summary = self.store.summary()
        focus = MODULES.get(p["focus"], MODULES["conversation"])[1]
        return f"""You are {s.assistant_name}, now in LANGUAGE PRACTICE MODE: a warm, patient, encouraging {p['target']} speaking coach for {name}.
Created by Musaddiq Khan.

The learner:
- Own language: {p['native']}. Learning: {p['target']}. Level: {p['level']}. Focus module: {focus}.
- Practice day {summary['day']}, streak {summary['streak']} day(s), {summary['xp']} XP so far.

Today's tasks (ids are only for your tools - never say an id like "d1-speak" out loud):
{lines}

How to coach (you are on a live voice call):
- Speak mostly in {p['target']}, using words and speed that suit a {p['level']} learner. When something is hard,
  explain it briefly in {p['native']}, then switch back.
- Be human and kind: praise real progress, keep energy up, never make the learner feel bad.
- Correct gently. After the learner speaks, pick the one or two most useful mistakes (grammar, word choice,
  word order, unnatural phrasing) and call show_correction(said, better, tip) for each - the tip in
  {p['native']} if they are a beginner. Then say the better sentence once and let them repeat it.
- Don't correct every tiny slip in casual chat; focus on what helps most.
- SPEAKING TASKS: when a "(system: practice task ...)" message arrives, run it like a real situation (interview,
  role-play, daily routine, pronunciation). Ask, listen, ask one follow-up. Then give feedback: two things they did
  well and one or two to improve, call show_correction for the main mistakes, and finally call
  mark_task_done(task_id, score) with a fair score from 0 to 10. Say the score in words ("eight out of ten").
- LISTENING: when asked to read a script aloud, read exactly that text, slowly and clearly, with natural
  intonation, and say nothing else.
- When it helps (e.g. a new word came up), you may give a quick quiz question with show_mcq.
- When the learner asks what to do, use today_plan and suggest the next unfinished task.
- You CANNOT control the computer in this mode (no apps, files, messages, settings). If asked, say practice
  mode is on and they can press "Back to Jarvis" to use commands again.
- Messages starting with "(system" come from the app, not the learner: follow them naturally.

Today: {date.today():%A %d %B %Y}.
"""

    # -- tools ----------------------------------------------------------------------------
    def _emit(self, kind: str, **data) -> None:
        self.app.events.emit(kind, **data)

    def _build_registry(self) -> ToolRegistry:
        reg = ToolRegistry()

        def show_mcq(ctx, question: str, options: list, answer: int, why: str = ""):
            options = [str(o) for o in (options or [])][:4]
            if len(options) < 2 or not 0 <= int(answer) < len(options):
                return {"error": "Give 2-4 options and the index of the right one."}
            self._emit("practice_mcq", question=question, options=options, answer=int(answer), why=why)
            return "The question is on the learner's screen. Wait for their answer (they tap it or say it)."

        def show_correction(ctx, said: str, better: str, tip: str = ""):
            if not better.strip():
                return {"error": "Give the better sentence."}
            self.store.add_correction(said, better, tip)
            self._emit("practice_correction", said=said, better=better, tip=tip)
            return "Shown on screen. Say the better sentence once and invite them to repeat it."

        def mark_task_done(ctx, task_id: str, score: float, total: float = 10):
            task = self.store.task(task_id)
            if task is None:
                ids = ", ".join(t.id for t in self.store.plan())
                return {"error": f"Unknown task '{task_id}'. Today's ids: {ids}"}
            info = self.store.record(task_id, task.skill, score, total)
            self._emit("practice_progress", task_id=task_id, score=float(score), total=float(total), **info)
            left = [t.title for t in self.store.plan() if t.id not in self.store.done_today()]
            nxt = f" Next suggestion: {left[0]}." if left else " All of today's tasks are done - celebrate!"
            return f"Saved: {score}/{total}, +{info['xp_gained']} XP, streak {info['streak']}.{nxt}"

        def today_plan(ctx):
            done = self.store.done_today()
            return {"day": self.store.day_number(),
                    "tasks": [{"id": t.id, "title": t.title, "kind": t.kind, "done": t.id in done,
                               **({"prompt": t.payload.get("prompt")} if t.kind == "speaking" else {})}
                              for t in self.store.plan()]}

        reg.add(Tool("show_mcq", "Show a multiple-choice question on the learner's screen.", show_mcq,
                     {"question": {"type": "string"},
                      "options": {"type": "array", "items": {"type": "string"}},
                      "answer": {"type": "integer", "description": "index of the correct option, from 0"},
                      "why": {"type": "string", "description": "short explanation shown after answering"}},
                     ["question", "options", "answer"]))
        reg.add(Tool("show_correction", "Show a correction card: what the learner said, a better version, a tip.",
                     show_correction,
                     {"said": {"type": "string"}, "better": {"type": "string"}, "tip": {"type": "string"}},
                     ["said", "better"]))
        reg.add(Tool("mark_task_done", "Save the score of a finished practice task (speaking tasks: 0-10).",
                     mark_task_done,
                     {"task_id": {"type": "string"}, "score": {"type": "number"}, "total": {"type": "number"}},
                     ["task_id", "score"]))
        reg.add(Tool("today_plan", "Today's practice tasks and which are done.", today_plan))
        return reg

    def generator(self):
        if self.store.profile.get("target") == "English":
            return None
        from jarvis.english.generator import make_generator

        try:
            return make_generator(self.app.ctx.llm, log=lambda t: self.app.events.emit("log", text=t))
        except Exception:
            return None
