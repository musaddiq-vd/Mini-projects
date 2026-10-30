"""Things Jarvis does on its own: morning briefing, check-ins, hardware alerts,
watched news topics, and a summary of each conversation for next time."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import date, datetime

import psutil

from jarvis.app import Jarvis

HW_CHECK_SECONDS = 30
TOPIC_CHECK_SECONDS = 3 * 60 * 60
ALERT_COOLDOWN = 30 * 60


class BackgroundServices:
    def __init__(self, app: Jarvis, say: Callable[[str], None]):
        """`say` sends a "(system) ..." instruction to the live model so it speaks naturally."""
        self.app = app
        self.say = say
        self.s = app.settings
        self.transcript: list[tuple[str, str]] = []
        self.last_activity = time.monotonic()
        self.sleeping = False
        self._stop = threading.Event()
        self._last_alert: dict[str, float] = {}
        app.events.subscribe(self._on_event)

    # -- events --------------------------------------------------------------
    def _on_event(self, kind: str, data: dict) -> None:
        if kind in {"user", "assistant"}:
            self.transcript.append((kind, data["text"]))
            self.transcript = self.transcript[-200:]
            self.last_activity = time.monotonic()
        elif kind == "state":
            self.sleeping = data["state"] == "sleeping"

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        for target in (self._hardware_loop, self._topic_loop, self._proactive_loop):
            threading.Thread(target=target, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def on_connected(self) -> None:
        """Called once the first live session is up."""
        if self.s.morning_briefing and self.app.state.get("last_briefing") != date.today().isoformat():
            self.app.state.set("last_briefing", date.today().isoformat())
            self.say(self.briefing_instruction())

    # -- briefing ------------------------------------------------------------
    def _city(self) -> str:
        if self.s.city:
            return self.s.city
        fact = self.app.memory.get("city") or self.app.memory.get("home_city")
        return fact.value if fact else ""

    def briefing_instruction(self) -> str:
        part_of_day = "morning" if datetime.now().hour < 12 else "afternoon" if datetime.now().hour < 17 else "evening"
        city = self._city()
        steps = [f"greet the user warmly for the {part_of_day}", "say the time"]
        steps.append(f"call get_weather for {city} and give it in one line" if city
                     else "ask which city they live in so you can give weather next time")
        steps.append("call web_search with mode news for today's top headlines and mention two briefly")
        last = self.app.state.get("last_session_summary")
        if last:
            steps.append(f"mention naturally what you talked about last time: {last}")
        return ("(system: first start today) Give a short briefing: " + "; ".join(steps)
                + ". Keep it under six short sentences, in the user's language.")

    # -- loops ---------------------------------------------------------------
    def _alert(self, key: str, text: str) -> None:
        now = time.monotonic()
        if now - self._last_alert.get(key, -ALERT_COOLDOWN) >= ALERT_COOLDOWN:
            self._last_alert[key] = now
            self.say(f"(system: hardware alert) Briefly tell the user: {text}")

    def _hardware_loop(self) -> None:
        high_cpu_since = None
        while not self._stop.wait(HW_CHECK_SECONDS):
            if not self.s.hardware_alerts:
                continue
            cpu = psutil.cpu_percent(interval=1)
            high_cpu_since = (high_cpu_since or time.monotonic()) if cpu > 90 else None
            if high_cpu_since and time.monotonic() - high_cpu_since > 180:
                self._alert("cpu", f"CPU has been above 90% for 3 minutes (now {cpu:.0f}%).")
            ram = psutil.virtual_memory().percent
            if ram > 92:
                self._alert("ram", f"memory is almost full ({ram:.0f}% used).")
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged and battery.percent <= 20:
                self._alert("battery", f"battery is at {battery.percent:.0f}%, plug in the charger.")

    def _topic_loop(self) -> None:
        if self._stop.wait(120):  # let startup settle
            return
        while True:
            try:
                self.check_topics()
            except Exception as exc:
                self.app.events.emit("log", text=f"[topics] {exc}")
            if self._stop.wait(TOPIC_CHECK_SECONDS):
                return

    def check_topics(self, search=None) -> list[str]:
        topics = self.app.state.get("watch_topics", [])
        if not topics:
            return []
        if search is None:
            from ddgs import DDGS

            def search(q):
                return DDGS().news(q, max_results=5, timelimit="d") or []

        seen: dict[str, list[str]] = self.app.state.get("seen_headlines", {})
        fresh = []
        for topic in topics:
            old = set(seen.get(topic, []))
            titles = [r.get("title", "") for r in search(topic) if r.get("title")]
            new = [t for t in titles if t not in old]
            seen[topic] = (list(old) + new)[-50:]
            if new and old:  # the first check only records what exists
                fresh.append(f"{topic}: {new[0]}")
        self.app.state.set("seen_headlines", seen)
        if fresh:
            self.say("(system: watched topics) Tell the user about these new headlines briefly: " + " | ".join(fresh))
        return fresh

    def _proactive_loop(self) -> None:
        while not self._stop.wait(60):
            if not self.s.proactive or self.sleeping:
                continue
            idle = time.monotonic() - self.last_activity
            if idle >= self.s.proactive_minutes * 60 and 9 <= datetime.now().hour < 22:
                self.last_activity = time.monotonic()
                self.say("(system: check-in) The user has been quiet for a while. Make ONE short, friendly, "
                         "useful check-in based on the time of day and what you know about them "
                         "(their projects, plans, reminders). Don't repeat earlier check-ins.")

    # -- session summary -------------------------------------------------------
    def save_summary(self) -> str | None:
        turns = [(k, t) for k, t in self.transcript if not t.startswith("(system")]
        if sum(1 for k, _ in turns if k == "user") < 2:
            return None
        text = "\n".join(f"{'User' if k == 'user' else 'Assistant'}: {t}" for k, t in turns[-60:])
        try:
            summary = self.app.ctx.llm.ask(
                "Summarise this conversation in one or two short sentences, written as a note to remember "
                "(topics, decisions, plans). No greeting.\n\n" + text
            )
        except Exception as exc:
            self.app.events.emit("log", text=f"[summary] {exc}")
            return None
        if summary:
            self.app.state.set("last_session_summary", f"{summary} ({date.today():%d %b})")
        return summary
