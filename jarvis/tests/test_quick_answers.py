"""Calculator, units, world clock, countdowns, random picks, stopwatch and PC health tools."""

from datetime import date, timedelta

import pytest

from jarvis.tools import pc_health, utilities
from jarvis.tools.utilities import calc, convert


@pytest.mark.parametrize("expr, want", [
    ("18% of 2500", 450), ("2500 ka 18%", 450), ("(1200+450)/3", 550), ("sqrt(144) + 2^3", 20),
    ("5 times 7", 35), ("2 lakh / 4", 50000), ("1,250 + 750", 2000), ("what is 10 minus 3?", 7),
])
def test_calculator(expr, want):
    assert calc(expr) == pytest.approx(want)


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "9**9**9", "factorial(10000)", "a + 1"])
def test_calculator_refuses_unsafe_or_huge(registry, ctx, expr):
    assert "error" in registry.call("calculate", {"expression": expr}, ctx)


def test_calculator_tool_and_divide_by_zero(registry, ctx):
    assert registry.call("calculate", {"expression": "1850 / 4"}, ctx)["result"] == "462.5"
    assert "zero" in registry.call("calculate", {"expression": "5/0"}, ctx)["error"]


def test_units():
    assert convert(100, "km", "miles") == pytest.approx(62.137, rel=1e-4)
    assert convert(98.6, "°F", "C") == pytest.approx(37)
    assert convert(5, "GB", "MB") == 5120
    assert convert(2, "hours", "minutes") == 120
    with pytest.raises(ValueError, match="Can't turn"):
        convert(1, "kg", "km")
    with pytest.raises(ValueError, match="don't know"):
        convert(1, "parsec", "km")


def test_world_time(registry, ctx):
    out = registry.call("world_time", {"place": "London"}, ctx)
    assert out["zone"] == "Europe/London" and out["time"]
    assert utilities.find_zone("Tokyo") == "Asia/Tokyo" and utilities.find_zone("Dubai") == "Asia/Dubai"
    assert "error" in registry.call("world_time", {"place": "Narnia"}, ctx)


def test_days_until(registry, ctx):
    soon = date.today() + timedelta(days=20)
    out = registry.call("days_until", {"when": soon.isoformat(), "what": "my exam"}, ctx)
    assert out["days"] == 20 and "20 days until my exam" in out["result"]
    assert "today" in registry.call("days_until", {"when": "today"}, ctx)["result"]
    assert "error" in registry.call("days_until", {"when": "someday maybe"}, ctx)


def test_random_pick(registry, ctx):
    assert registry.call("random_pick", {"kind": "coin"}, ctx)["result"] in {"Heads", "Tails"}
    dice = registry.call("random_pick", {"kind": "dice", "count": 3}, ctx)
    assert len(dice["rolls"]) == 3 and all(1 <= r <= 6 for r in dice["rolls"])
    assert registry.call("random_pick", {"kind": "choose", "options": ["pizza", "biryani"]}, ctx)["result"] in {
        "pizza", "biryani"}
    assert "error" in registry.call("random_pick", {"kind": "choose", "options": ["only one"]}, ctx)
    n = registry.call("random_pick", {"kind": "number", "low": 10, "high": 1}, ctx)["result"]
    assert 1 <= n <= 10


def test_stopwatch(registry, ctx, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(utilities, "_clock", lambda: clock["t"])
    assert "error" in registry.call("stopwatch", {"action": "status"}, ctx)
    registry.call("stopwatch", {"action": "start"}, ctx)
    clock["t"] += 65
    assert registry.call("stopwatch", {"action": "lap"}, ctx)["result"] == "Lap 1: 1 minute 5 seconds."
    clock["t"] += 3600
    out = registry.call("stopwatch", {"action": "stop"}, ctx)
    assert out["result"] == "Stopped at 1 hour 1 minute 5 seconds." and out["laps"] == [65.0]


def test_internet_check(registry, ctx, monkeypatch):
    monkeypatch.setattr(pc_health, "_tcp_ms", lambda host, port, timeout=2.0: 25.0)
    monkeypatch.setattr(pc_health, "_dns_ok", lambda name="": True)
    out = registry.call("internet_check", {}, ctx)
    assert out["online"] and out["verdict"].startswith("fast")
    monkeypatch.setattr(pc_health, "_dns_ok", lambda name="": False)
    assert "DNS" in registry.call("internet_check", {}, ctx)["verdict"]
    monkeypatch.setattr(pc_health, "_tcp_ms", lambda host, port, timeout=2.0: None)
    out = registry.call("internet_check", {}, ctx)
    assert not out["online"] and out["verdict"].startswith("offline")


def test_top_processes_and_disk_space(registry, ctx):
    out = registry.call("top_processes", {"by": "memory", "count": 3}, ctx)
    assert len(out["top"]) <= 3 and "ram_used_percent" in out
    assert registry.call("disk_space", {}, ctx)["drives"]


def test_currency(registry, ctx, monkeypatch):
    class Resp:
        def json(self):
            return {"amount": 100, "base": "USD", "date": "2026-11-13", "rates": {"INR": 8450.5}}

    seen = {}

    def get(url, params=None, timeout=10):
        seen.update(params)
        return Resp()

    monkeypatch.setattr(pc_health.requests, "get", get)
    out = registry.call("convert_currency", {"amount": 100, "from_currency": "dollars", "to_currency": "rupees"}, ctx)
    assert seen["from"] == "USD" and seen["to"] == "INR"
    assert out["result"] == "100.00 USD = 8,450.50 INR"


def test_helpers_are_not_tool_arguments(registry):
    for name in ("internet_check", "convert_currency", "stopwatch"):
        assert set(registry.tools[name].parameters) <= {"action", "amount", "from_currency", "to_currency"}
