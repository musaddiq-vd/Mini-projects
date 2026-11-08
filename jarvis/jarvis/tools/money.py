"""Expense tracker: "100 ki chai pi" -> saved. Summaries by period and category, monthly budgets.

Stored in ~/.jarvis/expenses.json (listed in Settings > Privacy). Amounts in the user's currency.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import date

from jarvis.dates import parse_day, period_range
from jarvis.datafile import store
from jarvis.tools import tool

CATEGORIES = ["food", "transport", "shopping", "bills", "health", "entertainment", "education", "travel",
              "family", "other"]
ICONS = {"food": "🍔", "transport": "🚕", "shopping": "🛍", "bills": "🧾", "health": "💊", "entertainment": "🎬",
         "education": "📚", "travel": "✈", "family": "👨‍👩‍👧", "other": "💸"}
MAX_AMOUNT = 10_000_000


def expenses():
    return store("expenses", {"items": [], "budgets": {}, "next_id": 1})


def money(ctx, amount: float) -> str:
    cur = getattr(ctx.settings, "currency", "₹") or "₹"
    text = f"{amount:,.2f}".rstrip("0").rstrip(".")
    return f"{cur}{text}"


def _in(items: list[dict], start: date, end: date) -> list[dict]:
    s, e = start.isoformat(), end.isoformat()
    return [i for i in items if s <= i["date"] <= e]


def _month_total(items: list[dict], day: date, category: str | None = None) -> float:
    month = day.isoformat()[:7]
    return sum(i["amount"] for i in items
               if i["date"][:7] == month and (category is None or i["category"] == category))


@tool(
    "add_expense",
    "Record money the user spent: '100 ki chai pi', 'paid 450 for Uber', 'kal 2000 ka grocery'. Pick the closest "
    "category. 'when' is 'today' (default), 'yesterday'/'kal', or a date.",
    {"amount": {"type": "number"},
     "category": {"type": "string", "enum": CATEGORIES},
     "note": {"type": "string", "description": "what it was for, e.g. 'chai', 'Uber to office'"},
     "when": {"type": "string"}},
    required=["amount"],
)
def add_expense(ctx, amount: float, category: str = "other", note: str = "", when: str = ""):
    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        return {"error": "The amount must be a number."}
    if not 0 < amount <= MAX_AMOUNT:
        return {"error": "That amount looks wrong - it must be more than 0."}
    day = parse_day(when, past=True)
    if day is None:
        return {"error": f"Couldn't understand the date '{when}'. Try 'today', 'yesterday' or '12 Nov'."}
    if day > date.today():
        return {"error": "That date is in the future - expenses are for money already spent."}
    category = category if category in CATEGORIES else "other"
    store_ = expenses()

    def change(data):
        eid = data["next_id"]
        data["next_id"] = eid + 1
        data["items"].append({"id": eid, "amount": amount, "category": category, "note": note.strip()[:80],
                              "date": day.isoformat(), "added": time.time()})
        return eid, data["items"], data["budgets"]

    eid, items, budgets = store_.update(change)
    ctx.undo.push(f"expense {eid}", lambda: store_.update(lambda d: d.__setitem__(
        "items", [i for i in d["items"] if i["id"] != eid])))
    month = _month_total(items, day)
    msg = f"Saved {money(ctx, amount)} for {note or category} ({category}). This month so far: {money(ctx, month)}."
    warn = _budget_warning(ctx, items, budgets, day, category)
    return {"result": msg + (f" {warn}" if warn else ""), "id": eid}


def _budget_warning(ctx, items, budgets, day, category) -> str:
    for key, label in ((category, f"{category} budget"), ("total", "monthly budget")):
        limit = budgets.get(key)
        if not limit:
            continue
        spent = _month_total(items, day, None if key == "total" else key)
        if spent >= limit:
            return f"Heads up: the {label} of {money(ctx, limit)} is used up ({money(ctx, spent)})."
        if spent >= 0.8 * limit:
            return f"Note: {spent / limit:.0%} of the {label} ({money(ctx, limit)}) is used."
    return ""


@tool(
    "expense_summary",
    "How much the user spent: total, by category, biggest items, budget left. period: today, yesterday, week, "
    "month (default), last_month, year, all.",
    {"period": {"type": "string", "enum": ["today", "yesterday", "week", "month", "last_month", "year", "all"]},
     "category": {"type": "string", "enum": CATEGORIES}},
)
def expense_summary(ctx, period: str = "month", category: str = ""):
    start, end, label = period_range(period)
    data = expenses().load()
    rows = _in(data["items"], start, end)
    if category:
        rows = [r for r in rows if r["category"] == category]
    if not rows:
        return f"No expenses recorded for {label}" + (f" in {category}." if category else ".")
    by_cat: dict[str, float] = defaultdict(float)
    for r in rows:
        by_cat[r["category"]] += r["amount"]
    total = sum(by_cat.values())
    out = {
        "period": label, "total": money(ctx, total), "count": len(rows),
        "by_category": {k: money(ctx, v) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "biggest": [f"{money(ctx, r['amount'])} {r['note'] or r['category']} ({r['date']})"
                    for r in sorted(rows, key=lambda r: -r["amount"])[:3]],
    }
    if period in ("month", None, "") and data["budgets"].get("total"):
        limit = data["budgets"]["total"]
        out["budget"] = f"{money(ctx, total)} of {money(ctx, limit)} used ({total / limit:.0%})"
    return out


@tool(
    "list_expenses",
    "List individual expenses with their numbers (to check or delete one). period like expense_summary.",
    {"period": {"type": "string", "enum": ["today", "yesterday", "week", "month", "last_month"]}},
)
def list_expenses(ctx, period: str = "week"):
    start, end, label = period_range(period)
    rows = sorted(_in(expenses().load()["items"], start, end), key=lambda r: (r["date"], r["id"]))
    if not rows:
        return f"No expenses for {label}."
    return {"period": label, "expenses": [
        {"id": r["id"], "date": r["date"], "amount": money(ctx, r["amount"]), "category": r["category"],
         "note": r["note"]} for r in rows[-30:]]}


@tool("delete_expense", "Delete an expense by its number (from list_expenses).", {"id": {"type": "integer"}},
      required=["id"])
def delete_expense(ctx, id: int):  # noqa: A002
    eid = int(id)
    store_ = expenses()

    def change(data):
        found = next((i for i in data["items"] if i["id"] == eid), None)
        if found:
            data["items"] = [i for i in data["items"] if i["id"] != eid]
        return found

    found = store_.update(change)
    if not found:
        return {"error": f"No expense #{eid}."}
    ctx.undo.push(f"delete expense {eid}", lambda: store_.update(lambda d: d["items"].append(found)))
    return f"Deleted {money(ctx, found['amount'])} {found['note'] or found['category']} ({found['date']})."


@tool(
    "set_budget",
    "Set a monthly budget, overall or for one category ('mera monthly budget 15000', 'food budget 5000'). "
    "amount 0 removes it.",
    {"amount": {"type": "number"}, "category": {"type": "string", "enum": ["total", *CATEGORIES]}},
    required=["amount"],
)
def set_budget(ctx, amount: float, category: str = "total"):
    amount = float(amount)
    if amount < 0 or amount > MAX_AMOUNT:
        return {"error": "That budget looks wrong."}
    key = category if category in ("total", *CATEGORIES) else "total"

    def change(data):
        if amount == 0:
            data["budgets"].pop(key, None)
        else:
            data["budgets"][key] = amount

    expenses().update(change)
    name = "monthly budget" if key == "total" else f"{key} budget"
    return f"Removed the {name}." if amount == 0 else f"Set the {name} to {money(ctx, amount)} a month."
