"""A small contacts book: names with phone numbers, e-mail, how they're related, and birthdays.

Contacts live in Jarvis's memory (keys contact_<name>, email_<name>, birthday_<name>, relation_<name>), so
WhatsApp messages and "what do you know about me?" keep working with them, and backups include them.
"""

from __future__ import annotations

import re
from datetime import date

from jarvis.tools import tool

PREFIX = {"phone": "contact_", "email": "email_", "birthday": "birthday_", "relation": "relation_"}


def _key(name: str) -> str:
    return re.sub(r"[^\w]+", "_", (name or "").strip().lower()).strip("_")


def _phone(raw: str, default_country: str = "91") -> str | None:
    """'+91 98765 43210', '098765 43210', '9876543210' -> '919876543210'."""
    digits = re.sub(r"[^\d]", "", raw or "")
    if (raw or "").strip().startswith("00"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10:
        digits = default_country + digits
    return digits if 11 <= len(digits) <= 15 else None


def _birthday(raw: str) -> str | None:
    """'12 march', '12/03', '1995-03-12' -> 'MM-DD' (the year isn't needed)."""
    from jarvis.dates import parse_day

    raw = (raw or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if m:
        return f"{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    d = parse_day(raw, past=False)
    return f"{d.month:02d}-{d.day:02d}" if d else None


def next_birthday(mm_dd: str, today: date | None = None) -> tuple[date, int]:
    today = today or date.today()
    month, day = (int(x) for x in mm_dd.split("-"))
    for year in (today.year, today.year + 1, today.year + 2):
        try:
            d = date(year, month, day)
        except ValueError:  # 29 February in a normal year: celebrate on the 28th
            d = date(year, month, 28)
        if d >= today:
            return d, (d - today).days
    raise ValueError(mm_dd)


def all_contacts(memory) -> dict[str, dict]:
    people: dict[str, dict] = {}
    for fact in memory.all():
        for field, prefix in PREFIX.items():
            if fact.key.startswith(prefix) and len(fact.key) > len(prefix):
                name = fact.key[len(prefix):]
                people.setdefault(name, {"name": name.replace("_", " ").title()})[field] = fact.value
    return people


@tool(
    "save_contact",
    "Save or update a contact: name with phone number, e-mail, relation (mom, boss, friend) and/or birthday. "
    "Use for 'save mom's number 98765 43210', 'Ali ka birthday 12 march hai'. Only the given fields change.",
    {"name": {"type": "string"}, "phone": {"type": "string"}, "email": {"type": "string"},
     "relation": {"type": "string"}, "birthday": {"type": "string", "description": "e.g. '12 march'"}},
    required=["name"],
)
def save_contact(ctx, name: str, phone: str = "", email: str = "", relation: str = "", birthday: str = ""):
    key = _key(name)
    if not key:
        return {"error": "The contact needs a name."}
    changes: dict[str, str] = {}
    if phone:
        p = _phone(phone)
        if not p:
            return {"error": f"'{phone}' doesn't look like a phone number. Say it with the country code."}
        changes["phone"] = p
    if email:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
            return {"error": f"'{email}' doesn't look like an e-mail address."}
        changes["email"] = email.strip()
    if relation:
        changes["relation"] = relation.strip()[:40]
    if birthday:
        b = _birthday(birthday)
        if not b:
            return {"error": f"I couldn't understand the birthday '{birthday}'. Try '12 March'."}
        changes["birthday"] = b
    if not changes:
        return {"error": "Tell me something to save: a phone number, e-mail, relation or birthday."}
    before = {f: ctx.memory.get(PREFIX[f] + key) for f in changes}
    for field, value in changes.items():
        ctx.memory.remember(PREFIX[field] + key, value, "relationship")

    def revert():
        for field, old in before.items():
            if old is None:
                ctx.memory.forget(PREFIX[field] + key)
            else:
                ctx.memory.remember(old.key, old.value, old.category)

    ctx.undo.push(f"save contact {name}", revert)
    shown = {k: ("+" + v if k == "phone" else v) for k, v in changes.items()}
    return {"result": f"Saved {name.strip().title()}.", "saved": shown}


@tool(
    "find_contact",
    "Look up contacts by name or relation ('mom ka number kya hai?', 'who is my boss?'). Empty query lists all.",
    {"query": {"type": "string"}},
)
def find_contact(ctx, query: str = ""):
    people = all_contacts(ctx.memory)
    q = _key(query)
    rows = [p for k, p in sorted(people.items())
            if not q or q in k or q in _key(p.get("relation", ""))]
    if not rows:
        return f"No contact matches '{query}'." if query else "No contacts saved yet."
    for p in rows:
        if p.get("phone"):
            p["phone"] = "+" + p["phone"]
        if p.get("birthday"):
            d, days = next_birthday(p["birthday"])
            p["birthday"] = d.strftime("%d %B").lstrip("0") + (" (today!)" if days == 0 else f" (in {days} days)")
    return {"contacts": rows[:25], "count": len(rows)}


@tool(
    "delete_contact",
    "Remove a contact and everything saved about them (number, e-mail, relation, birthday). Can be undone.",
    {"name": {"type": "string"}},
    required=["name"],
)
def delete_contact(ctx, name: str):
    key = _key(name)
    facts = [ctx.memory.get(prefix + key) for prefix in PREFIX.values()]
    facts = [f for f in facts if f is not None]
    if not facts:
        return {"error": f"There's no contact called '{name}'."}
    for f in facts:
        ctx.memory.forget(f.key)
    ctx.undo.push(f"delete contact {name}", lambda: [ctx.memory.remember(f.key, f.value, f.category) for f in facts])
    return f"Deleted {name.strip().title()}."


@tool(
    "upcoming_birthdays",
    "Birthdays coming up in the next days ('kiska birthday aane wala hai?'). Offer to set a reminder or send a wish.",
    {"days": {"type": "integer", "description": "how far ahead, default 30"}},
)
def upcoming_birthdays(ctx, days: int = 30):
    days = max(1, min(int(days or 30), 366))
    rows = []
    for p in all_contacts(ctx.memory).values():
        if not p.get("birthday"):
            continue
        d, n = next_birthday(p["birthday"])
        if n <= days:
            rows.append({"name": p["name"], "date": d.strftime("%A %d %B"), "in_days": n,
                         **({"relation": p["relation"]} if p.get("relation") else {})})
    rows.sort(key=lambda r: r["in_days"])
    if not rows:
        return f"No birthdays in the next {days} days."
    return {"birthdays": rows}
