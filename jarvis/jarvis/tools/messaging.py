"""Send WhatsApp messages and draft emails. Every send is confirmed by the user on screen."""

from __future__ import annotations

import os
import platform
import re
import time
import urllib.parse
import webbrowser

from jarvis.tools import tool

OS = platform.system()


def _resolve_phone(ctx, contact: str) -> str | None:
    """A phone number as digits with country code, from the text itself or from memory."""
    digits = re.sub(r"[^\d]", "", contact)
    if len(digits) >= 10:
        return digits
    fact = ctx.memory.get(f"contact_{contact}") or ctx.memory.get(contact)
    if not fact:
        hits = ctx.memory.recall(f"contact {contact}", limit=1)
        fact = hits[0] if hits and "contact" in hits[0].key else None
    if fact:
        digits = re.sub(r"[^\d]", "", fact.value)
        return digits if len(digits) >= 10 else None
    return None


@tool(
    "send_whatsapp",
    "Send a WhatsApp message. 'contact' is a saved contact name (memory key contact_<name>) or a phone "
    "number with country code. The user confirms on screen before it is sent.",
    {"contact": {"type": "string"}, "message": {"type": "string"}},
    required=["contact", "message"],
)
def send_whatsapp(ctx, contact: str, message: str):
    phone = _resolve_phone(ctx, contact)
    if not phone:
        return {"error": f"I don't have a number for '{contact}'. Ask for it (with country code) "
                         f"and save it with remember as contact_{contact.lower().replace(' ', '_')}."}
    if not ctx.confirm.confirm(f"Send WhatsApp to {contact} (+{phone}):\n\"{message}\""):
        return "Cancelled - the user did not confirm."
    query = urllib.parse.urlencode({"phone": phone, "text": message}, quote_via=urllib.parse.quote)
    if OS == "Windows":
        try:
            os.startfile(f"whatsapp://send?{query}")  # noqa: S606 - WhatsApp desktop URI
        except OSError:
            webbrowser.open(f"https://web.whatsapp.com/send?{query}")
    else:
        webbrowser.open(f"https://web.whatsapp.com/send?{query}")
    return _press_send_when_ready()


def _press_send_when_ready() -> str:
    if OS != "Windows":
        return "Opened the chat with the message typed in; press Enter to send."
    import pyautogui

    from jarvis.tools import _win

    for _ in range(20):  # wait up to ~10 s for WhatsApp to come to the front
        time.sleep(0.5)
        fg = _win.foreground()
        if fg and "whatsapp" in (fg.title + fg.process).lower():
            time.sleep(1.5)  # let the chat load the draft
            pyautogui.press("enter")
            return "Message sent."
    return "Opened the chat with the message typed in, but couldn't press send - please press Enter."


@tool(
    "draft_email",
    "Open a new email in the default mail app with recipient, subject and body filled in (the user presses send).",
    {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
    required=["to", "subject", "body"],
)
def draft_email(ctx, to: str, subject: str, body: str):
    if not re.fullmatch(r"[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+", to.strip()):
        fact = ctx.memory.get(f"email_{to}") or ctx.memory.get(f"contact_{to}")
        if fact and "@" in fact.value:
            to = fact.value.strip()
        else:
            return {"error": f"I need an email address for '{to}'."}
    params = urllib.parse.urlencode({"subject": subject, "body": body}, quote_via=urllib.parse.quote)
    webbrowser.open(f"mailto:{to.strip()}?{params}")
    return "Email draft opened - review it and press send."
