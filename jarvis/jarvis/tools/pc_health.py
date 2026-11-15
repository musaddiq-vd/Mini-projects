"""How the PC and the internet are doing: connection check, what is using the CPU and memory, free disk
space, and currency conversion with today's rates."""

from __future__ import annotations

import socket
import time

import psutil
import requests

from jarvis.tools import tool

# well-known, privacy-friendly endpoints; only a TCP connection is opened, nothing is sent
PROBES = [("1.1.1.1", 443, "Cloudflare"), ("8.8.8.8", 443, "Google DNS"), ("9.9.9.9", 443, "Quad9")]
RATES_URL = "https://api.frankfurter.app/latest"  # European Central Bank rates, no key, no tracking
_CURRENCY_WORDS = {
    "rupee": "INR", "rupees": "INR", "rs": "INR", "₹": "INR", "dollar": "USD", "dollars": "USD", "$": "USD",
    "euro": "EUR", "euros": "EUR", "€": "EUR", "pound": "GBP", "pounds": "GBP", "£": "GBP", "yen": "JPY",
    "dirham": "AED", "dirhams": "AED", "riyal": "SAR", "riyals": "SAR",
}


def _tcp_ms(host: str, port: int, timeout: float = 2.0) -> float | None:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return (time.perf_counter() - start) * 1000
    except OSError:
        return None


def _dns_ok(name: str = "www.google.com") -> bool:
    try:
        socket.getaddrinfo(name, 443)
        return True
    except OSError:
        return False


@tool(
    "internet_check",
    "Check whether the internet works and how fast it answers ('is my internet slow?', 'net chal raha hai?'). "
    "Reports the response time and whether websites' names can be found (DNS).",
    slow=True,
)
def internet_check(ctx):
    times = {name: _tcp_ms(host, port) for host, port, name in PROBES}
    ok = [t for t in times.values() if t is not None]
    dns_works = _dns_ok() if ok else False
    if not ok:
        verdict = "offline: no connection to the internet. Check the WiFi or the router."
    elif not dns_works:
        verdict = "connected, but website names don't resolve (DNS problem). Restarting the router often helps."
    else:
        best = min(ok)
        verdict = ("fast" if best < 60 else "okay" if best < 150 else "slow") + f" ({best:.0f} ms)"
    out = {"online": bool(ok), "dns": dns_works, "verdict": verdict,
           "ms": {k: (round(v) if v is not None else None) for k, v in times.items()}}
    try:
        wifi = [n for n, s in psutil.net_if_stats().items() if s.isup and "wi" in n.lower()]
        if wifi:
            out["adapter"] = wifi[0]
    except Exception:  # noqa: BLE001 - adapter names are a nice extra only
        pass
    return out


@tool(
    "top_processes",
    "Which programs are using the most CPU or memory right now ('why is my PC slow?', 'RAM kaun kha raha hai?').",
    {"by": {"type": "string", "enum": ["cpu", "memory"]}, "count": {"type": "integer"}},
)
def top_processes(ctx, by: str = "memory", count: int = 5):
    count = max(1, min(int(count or 5), 15))
    procs = list(psutil.process_iter(["name", "memory_info"]))
    if by == "cpu":
        for p in procs:
            try:
                p.cpu_percent(None)
            except (psutil.Error, OSError):
                pass
        time.sleep(0.5)
    grouped: dict[str, dict] = {}  # Chrome has many processes: add them up by name
    for p in procs:
        try:
            name = (p.info.get("name") or "").removesuffix(".exe") or "unknown"
            if name.lower() in {"system idle process", "idle", "system", "registry", "memory compression"}:
                continue
            row = grouped.setdefault(name, {"name": name, "memory_mb": 0.0, "cpu_percent": 0.0, "processes": 0})
            mem = p.info.get("memory_info")
            row["memory_mb"] += (mem.rss if mem else 0) / 1024 / 1024
            row["processes"] += 1
            if by == "cpu":
                row["cpu_percent"] += p.cpu_percent(None) / (psutil.cpu_count() or 1)
        except (psutil.Error, OSError):
            continue
    key = "cpu_percent" if by == "cpu" else "memory_mb"
    rows = sorted(grouped.values(), key=lambda r: r[key], reverse=True)[:count]
    for r in rows:
        r["memory_mb"] = round(r["memory_mb"])
        r["cpu_percent"] = round(r["cpu_percent"], 1)
    vm = psutil.virtual_memory()
    return {"by": by, "top": rows, "ram_used_percent": vm.percent,
            "ram_free_gb": round(vm.available / 1024 ** 3, 1),
            "tip": "To close one, the user can say 'close <name>'."}


@tool(
    "disk_space",
    "Free and used space on every drive ('C drive me kitni jagah hai?', 'is my disk full?').",
)
def disk_space(ctx):
    drives = []
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            u = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        drives.append({"drive": part.mountpoint.rstrip("\\"), "free_gb": round(u.free / 1024 ** 3, 1),
                       "total_gb": round(u.total / 1024 ** 3, 1), "used_percent": u.percent})
    low = [d["drive"] for d in drives if d["used_percent"] >= 90]
    out = {"drives": drives}
    if low:
        out["warning"] = (f"{', '.join(low)} almost full. Emptying the Recycle Bin, Downloads and running "
                          "Windows 'Disk Cleanup' usually frees space.")
    return out


def _code(text: str) -> str:
    t = (text or "").strip()
    return _CURRENCY_WORDS.get(t.lower(), t.upper())


@tool(
    "convert_currency",
    "Convert money between currencies with today's exchange rates (European Central Bank), e.g. 100 USD to INR. "
    "Information only: never buy, sell or transfer money.",
    {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "e.g. USD, INR, EUR"},
     "to_currency": {"type": "string"}},
    required=["amount", "from_currency", "to_currency"],
    slow=True,
)
def convert_currency(ctx, amount: float, from_currency: str, to_currency: str):
    a, b = _code(from_currency), _code(to_currency)
    if a == b:
        return {"result": f"{amount:,.2f} {a}"}
    try:
        r = requests.get(RATES_URL, params={"amount": float(amount), "from": a, "to": b}, timeout=10)
        data = r.json()
    except Exception:  # noqa: BLE001
        return {"error": "Couldn't get exchange rates right now (no internet?)."}
    value = (data.get("rates") or {}).get(b)
    if value is None:
        return {"error": f"I don't have a rate for {a} to {b}."}
    return {"result": f"{float(amount):,.2f} {a} = {value:,.2f} {b}", "value": value, "date": data.get("date"),
            "note": "Banks and cards charge a little more than this rate."}
