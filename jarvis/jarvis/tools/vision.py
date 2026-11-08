"""Let Jarvis see the screen or the webcam. The image goes to the model with the tool reply."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from jarvis.tools import tool

MAX_SIDE = 1280
SCREENSHOT_DIR = Path.home() / "Pictures" / "Jarvis Screenshots"


def _to_jpeg(image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def grab_screen(monitor: int = 1):
    """Full-size PIL image of one monitor (1 = main)."""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        monitors = sct.monitors
        shot = sct.grab(monitors[monitor if 0 < monitor < len(monitors) else 1])
        return Image.frombytes("RGB", shot.size, shot.rgb)


def screenshot_jpeg(monitor: int = 1) -> bytes:
    return _to_jpeg(grab_screen(monitor))


def _seen_path(ctx) -> Path | None:
    """Where the last image Jarvis looked at is kept, so the window can show it."""
    from jarvis.config import home_dir

    try:
        folder = home_dir() / "cache"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"seen_{datetime.now():%H%M%S_%f}.jpg"
    except OSError:
        return None


KEEP_SEEN = 5


def clear_seen_cache() -> int:
    """Delete the copies of what Jarvis looked at (done when the window closes). Returns how many."""
    from jarvis.config import home_dir

    count = 0
    for f in (home_dir() / "cache").glob("seen_*.jpg"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count


def _captured(ctx, what: str) -> None:
    """Tell the window so it can flash and show a notice: you always know when Jarvis looks."""
    if ctx.events is not None:
        ctx.events.emit("capture", what=what)


def _keep_seen(ctx, jpeg: bytes) -> str | None:
    path = _seen_path(ctx)
    if path is None:
        return None
    try:
        for old in sorted(path.parent.glob("seen_*.jpg"))[:-KEEP_SEEN + 1 or None]:  # keep only a few
            old.unlink(missing_ok=True)
        path.write_bytes(jpeg)
        return str(path)
    except OSError:
        return None


@tool(
    "take_screenshot",
    "SAVE a screenshot as a PNG file (Pictures/Jarvis Screenshots) and SHOW it to the user: it appears "
    "in the Jarvis chat and, unless open is false, opens in the photo viewer. Use for 'screenshot lo', "
    "'take/save a screenshot', 'screenshot dikhao/kholo'. You also get to see the image.",
    {"monitor": {"type": "integer", "description": "1 = main screen (default)"},
     "open": {"type": "boolean", "description": "open it in the photo viewer (default true)"}},
)
def take_screenshot(ctx, monitor: int = 1, open: bool = True):  # noqa: A002 - tool argument name
    from jarvis.tools.files import open_path

    image = grab_screen(int(monitor or 1))
    _captured(ctx, "screen")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"Screenshot {datetime.now():%Y-%m-%d %H-%M-%S}.png"
    image.save(path)
    if open:
        open_path(path)
    if ctx.attach_image is not None:
        try:
            ctx.attach_image(_to_jpeg(image))
        except Exception:
            pass  # seeing it is a bonus; the file is what was asked for
    where = "Pictures > Jarvis Screenshots"
    return {"result": f"Screenshot saved in {where}" + (" and opened in the photo viewer" if open else "")
            + ". It is also shown in the Jarvis window. Tell the user in a few words; don't read the path.",
            "image": str(path), "path": str(path)}


@tool(
    "look_at_screen",
    "Take a screenshot so you can see what is on the user's screen (to read, explain or help with it).",
    {"monitor": {"type": "integer", "description": "1 = main screen (default)"}},
    slow=True,
)
def look_at_screen(ctx, monitor: int = 1):
    if ctx.attach_image is None:
        return {"error": "Seeing images isn't available in this mode."}
    jpeg = screenshot_jpeg(int(monitor or 1))
    _captured(ctx, "screen")
    ctx.attach_image(jpeg)
    out = {"result": "Screenshot attached. Describe or answer based on it. (To save or open it for the "
                     "user, call take_screenshot.)"}
    seen = _keep_seen(ctx, jpeg)
    if seen:
        out["image"] = seen
    return out


@tool(
    "look_at_camera",
    "Take a photo with the webcam so you can see the user or an object they are holding up.",
    slow=True,
)
def look_at_camera(ctx):
    if ctx.attach_image is None:
        return {"error": "Seeing images isn't available in this mode."}
    if getattr(ctx.settings, "ask_camera", True) and not ctx.confirm.confirm("Let Jarvis take a photo with your camera?"):
        return "The user didn't allow the camera. Respect that and don't ask again right away."
    try:
        import cv2
    except ImportError:
        return {"error": "Camera needs: pip install opencv-python"}
    from PIL import Image

    cam = cv2.VideoCapture(0)
    try:
        for _ in range(5):  # let the camera adjust exposure
            ok, frame = cam.read()
        if not ok:
            return {"error": "Couldn't read from the camera."}
    finally:
        cam.release()
    jpeg = _to_jpeg(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    _captured(ctx, "camera")
    ctx.attach_image(jpeg)
    out = {"result": "Photo attached. Describe or answer based on it."}
    seen = _keep_seen(ctx, jpeg)
    if seen:
        out["image"] = seen
    return out
