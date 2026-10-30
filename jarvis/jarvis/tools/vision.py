"""Let Jarvis see the screen or the webcam. The image goes to the model with the tool reply."""

from __future__ import annotations

import io

from jarvis.tools import tool

MAX_SIDE = 1280


def _to_jpeg(image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def screenshot_jpeg(monitor: int = 1) -> bytes:
    import mss
    from PIL import Image

    with mss.mss() as sct:
        monitors = sct.monitors
        shot = sct.grab(monitors[monitor if monitor < len(monitors) else 1])
        return _to_jpeg(Image.frombytes("RGB", shot.size, shot.rgb))


@tool(
    "look_at_screen",
    "Take a screenshot so you can see what is on the user's screen (to read, explain or help with it).",
    {"monitor": {"type": "integer", "description": "1 = main screen (default)"}},
    slow=True,
)
def look_at_screen(ctx, monitor: int = 1):
    if ctx.attach_image is None:
        return {"error": "Seeing images isn't available in this mode."}
    ctx.attach_image(screenshot_jpeg(int(monitor or 1)))
    return "Screenshot attached. Describe or answer based on it."


@tool(
    "look_at_camera",
    "Take a photo with the webcam so you can see the user or an object they are holding up.",
    slow=True,
)
def look_at_camera(ctx):
    if ctx.attach_image is None:
        return {"error": "Seeing images isn't available in this mode."}
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
    ctx.attach_image(_to_jpeg(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    return "Photo attached. Describe or answer based on it."
