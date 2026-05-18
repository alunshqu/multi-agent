import base64
import io
import pyautogui
from PIL import ImageGrab

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.3


def get_screen_size() -> tuple[int, int]:
    return pyautogui.size()


def _screenshot_b64() -> str:
    img = ImageGrab.grab()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def execute_computer_action(action: str, **kwargs) -> dict:
    try:
        if action == "screenshot":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _screenshot_b64(),
                },
            }

        elif action == "left_click":
            x, y = kwargs["coordinate"]
            pyautogui.click(x, y)
            return {"type": "text", "text": f"Clicked ({x}, {y})"}

        elif action == "right_click":
            x, y = kwargs["coordinate"]
            pyautogui.rightClick(x, y)
            return {"type": "text", "text": f"Right-clicked ({x}, {y})"}

        elif action == "double_click":
            x, y = kwargs["coordinate"]
            pyautogui.doubleClick(x, y)
            return {"type": "text", "text": f"Double-clicked ({x}, {y})"}

        elif action == "middle_click":
            x, y = kwargs["coordinate"]
            pyautogui.middleClick(x, y)
            return {"type": "text", "text": f"Middle-clicked ({x}, {y})"}

        elif action == "mouse_move":
            x, y = kwargs["coordinate"]
            pyautogui.moveTo(x, y, duration=0.2)
            return {"type": "text", "text": f"Moved to ({x}, {y})"}

        elif action == "left_click_drag":
            x, y = kwargs["coordinate"]
            if "start_coordinate" in kwargs:
                sx, sy = kwargs["start_coordinate"]
                pyautogui.moveTo(sx, sy)
            pyautogui.dragTo(x, y, duration=0.4, button="left")
            return {"type": "text", "text": f"Dragged to ({x}, {y})"}

        elif action == "type":
            text = kwargs["text"]
            pyautogui.write(text, interval=0.03)
            return {"type": "text", "text": f"Typed text ({len(text)} chars)"}

        elif action == "key":
            key = kwargs["text"]
            parts = key.lower().split("+")
            if len(parts) > 1:
                pyautogui.hotkey(*parts)
            else:
                pyautogui.press(parts[0])
            return {"type": "text", "text": f"Pressed key: {key}"}

        elif action == "hold_key":
            key = kwargs["text"]
            duration = kwargs.get("duration", 0.5)
            pyautogui.keyDown(key)
            import time; time.sleep(duration)
            pyautogui.keyUp(key)
            return {"type": "text", "text": f"Held key {key} for {duration}s"}

        elif action == "scroll":
            x, y = kwargs["coordinate"]
            direction = kwargs.get("direction", "down")
            amount = int(kwargs.get("amount", 3))
            clicks = -amount if direction == "down" else amount
            pyautogui.scroll(clicks, x=x, y=y)
            return {"type": "text", "text": f"Scrolled {direction} at ({x}, {y})"}

        elif action == "cursor_position":
            x, y = pyautogui.position()
            return {"type": "text", "text": f"Cursor at ({x}, {y})"}

        elif action == "wait":
            import time
            duration = kwargs.get("duration", 1)
            time.sleep(duration)
            return {"type": "text", "text": f"Waited {duration}s"}

        else:
            return {"type": "text", "text": f"Unknown action: {action}"}

    except Exception as e:
        return {"type": "text", "text": f"Error in {action}: {e}"}
