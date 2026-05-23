import threading
import time
from typing import Optional

import pyperclip
from pynput.keyboard import Controller, Key

_keyboard = Controller()
_PASTE_SETTLE_MS = 0.05  # short pause between copy and Cmd+V to let clipboard settle


class ClipboardPaster:
    def paste(
        self,
        text: str,
        restore: bool = True,
        restore_delay: float = 1.5,
    ) -> None:
        previous: Optional[str] = None
        if restore:
            try:
                previous = pyperclip.paste()
            except Exception:
                previous = None

        try:
            pyperclip.copy(text)
        except Exception as e:
            raise RuntimeError(f"Failed to write to clipboard: {e}") from e

        time.sleep(_PASTE_SETTLE_MS)

        with _keyboard.pressed(Key.cmd):
            _keyboard.press("v")
            _keyboard.release("v")

        if restore and previous is not None:
            def _restore_later() -> None:
                time.sleep(restore_delay)
                try:
                    pyperclip.copy(previous)
                except Exception:
                    pass

            t = threading.Thread(target=_restore_later, daemon=True)
            t.start()
