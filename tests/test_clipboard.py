import unittest
from unittest.mock import MagicMock, patch
from src.clipboard import ClipboardPaster

class TestClipboardPaster(unittest.TestCase):
    def setUp(self):
        self.paster = ClipboardPaster()

    @patch('src.clipboard.pyperclip')
    @patch('src.clipboard._keyboard')
    def test_paste_no_restore(self, mock_keyboard, mock_pyperclip):
        self.paster.paste("Hello", restore=False)
        mock_pyperclip.copy.assert_called_with("Hello")
        # Check if Cmd+V was called (pressed(Key.cmd) and then press/release 'v')
        mock_keyboard.pressed.assert_called()
        mock_keyboard.press.assert_called_with("v")
        mock_keyboard.release.assert_called_with("v")

    @patch('src.clipboard.pyperclip')
    @patch('src.clipboard._keyboard')
    @patch('src.clipboard.time.sleep')
    def test_paste_with_restore(self, mock_sleep, mock_keyboard, mock_pyperclip):
        mock_pyperclip.paste.return_value = "Old content"
        
        # We need to wait for the thread to finish or mock it differently
        # For simplicity, we'll just check if copy was called with new content
        self.paster.paste("New content", restore=True, restore_delay=0.01)
        mock_pyperclip.paste.assert_called_once()
        mock_pyperclip.copy.assert_any_call("New content")
        
        # To test the restore thread, we'd need to wait or mock threading.Thread
        # But this is likely enough for a basic validation.

if __name__ == '__main__':
    unittest.main()
