import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.transcriber import Transcriber

class TestTranscriber(unittest.TestCase):
    def setUp(self):
        self.transcriber = Transcriber(model_name="tiny.en")

    @patch('src.transcriber.WhisperModel')
    def test_load_unload(self, mock_whisper_model):
        mock_instance = mock_whisper_model.return_value
        # Mock transcribe for the warmup call
        mock_instance.transcribe.return_value = ([], None)
        
        self.transcriber.load()
        self.assertTrue(self.transcriber.is_loaded)
        mock_whisper_model.assert_called_once()
        
        self.transcriber.unload()
        self.assertFalse(self.transcriber.is_loaded)

    @patch('src.transcriber.WhisperModel')
    def test_transcribe(self, mock_whisper_model):
        mock_instance = mock_whisper_model.return_value
        mock_instance.transcribe.side_effect = [
            ([], None), # warmup
            ([MagicMock(text=" Hello world ")], None) # actual
        ]
        
        self.transcriber.load()
        audio = np.zeros(16000, dtype="float32")
        result = self.transcriber.transcribe(audio)
        self.assertEqual(result, "Hello world")

if __name__ == '__main__':
    unittest.main()
