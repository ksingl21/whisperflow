import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.audio import AudioRecorder

class TestAudioRecorder(unittest.TestCase):
    def setUp(self):
        self.recorder = AudioRecorder(min_seconds=0.1, max_seconds=1.0)

    @patch('src.audio.sd.InputStream')
    def test_start_stop(self, mock_input_stream):
        # Mock the stream
        mock_stream_instance = mock_input_stream.return_value
        
        self.recorder.start()
        self.assertTrue(self.recorder.is_recording)
        mock_input_stream.assert_called_once()
        
        # Simulate some audio data: 2 blocks to exceed 0.1s (1600 samples)
        # and non-zero to exceed silence threshold (0.01)
        dummy_data = np.ones((1024, 1), dtype='float32') * 0.1
        self.recorder._callback(dummy_data, 1024, None, None)
        self.recorder._callback(dummy_data, 1024, None, None)
        
        audio = self.recorder.stop()
        self.assertFalse(self.recorder.is_recording)
        self.assertIsNotNone(audio)
        self.assertEqual(len(audio), 2048)
        mock_stream_instance.stop.assert_called_once()
        mock_stream_instance.close.assert_called_once()

    @patch('src.audio.sd.InputStream')
    def test_too_short(self, mock_input_stream):
        self.recorder.start()
        # No data added
        audio = self.recorder.stop()
        self.assertIsNone(audio)

    @patch('src.audio.sd.InputStream')
    def test_silence(self, mock_input_stream):
        self.recorder.start()
        # Very quiet data but enough of it to pass duration check
        dummy_data = np.zeros((1024, 1), dtype='float32') + 0.0001
        self.recorder._callback(dummy_data, 1024, None, None)
        self.recorder._callback(dummy_data, 1024, None, None)
        audio = self.recorder.stop()
        self.assertIsNone(audio)

if __name__ == '__main__':
    unittest.main()
