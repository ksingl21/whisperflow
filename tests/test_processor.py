import unittest
from unittest.mock import MagicMock, patch
from src.processor import Processor

class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = Processor(model="test-model")

    @patch('src.processor.ollama.Client')
    def test_process_success(self, mock_ollama_client):
        mock_client_instance = mock_ollama_client.return_value
        mock_response = MagicMock()
        mock_response.response = "Cleaned text"
        mock_client_instance.generate.return_value = mock_response
        
        # We need to re-initialize or mock the internal client because it was created in __init__
        self.processor._client = mock_client_instance
        
        result = self.processor.process("Raw text")
        self.assertEqual(result, "Cleaned text")
        mock_client_instance.generate.assert_called()

    @patch('src.processor.ollama.Client')
    def test_process_failure_fallback(self, mock_ollama_client):
        mock_client_instance = mock_ollama_client.return_value
        mock_client_instance.generate.side_effect = Exception("Ollama down")
        
        self.processor._client = mock_client_instance
        
        result = self.processor.process("Raw text")
        self.assertEqual(result, "Raw text")

if __name__ == '__main__':
    unittest.main()
