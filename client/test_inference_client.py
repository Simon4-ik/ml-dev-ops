import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
import tempfile
from PIL import Image

# Import the class and function to test
from client.inference_client import TritonInferenceClient, log_result

class TestTritonInferenceClient(unittest.TestCase):
    def setUp(self):
        # Create a temporary image for preprocessing tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_img_path = os.path.join(self.temp_dir.name, "test.jpg")
        img = Image.new("RGB", (100, 100), color="red")
        img.save(self.temp_img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preprocess_resnet(self):
        with patch("client.inference_client.httpclient.InferenceServerClient") as mock_client:
            client = TritonInferenceClient(url="localhost:8000")
            processed = client.preprocess(self.temp_img_path, h=224, w=224, is_resnet=True)
            
            # Check shape: [batch, channels, height, width]
            self.assertEqual(processed.shape, (1, 3, 224, 224))
            self.assertEqual(processed.dtype, np.float32)

    def test_preprocess_yolo(self):
        with patch("client.inference_client.httpclient.InferenceServerClient") as mock_client:
            client = TritonInferenceClient(url="localhost:8000")
            processed = client.preprocess(self.temp_img_path, h=640, w=640, is_resnet=False)
            
            # Check shape
            self.assertEqual(processed.shape, (1, 3, 640, 640))
            self.assertEqual(processed.dtype, np.float32)
            # Without resnet normalization, values should be between 0 and 1
            self.assertTrue(np.all(processed >= 0.0) and np.all(processed <= 1.0))

    def test_postprocess_resnet(self):
        with patch("client.inference_client.httpclient.InferenceServerClient") as mock_client:
            client = TritonInferenceClient(url="localhost:8000")
            # Setup dummy output scores: class index 5 has the highest score
            dummy_output = np.zeros((1, 1000), dtype=np.float32)
            dummy_output[0, 5] = 20.0 # Extremely high score, will dominate after softmax
            
            results = client.postprocess_resnet(dummy_output)
            self.assertEqual(len(results), 1)
            self.assertIn("score", results[0])
            self.assertGreater(results[0]["score"], 0.99)

    def test_log_result(self):
        # Mock file writing to avoid creating garbage files
        with patch("client.inference_client.os.makedirs") as mock_makedirs, \
             patch("client.inference_client.open", unittest.mock.mock_open()) as mock_open:
            log_result("test_model", "test.jpg", "test_label", 0.95, 0.0123)
            mock_open.assert_called_once_with("logs/inference_history.csv", "a", newline="")

if __name__ == "__main__":
    unittest.main()
