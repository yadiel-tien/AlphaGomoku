import unittest
from inference import InferenceEngine as Infer

class TestInference(unittest.TestCase):
    def test_infer(self):
        inference = Infer.makeEngine(299)
        inference.infer()
        self.assertTrue(True)