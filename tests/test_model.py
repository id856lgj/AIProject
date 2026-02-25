"""
Unit test per il modello e le funzioni di training
"""

import unittest
import torch
import torch.nn as nn

from src.model import FashionMNISTCNN, count_parameters



class TestModel(unittest.TestCase):
    """Test per il modello"""
    
    def setUp(self):
        """Setup prima di ogni test"""
        self.batch_size = 4
        self.model = FashionMNISTCNN()
        
    def test_model_creation(self):
        """Test che il modello venga creato correttamente"""
        self.assertIsInstance(self.model, FashionMNISTCNN)
        
    def test_model_parameters(self):
        """Test che il modello abbia parametri"""
        params = count_parameters(self.model)
        self.assertGreater(params, 0)
        print(f"Parametri del modello: {params:,}")
        
    def test_forward_pass(self):
        """Test forward pass con input fittizio"""
        test_input = torch.randn(self.batch_size, 1, 28, 28)
        output = self.model(test_input)
        self.assertEqual(output.shape, (self.batch_size, 10))
        
    def test_forward_pass_different_batch(self):
        """Test forward pass con batch size diverso"""
        test_input = torch.randn(2, 1, 28, 28)
        output = self.model(test_input)
        self.assertEqual(output.shape, (2, 10))
        
    def test_model_gradient_flow(self):
        """Test che i gradienti vengano calcolati"""
        test_input = torch.randn(1, 1, 28, 28)
        target = torch.tensor([3])
        
        output = self.model(test_input)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Verifica che ci siano gradienti
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad)



if __name__ == '__main__':
    unittest.main()