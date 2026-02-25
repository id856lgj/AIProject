"""
Unit test per le funzioni di training
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import FashionMNISTCNN
from src.train import train_epoch, validate


class TestTrainingFunctions(unittest.TestCase):
    """Test per le funzioni di training"""
    
    def setUp(self):
        """Setup per test di training"""
        self.model = FashionMNISTCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Crea dataloader fittizio
        dummy_images = torch.randn(5, 1, 28, 28)
        dummy_labels = torch.randint(0, 10, (5,))
        dummy_dataset = TensorDataset(dummy_images, dummy_labels)
        self.dummy_loader = DataLoader(dummy_dataset, batch_size=2)
        
    def test_train_epoch_executes(self):
        """Test che train_epoch venga eseguito senza errori"""
        try:
            loss, acc = train_epoch(self.model, self.dummy_loader, 
                                    self.criterion, self.optimizer)
            self.assertIsInstance(loss, float)
            self.assertIsInstance(acc, float)
        except Exception as e:
            self.fail(f"train_epoch ha sollevato un'eccezione: {e}")
            
    def test_validate_executes(self):
        """Test che validate venga eseguito senza errori"""
        try:
            loss, acc = validate(self.model, self.dummy_loader, self.criterion)
            self.assertIsInstance(loss, float)
            self.assertIsInstance(acc, float)
        except Exception as e:
            self.fail(f"validate ha sollevato un'eccezione: {e}")


if __name__ == '__main__':
    unittest.main()