"""
Unit test per il dataset personalizzato
"""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import torch

from src.train import FashionMNISTCSVDataset


class TestDataset(unittest.TestCase):
    """Test per il dataset personalizzato"""
    
    def setUp(self):
        """Crea un file CSV temporaneo per i test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = Path(self.temp_dir) / "test.csv"
        
        # Crea dati fittizi
        data = []
        for i in range(10):
            # Valori pixel realistici tra 0 e 255 (usando modulo 256)
            pixel_values = [float(j % 256) for j in range(784)]
            row = [i % 10] + pixel_values
            data.append(row)
        
        # Crea colonne: label, pixel1, pixel2, ..., pixel784
        columns = ['label'] + [f'pixel{j}' for j in range(1, 785)]
        
        # Salva come CSV CON header
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.test_csv, index=False)
        
    def tearDown(self):
        """Pulisce i file temporanei"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_dataset_loading(self):
        """Test che il dataset carichi correttamente"""
        dataset = FashionMNISTCSVDataset(self.test_csv)
        
        # Verifica dimensioni
        self.assertEqual(len(dataset), 10)
        
        # Verifica primo elemento
        image, label = dataset[0]
        self.assertEqual(image.shape, (1, 28, 28))
        self.assertIsInstance(label, torch.Tensor)
        
    def test_dataset_labels(self):
        """Test che le label siano corrette"""
        dataset = FashionMNISTCSVDataset(self.test_csv)
        
        for i in range(10):
            _, label = dataset[i]
            self.assertEqual(label.item(), i % 10)
            
    def test_dataset_image_values(self):
        """Test che i valori dell'immagine siano normalizzati (0-1)"""
        dataset = FashionMNISTCSVDataset(self.test_csv)
        
        image, _ = dataset[0]
        self.assertLessEqual(image.max(), 1.0)
        self.assertGreaterEqual(image.min(), 0.0)

    def test_setup_data_loaders(self):
        """Test che i dataloader vengano creati correttamente"""
        from src.train import setup_data_loaders
        
        # Usa batch size piccolo per test
        train_loader, test_loader = setup_data_loaders(batch_size=4)
        
        # Verifica che siano dataloader
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        
        # Verifica che abbiano dati
        train_batch = next(iter(train_loader))
        self.assertEqual(len(train_batch), 2)  # (images, labels)



if __name__ == '__main__':
    unittest.main()