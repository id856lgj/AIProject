"""
Script di training per Fashion-MNIST
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.model import FashionMNISTCNN, count_parameters


class FashionMNISTCSVDataset(Dataset):
    
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Percorso al file CSV
            transform (callable, optional): Transform da applicare
        """
        # Leggi il CSV
        self.data_frame = pd.read_csv(csv_file)
        
        # La prima colonna è il label, le altre sono i pixel
        self.labels = self.data_frame.iloc[:, 0].values
        self.images = self.data_frame.iloc[:, 1:].values
        
        # Reshape per PyTorch
        self.images = self.images.reshape(-1, 1, 28, 28).astype(np.float32)
        
        # Normalizza da 0-255 a 0-1
        self.images = self.images / 255.0
        
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Converti in tensor
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def setup_data_loaders(batch_size=32):
    """Prepara i dataloader"""
    
    # Path ai file CSV
    train_csv = Path("./data/fashion-mnist_train.csv")
    test_csv = Path("./data/fashion-mnist_test.csv")
    
    # Verifica che i file esistano
    if not train_csv.exists():
        raise FileNotFoundError(f"File non trovato: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"File non trovato: {test_csv}")
    
    print(f"Caricamento da: {train_csv}")
    print(f"Caricamento da: {test_csv}")
    
    # Crea dataset
    train_dataset = FashionMNISTCSVDataset(train_csv)
    test_dataset = FashionMNISTCSVDataset(test_csv)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer):
    """Addestra per un'epoca"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for data, target in pbar:
        # Forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistiche
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Aggiorna barra
        pbar.set_postfix({
            'loss': running_loss/(pbar.n+1),
            'acc': 100.*correct/total
        })
    
    return running_loss/len(train_loader), 100.*correct/total


def validate(model, test_loader, criterion):
    """Validazione"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Validation"):
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return test_loss/len(test_loader), 100.*correct/total


def main():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=5, help="Numero di epoche")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Directory fissa per i modelli
    SAVE_DIR = Path("./models")
    
    print("-"*60)
    print("TRAINING (usando file CSV locali)")
    print("-"*60)
    print(f"Epoche: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Salvataggio modelli: {SAVE_DIR}")
    print("-"*60)
    
    # Setup data
    print("\nPreparazione dati...")
    train_loader, test_loader = setup_data_loaders(args.batch_size)
    
    # Setup model
    print("Creazione modello...")
    model = FashionMNISTCNN()
    print(f"   Parametri totali: {count_parameters(model):,}")
    
    # Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nInizio training...")
    best_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Salva best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pth")
            print(f"Salvato best model con accuracy: {best_acc:.2f}%")
    
    # Salva modello finale
    torch.save(model.state_dict(), SAVE_DIR / "final_model.pth")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completato! Tempo: {elapsed_time:.2f} secondi")
    print(f"Modelli salvati in: {SAVE_DIR}")
    print(f"Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()