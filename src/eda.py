"""
EDA per Fashion-MNIST
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
DATA_DIR = Path("./data")
REPORTS_DIR = Path("./reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Classi di Fashion-MNIST preso dalla documentazione
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def eda():
    print("-"*60)
    print("EDA - Fashion-MNIST Dataset")
    print("-"*60)

    print("\nCaricamento dati...")

    train_df = pd.read_csv(DATA_DIR / "fashion-mnist_train.csv")
    test_df = pd.read_csv(DATA_DIR / "fashion-mnist_test.csv")

    print(f"Train set: {train_df.shape[0]} samples, {train_df.shape[1]-1} features")
    print(f"Test set: {test_df.shape[0]} samples")

    # Statistiche di base
    print("\nStatistiche di base:")
    print(f"- Train data size: {len(train_df)}")
    print(f"- Test data size: {len(test_df)}")

    # Distribuzione classi
    labels = train_df['label'].values
    unique, counts = np.unique(labels, return_counts=True)
    print("\n- Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  {CLASSES[cls]}: {count} samples ({count/len(labels)*100:.1f}%)")

    # Statistiche pixel
    print("\nStatistiche pixel:")
    pixel_cols = [col for col in train_df.columns if col != 'label']
    pixel_data = train_df[pixel_cols].values
    print(f"- Pixel min: {pixel_data.min()}")
    print(f"- Pixel max: {pixel_data.max()}")
    print(f"- Pixel mean: {pixel_data.mean():.2f}")
    print(f"- Pixel std: {pixel_data.std():.2f}")

    # GRAFICO 1: Distribuzione classi
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    bars = plt.bar(CLASSES, counts, color='skyblue', edgecolor='navy')
    plt.title("Class Distribution - Training Set", fontsize=14)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')

    # Aggiungi valori sulle barre
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=9)

    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=CLASSES, autopct='%1.1f%%', startangle=90)
    plt.title("Class Distribution (%)", fontsize=14)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
    print(f"\n Grafico salvato: {REPORTS_DIR}/class_distribution.png")
    plt.close()

    # GRAFICO 2: Sample images
    print("\n Visualizzazione sample images...")

    # Prendi 10 immagini random (una per classe)
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(10):
        # Prendi un sample della classe i
        class_samples = train_df[train_df['label'] == i].iloc[0, 1:].values
        class_samples = class_samples.reshape(28, 28)
        
        axes[i].imshow(class_samples, cmap='gray')
        axes[i].set_title(f"{CLASSES[i]}")
        axes[i].axis('off')

    plt.suptitle("Sample Images - One per Class", fontsize=16)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "sample_images.png", dpi=150, bbox_inches='tight')
    print(f" Grafico salvato: {REPORTS_DIR}/sample_images.png")
    plt.close()

    print("\n" + "-"*60)
    print(" EDA completata!")
    print(f" Tutti i grafici sono stati salvati in: {REPORTS_DIR}")
    print("-"*60)


if __name__ == "__main__":
    eda()
