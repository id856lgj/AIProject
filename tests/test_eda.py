"""
Test per EDA
"""

import os
from pathlib import Path
from src.eda import eda

def test_eda():
    """Test che la funzione EDA venga eseguita senza errori"""
    
    # Esegue la funzione EDA
    result = eda()
    
    # Verifica che i file siano stati creati
    reports_dir = Path("./reports/figures")
    assert (reports_dir / "class_distribution.png").exists()
    assert (reports_dir / "sample_images.png").exists()
    
    # Pulisce
    os.remove(reports_dir / "class_distribution.png")
    os.remove(reports_dir / "sample_images.png")
    
    print("EDA test passed: EDA function executed successfully.")