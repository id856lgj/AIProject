FROM python:3.13-slim

WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential 

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia il resto del progetto
COPY . .

# Installa il pacchetto in modalità development
RUN pip install -e .

# Crea directory necessarie
RUN mkdir -p /app/data /app/models /app/reports/figures

# Comando di default
CMD ["python", "-u", "-m", "src.train", "--epochs", "3", "--batch-size", "32"]