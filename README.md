## Programma per Esame MLOps

Il seguente repository è dello studente Luca Cobelli(VR391611) relativo alla consegna dell'esame Sviluppo e ciclo vitale di software di intelligenza artificiale (2024/2025).
Come data set è stato scelto Fashion-MNIST Dataset.

Per testare il software sono disponibili due modalità, ed i vari comandi da terminale sotto indicati sono stati eseguiti da PowerShell.

1) **ZIP repository**
   
   - scaricare in formato zip sul proprio PC il repository, ed estrarlo
   
   - Apri PowerShell, e spospostarsi nella cartella generatasi "AIProject-master"
   
   - sempre dal repository si posso trovare i DataSet da utilizzare, sulla destra del progetto in GitHub è presente una Release contenente i due file necessari, da scaricare:
     
     - fashion-mnist_test.csv
       `https://github.com/id856lgj/AIProject/releases/download/v1.0/fashion-mnist_test.csv`
     
     - fashion-mnist_train.csv
       `https://github.com/id856lgj/AIProject/releases/download/v1.0/fashion-mnist_train.csv`
   
   - all'interno del progetto estratto in precedenza vi è la cartella "data". Vanno copiati dentro i due file scaricati al punto precedente (i dataset di TEST e TRAIN)
   
   - ora creiamo un VENV per l'esecuzione del progetto
     
     python -m venv venv
   
   - attiviamo venv
     
     .\venv\Scripts\activate
   
   - installiamo tutte le dipendenze necessarie
       make install
   
   - eseguiamo l'EDA, ed una volta fatto troveremo i grafici nella cartella "/reports/figures"
      python src\eda.py
   
   - per eseguire il training ed ottenere il modello, modello che si troverà nella cartella "models"
       python -m src.train --epochs 3 --batch-size 32

2) **DOCKER**
   
   - crare una cartella per il progetto, ad esempio "AIProject", e spostarsi dentro
   
   - creare tre cartelle "data" - "models" - "reports/figures"
   
   - nel repository si possono trovare i DataSet da utilizzare, sulla destra del progetto in GitHub è presente una Release contenente i due file necessari, da scaricare:
     
     - fashion-mnist_test.csv
       `https://github.com/id856lgj/AIProject/releases/download/v1.0/fashion-mnist_test.csv`
     
     - fashion-mnist_train.csv
       `https://github.com/id856lgj/AIProject/releases/download/v1.0/fashion-mnist_train.csv`
   
   - posizionare i file scaricati nella cartella "data".
   
   - scaricare immagine da Docker hub:
       docker pull id856lgj/aiproject:latest
   
   - per eseguire l'EDA (troverete i grafici generati nella cartella "reports/figures" )
       docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/reports/figures:/app/reports/figures id856lgj/aiproject python -m src.eda
   
   - per eseguire il Training eseguire (troverete i modelli nella cartella "models"):
       docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports id856lgj/aiproject python -m src.train --epochs 3 --batch-size 32
