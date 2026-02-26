Programma per Esame MLOps


docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports aiproject python -m src.train --epochs 3 --batch-size 32

docker run -it --rm -v ${PWD}/data:/app/data -v ${PWD}/reports/figures:/app/reports/figures aiproject /bin/bash
python -m src.eda
