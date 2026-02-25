install:
	python -m pip install --upgrade pip && pip install -r requirements.txt
	@echo "Installation complete. You can now run the project."

lint:
	pylint --disable=R,C src/
	@echo "Linting complete. No issues found."

test:
	python -m pytest -vv --cov=src tests/
	@echo "Testing complete. All tests passed."

all: install lint test