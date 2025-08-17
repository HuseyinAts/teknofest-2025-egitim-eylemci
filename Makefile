.PHONY: help setup-light setup-full test clean

help:
	@echo "Commands:"
	@echo "  make setup-light  - Install core packages only"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean cache files"

setup-light:
	pip install pandas numpy requests python-dotenv

test:
	python test_setup.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true