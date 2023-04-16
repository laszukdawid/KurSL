build:
	pip install .

clean:
	rm -rf build dist *.egg-info
	find kursl -name '__pycache__' -exec rm -rf {} +
	find tests -name '__pycache__' -exec rm -rf {} +

test:
	python -m pytest tests

unit-test:
	python -m pytest tests/unit

func-test:
	python -m pytest tests/functional