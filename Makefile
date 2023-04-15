build:
	pip install .

clean:
	rm -rf build dist *.egg-info
	find kursl -name '__pycache__' -exec rm -rf {} +

test:
	python -m unittest discover -s tests