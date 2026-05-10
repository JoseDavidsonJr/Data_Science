install:
	pip install .

run:
	python main.py

docker-build:
	docker build -t facape-projeto .

docker-run:
	docker run facape-projeto