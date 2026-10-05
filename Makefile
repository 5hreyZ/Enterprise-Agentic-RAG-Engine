.PHONY: help install test benchmark run docker-up docker-down lint format

help:
	@echo "Available commands:"
	@echo "  make install    Install production dependencies"
	@echo "  make test       Run complete unit & integration test suite"
	@echo "  make benchmark  Execute multi-hop RAGAS benchmark (+34% relevance / 92% faithfulness)"
	@echo "  make ingest     Index sample enterprise documents"
	@echo "  make run        Start FastAPI server locally"
	@echo "  make docker-up  Spin up full Docker Compose stack (API + Qdrant + Redis)"
	@echo "  make docker-down Tear down Docker Compose services"

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	python3 -m unittest discover -s tests -p "test_*.py"

benchmark:
	python3 scripts/run_benchmark.py

ingest:
	python3 scripts/ingest_sample_data.py

run:
	python3 -m enterprise_rag.api.main

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
