# autoedit — common tasks.  Run `make help` to list targets.
PORT ?= 8000

.PHONY: help sample demo serve docker-build docker-run clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sample: ## generate the synthetic source clip + transcript
	python3 make_sample.py

demo: sample ## run the pipeline on the sample (cinematic + reel)
	python3 cli.py fixtures/source.mp4 --formats cinematic reel --budget 12 \
		--transcript fixtures/sample_transcript.json --outdir out

serve: ## run the HTTP backend locally on $(PORT)
	PORT=$(PORT) python3 server.py

docker-build: ## build the container image
	docker build -t autoedit-backend .

docker-run: ## run the container locally on $(PORT)
	docker run --rm -p $(PORT):8000 autoedit-backend

clean: ## remove generated artifacts
	rm -rf out out_vo fixtures/source.mp4 job_*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
