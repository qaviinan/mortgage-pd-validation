.PHONY: help data target features train validate figures report all test lint smoke clean

CONFIG ?= configs/default.yaml
YEARS  ?=

help:
	@echo "make data    - ingest data/raw/sample_YYYY.zip into data/processed/*.parquet, write manifest"
	@echo "make target  - build default targets and waterfall for each ingested vintage"
	@echo "make test    - pytest on synthetic fixtures (no download needed)"
	@echo "make smoke   - synthetic vintage through ingest and target, in data/smoke/"
	@echo "make lint    - ruff"

data:
	uv run python -m pdval.ingest --config $(CONFIG) $(if $(YEARS),--years $(YEARS),)

target:
	uv run python -m pdval.firstpass --config $(CONFIG)

features:
	uv run python -m pdval.features --config $(CONFIG)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

smoke:  ## writes only to data/smoke/, never data/raw
	uv run python -c "from pathlib import Path; from pdval.synth import make_vintage; print(make_vintage(2016, 2000, Path('data/smoke/raw'), seed=7))"
	uv run python -m pdval.ingest --config $(CONFIG) --years 2016 --raw-dir data/smoke/raw --interim-dir data/smoke/interim --processed-dir data/smoke/processed --manifest data/smoke/manifest.json
	uv run python -m pdval.firstpass --config $(CONFIG) --processed-dir data/smoke/processed --out-dir data/smoke/outputs

clean:
	rm -rf data/interim data/processed

train:
	uv run python -m pdval.train --config $(CONFIG)

validate:
	uv run python -m pdval.validate --config $(CONFIG)

figures:
	uv run python -m pdval.report.figures

report:
	uv run python -m pdval.report.build

all: data target features train validate figures report
