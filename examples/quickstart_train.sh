#!/usr/bin/env bash
set -euo pipefail

uv run python -m zebra_prop.train \
  data_dir=./examples/data \
  property_name=band_gap \
  task_name=sample \
  test_fold=0
