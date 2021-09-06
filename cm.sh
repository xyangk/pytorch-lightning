#!/bin/bash

cd ~
cd pytorch-lightning
git fetch --all
git checkout tpu_pod
git pull origin tpu_pod
python3 -m pytest tests/models/test_tpu.py::test_tpu_multi_node --capture=no -v
