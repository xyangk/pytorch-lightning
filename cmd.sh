
#!/bin/bash

# my commands

git clone https://github.com/PyTorchLightning/pytorch-lightning
cd pytorch-lightning
git fetch --all
git checkout tpu_multi_pod
git pull origin tpu_multi_pod
sudo python3 -m pip install -e .
sudo python3 -m pip install -r requirements.txt
sudo python3 -m pip install -r requirements/test.txt
python3 -m pytest tests/models/test_tpu.py::test_tpu_multi_pod
