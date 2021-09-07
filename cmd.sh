
git clone https://github.com/PyTorchLightning/pytorch-lightning
cd pytorch-lightning
git fetch --all
git checkout tpu_multi_pod
git pull origin tpu_multi_pod
python3 -m pip install -e '.[devel]'
python3 -m pytest tests/models/test_tpu.py::test_tpu_multi_pod
