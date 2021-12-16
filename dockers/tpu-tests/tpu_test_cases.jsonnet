local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'pl',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 900, # 15 minutes, in seconds.

  image: 'pytorchlightning/pytorch_lightning',
  imageTag: 'base-xla-py{PYTHON_VERSION}-torch{PYTORCH_VERSION}',

  tpuSettings+: {
    softwareVersion: 'pytorch-{PYTORCH_VERSION}',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      source ~/.bashrc
      conda activate lightning
      mkdir -p /home/runner/work/pytorch-lightning && cd /home/runner/work/pytorch-lightning
      git clone https://github.com/PyTorchLightning/pytorch-lightning.git
      cd pytorch-lightning
      echo $PWD
      git ls-remote --refs origin
      git fetch origin "refs/pull/{PR_NUMBER}/head:pr/{PR_NUMBER}" && git checkout "pr/{PR_NUMBER}"
      git checkout {SHA}
      pip install -e .
      echo $KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
      export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"

      # find tests marked as `@RunIf(tpu=True)`. done manually instead of with pytest because it is faster
      grep_output=$(grep --recursive --word-regexp 'tests' --regexp 'tpu=True' --include '*.py' --exclude 'tests/conftest.py')
      # file paths, remove duplicates
      files=$(echo "$grep_output" | cut -f1 -d: | sort | uniq)

      coverage run --source=pytorch_lightning -m pytest -v --capture=no $files
      test_exit_code=$?
      echo "\n||| END PYTEST LOGS |||\n"
      coverage xml
      cat coverage.xml | tr -d '\t'
      test $test_exit_code -eq 0
    |||
  ),
};

tputests.oneshotJob
