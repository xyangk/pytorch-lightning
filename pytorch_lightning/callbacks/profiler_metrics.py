# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback


class ProfilerMetrics(Callback):

    ON_TRAIN_BATCH_END_RECORD_FUNCTIONS = [
        "run_training_batch",
        "training_step_and_backward",
        "training_step",
        "backward",
    ]

    def on_train_batch_end(self, trainer: "pl.Trainer", *args, **kwargs):

        for record_function in self.ON_TRAIN_BATCH_END_RECORD_FUNCTIONS:
            record_time = trainer.profiler.recorded_durations[record_function][-1]
            metric = {record_function: record_time}
            trainer.logger.log_metrics(metric, step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        run_training_epoch_time = trainer.profiler.recorded_durations["run_training_epoch"]

        if len(run_training_epoch_time) > 0:
            epoch_time = {"epoch_time": run_training_epoch_time[-1]}
            trainer.logger.log_metrics(epoch_time, step=trainer.current_epoch)
