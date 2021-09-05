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
import torch.multiprocessing as mp
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.data._utils.worker import get_worker_info
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.dataset import Dataset, IterableDataset
import os, random, torch, time
import learn2learn as l2l
import numpy as np
from torch import nn, optim
from torch.utils.data.dataset import IterableDataset
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import tests.helpers.utils as tutils
from tests.helpers.runif import RunIf


class TaskDataParallel(IterableDataset):

    def __init__(
        self,
        taskset: l2l.data.TaskDataset,
        rank: int, 
        world_size: int,
        num_workers: int,
        epoch_length: int,
        seed: int,
    ):
        self.taskset = taskset
        self.rank = rank
        self.world_size = world_size
        self.num_workers = 1 if num_workers == 0 else num_workers
        self.worker_world_size = self.world_size * self.num_workers
        self.epoch_length = epoch_length
        self.seed = seed
        self.iteration = 0

        if epoch_length % self.worker_world_size != 0:
            raise MisconfigurationException("The `epoch_lenght` should be divisible by `world_size`.")

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    @property
    def worker_rank(self) -> int:
        return self.rank + self.worker_id

    def __len__(self):
        return self.epoch_length // self.world_size

    def __iter__(self):
        self.iteration += 1
        pl.seed_everything(self.seed + self.iteration)
        return self

    def __next__(self):
        task_descriptions = []
        for _ in range(self.worker_world_size):
            task_descriptions.append(self.taskset.sample_task_description())

        for task_description in task_descriptions:
            print(self.worker_rank, self.taskset.get_task(task_description)[1])

        return self.taskset.get_task(task_descriptions[self.worker_rank])

def _main(rank, world_size, seed):

    print(rank, world_size, seed)

    num_tasks = -1

    pl.seed_everything(42)
    
    # Create tasksets using the benchmark interface
    datasets, transforms = l2l.vision.benchmarks.omniglot_tasksets(train_samples = 2, train_ways = 5, test_samples = 2, test_ways = 5, root = '~/data')

    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )

    dataset = TaskDataParallel(taskset=train_tasks, rank=rank, world_size=world_size, num_workers=2, epoch_length=4, seed=seed)
    loader = DataLoader(dataset, num_workers=2)

    dataloader_iter = iter(loader)
    batch = next(dataloader_iter)
    print(rank, batch[1])

    batch = next(dataloader_iter)
    print(rank, batch[1])


def main():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(
        _main, args=(worldsize, 42), nprocs=worldsize
    )

if __name__ == "__main__":
    main()