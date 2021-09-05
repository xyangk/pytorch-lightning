import torch, time
import learn2learn as l2l
from typing import Optional
import numpy as np
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data._utils.worker import get_worker_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class LitMAML(pl.LightningModule):
    """docstring for LitMAML(pl.LightningModule) """
    
    def __init__(self, adapt_steps, shots, ways, meta_lr, fast_lr, classifier):
        super(LitMAML, self).__init__()
        self.adapt_steps = adapt_steps
        self.shots = shots
        self.ways = ways
        self.meta_lr = meta_lr
        self.fast_lr = fast_lr
        self.classifier = classifier
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')
    
    def training_step(self, batch, batch_idx):
        train_loss, train_accuracy = self.meta_learn(batch, batch_idx)
        values = {'train_loss':train_loss.item(), 'train_accuracy':train_accuracy.item()}
        self.log_dict(values, prog_bar = True)
        return train_loss.item()
        
    def validation_step(self, batch, batch_idx):
        valid_loss, valid_accuracy = self.meta_learn(batch, batch_idx)
        values = {'valid_loss':valid_loss.item(), 'valid_accuracy':valid_accuracy.item()}
        self.log_dict(values, prog_bar = True)
        return valid_loss.item()
    
    def test_step(self, batch, batch_idx):
        test_loss, test_accuracy = self.meta_learn(batch, batch_idx)
        values = {'test_loss':test_loss.item(), 'test_accuracy':test_accuracy.item()}
        self.log_dict(values, prog_bar = True)
        return test_loss.item()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.classifier.parameters(), lr = self.meta_lr)
        return optimizer
        
    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx):
        learner = self.classifier.clone()
        data, labels = batch

        print(data, labels, self.shots, self.ways)

        # Seperate data into adaptation and evaluation sets
        adaptation_indices = np.zeros(data.size(0), dtype = bool)
        adaptation_indices[np.arange(self.shots * self.ways)*2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        
        # Adapt the model
        for step in range(self.adapt_steps):
            train_error = self.loss(learner(adaptation_data), adaptation_labels)
            train_error /= len(adaptation_data)
            learner.adapt(train_error)
            
        # Evaluating the adapted model
        predictions = learner(evaluation_data)
        valid_error = self.loss(predictions, evaluation_labels)
        valid_error /= len(evaluation_data)
        valid_accuracy = accuracy(predictions, evaluation_labels)
        return valid_error, valid_accuracy

class TaskDataParallel(IterableDataset):

    def __init__(
        self,
        taskset: l2l.data.TaskDataset,
        global_rank: int, 
        world_size: int,
        num_workers: int,
        epoch_length: int,
        seed: int,
    ):
        self.taskset = taskset
        self.global_rank = global_rank
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
        is_global_zero = self.global_rank == 0
        return self.global_rank + self.worker_id + int(not is_global_zero)

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


class EpisodicBatcher(pl.LightningDataModule):
    """docstring for EpisodicBatcher(pl.LightningDataModule)"""
    
    def __init__(self, train_tasks, validation_tasks = None, test_tasks = None, epoch_length = 1, num_workers: int = 2, seed: int = 42):
        super(EpisodicBatcher, self).__init__()
        self.train_tasks = train_tasks
        if validation_tasks is None:
            validation_tasks = train_tasks
        self.validation_tasks = validation_tasks
        if test_tasks is None:
            test_tasks = validation_tasks
        self.test_tasks = test_tasks
        self.epoch_length = epoch_length
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: Optional[str]) -> None:
        if isinstance(self.trainer.training_type_plugin, DDPPlugin):
            print(self.trainer.global_rank, self.trainer.world_size)
            self.train_tasks_parallel = TaskDataParallel(
                taskset=self.train_tasks,
                global_rank=self.trainer.global_rank, 
                world_size=self.trainer.world_size,
                num_workers=self.num_workers,
                epoch_length=self.epoch_length,
                seed=self.seed)

            self.validation_tasks_parallel = TaskDataParallel(
                taskset=self.validation_tasks,
                global_rank=self.trainer.global_rank, 
                world_size=self.trainer.world_size,
                num_workers=self.num_workers,
                epoch_length=self.epoch_length,
                seed=self.seed)

            self.test_tasks_parallel = TaskDataParallel(
                taskset=self.test_tasks,
                global_rank=self.trainer.global_rank, 
                world_size=self.trainer.world_size,
                num_workers=self.num_workers,
                epoch_length=self.epoch_length,
                seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_tasks_parallel, num_workers=self.num_workers)
        
    def val_dataloader(self):
        pass
        
    def test_dataloader(self):
        pass


def main(
    ways = 5,
    shots = 1,
    meta_lr = 3e-3,
    fast_lr = 5e-1,
    adapt_steps = 1,
    meta_bsz = 32,
    iters = 60000,
    seed = 42
    ):
    
    pl.seed_everything(seed)
    
    # Create tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(
        'omniglot', train_samples = 2 * shots, train_ways = ways, test_samples = 2 * shots, test_ways = ways, root = '~/data')

    classifier = l2l.vision.models.OmniglotFC(28**2, ways)
    classifier = l2l.algorithms.MAML(classifier, lr = fast_lr, first_order = False)
    
    # init model
    maml = LitMAML(adapt_steps, shots, ways, meta_lr, fast_lr, classifier)
    
    episodic_data = EpisodicBatcher(tasksets.train, tasksets.validation, tasksets.test, epoch_length = 4)
    trainer = pl.Trainer(fast_dev_run=True, gpus = 2, accelerator = 'ddp')
    trainer.fit(maml, episodic_data)	


if __name__ == '__main__':
	t1 = time.time()
	main()
	print("Time taken for training:", time.time() - t1)