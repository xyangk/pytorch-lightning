import os
from argparse import ArgumentParser
from contextlib import contextmanager
from time import perf_counter
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class RandomFFCVDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        # convert to numpy: https://github.com/libffcv/ffcv/issues/101
        # return a tuple: https://github.com/libffcv/ffcv/issues/103
        return (self.data[index].numpy(),)

    def __len__(self):
        return self.len


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self, d):
        super().__init__()
        self.layer = torch.nn.Linear(d, 1)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


N, d = 2 ** 12, 128
batch_size = 8
num_workers = os.cpu_count()


def prepare_data_torch() -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = RandomDataset(d, N)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    val_data = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    test_data = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return train_data, val_data, test_data


def prepare_data_ffcv(cwd: str, gpus: int) -> Tuple["Loader", "Loader", "Loader"]:
    from ffcv.loader import Loader, OrderOption
    from ffcv.writer import DatasetWriter
    from ffcv.fields import NDArrayField
    import numpy as np

    dataset_path = os.path.join(cwd, "random")
    if not os.path.exists(dataset_path):
        dataset = RandomFFCVDataset(d, N)
        # https://docs.ffcv.io/writing_datasets.html#writing-a-dataset-to-ffcv-format
        fields = {"covariate": NDArrayField(shape=(d,), dtype=np.dtype("float32"))}
        writer = DatasetWriter(dataset_path, fields)
        writer.from_indexed_dataset(dataset)
    assert os.path.isfile(dataset_path)

    # https://docs.ffcv.io/making_dataloaders.html#dataset-ordering
    if gpus == 0:
        # dataset fits in memory: fastest option
        kwargs = {"order": OrderOption.RANDOM}
    elif gpus == 1:
        # dataset does not fit in memory
        kwargs = {"order": OrderOption.QUASI_RANDOM, "os_cache": False}
    elif gpus > 1:
        # we assume dataset will fit in memory here
        # FIXME: distributed
        kwargs = {"order": OrderOption.RANDOM, "os_cache": True, "distributed": False}
    else:
        raise ValueError

    train_data = Loader(dataset_path, batch_size=batch_size, num_workers=num_workers, **kwargs)

    kwargs["order"] = OrderOption.SEQUENTIAL
    val_data = Loader(dataset_path, batch_size=batch_size, num_workers=num_workers, **kwargs)
    test_data = Loader(dataset_path, batch_size=batch_size, num_workers=num_workers, **kwargs)
    return train_data, val_data, test_data


@contextmanager
def catchtime() -> float:
    """https://stackoverflow.com/a/62956469"""
    start = perf_counter()
    yield lambda: perf_counter() - start


def run():
    cwd = os.getcwd()

    parser = ArgumentParser()
    parser.add_argument("--ffcv", action="store_true")
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()

    if args.ffcv:
        train_data, val_data, test_data = prepare_data_ffcv(cwd, args.gpus)
    else:
        train_data, val_data, test_data = prepare_data_torch()

    kwargs = {}
    if args.gpus:
        kwargs = {"accelerator": "gpu", "devices": args.gpus}
        if args.gpus > 1:
            kwargs["strategy"] = "ddp_find_unused_parameters_False"

    trainer = Trainer(
        default_root_dir=cwd,
        num_sanity_val_steps=0,
        max_epochs=10,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        # callbacks=TQDMProgressBar(refresh_rate=100),
        benchmark=True,
        **kwargs,
    )

    model = BoringModel(d)

    with catchtime() as t:
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    if trainer.is_global_zero:
        print(f"Fit time: {t():.4f} secs")

    with catchtime() as t:
        trainer.test(model, dataloaders=test_data, verbose=False)
    if trainer.is_global_zero:
        print(f"Test time: {t():.4f} secs")


if __name__ == "__main__":
    run()
