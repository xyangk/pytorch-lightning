import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self, input_size=32, num_layers=1):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(input_size, input_size) for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)

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
        return torch.optim.SGD(self.layers.parameters(), lr=0.1)


def run(enabled=False, max_epochs=20, checkpointing=False, input_size=32, num_layers=1, **trainer_args):
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = str(int(enabled))
    start = time.monotonic()

    train_data = DataLoader(RandomDataset(input_size, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(input_size, 64), batch_size=2)

    model = BoringModel(input_size=input_size, num_layers=num_layers)
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        logger=False,
        checkpoint_callback=checkpointing,
        **trainer_args,
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    duration = time.monotonic() - start
    return duration


def benchmark(trials=1, **settings):
    duration_enabled = 0
    duration_disabled = 0
    for _ in range(trials):
        duration_enabled += run(enabled=True, **settings)
        duration_disabled += run(enabled=False, **settings)

    duration_enabled /= trials
    duration_disabled /= trials

    relative = duration_enabled / duration_disabled

    result = {"settings": settings, "enabled": duration_enabled, "disabled": duration_disabled, "relative": relative}
    return result


if __name__ == "__main__":
    gpu = torch.cuda.is_available()
    experiments = [
        dict(trials=5, max_epochs=50, checkpointing=False),
        dict(trials=5, max_epochs=50, checkpointing=True),
        dict(trials=5, max_epochs=50, input_size=32),
        dict(trials=5, max_epochs=50, input_size=64),
        dict(trials=5, max_epochs=50, input_size=128),
        dict(trials=5, max_epochs=50, num_layers=1),
        dict(trials=5, max_epochs=50, num_layers=10),
        dict(trials=5, max_epochs=50, num_layers=100),
    ]

    results = []
    for exp in experiments:
        r = benchmark(**exp, gpus=int(gpu))
        results.append(r)
        print(r)

    print("Summary")
    for r in results:
        print(r)
