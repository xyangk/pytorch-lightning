import os
import time
import torch
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
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
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


def run(enabled=False, max_epochs=20, checkpointing=False):
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = str(int(enabled))
    start = time.monotonic()

    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        logger=False,
        checkpoint_callback=checkpointing,
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
    r0 = benchmark(trials=5, max_epochs=200, checkpointing=False)
    r1 = benchmark(trials=5, max_epochs=200, checkpointing=True)
    print(r0)
    print(r1)
