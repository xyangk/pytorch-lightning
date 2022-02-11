import os

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


class MyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr):
        assert all(param.device.type == "cuda" for param in params)
        super().__init__(params, lr)


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)

        # assert all(param.device.type == "cuda" for param in self.optimizers().optimizer.param_groups[0]["params"])
        return {"loss": loss}

    def configure_optimizers(self):
        return MyOptimizer(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    params_before = list(param.clone() for param in model.parameters())
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=1,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=[5],
        strategy="ddp",
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    params_after = list(param.clone() for param in model.parameters())
    for p_before, p_after in zip(params_before, params_after):
        assert torch.not_equal(p_before, p_after).all()


if __name__ == "__main__":
    run()
