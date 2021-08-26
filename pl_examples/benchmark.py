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


def run(enabled=False, max_epochs=20, checkpointing=False, input_size=32, num_layers=1, num_workers=0, **trainer_args):
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = str(int(enabled))
    start = time.monotonic()

    train_data = DataLoader(RandomDataset(input_size, 64), batch_size=2, num_workers=num_workers)
    val_data = DataLoader(RandomDataset(input_size, 64), batch_size=2, num_workers=num_workers)

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
        # dict(trials=5, max_epochs=50, checkpointing=False),
        # dict(trials=5, max_epochs=50, checkpointing=True),
        # dict(trials=5, max_epochs=50, input_size=32),
        # dict(trials=5, max_epochs=50, input_size=64),
        # dict(trials=5, max_epochs=50, input_size=128),
        # dict(trials=5, max_epochs=50, num_layers=1),
        # dict(trials=5, max_epochs=50, num_layers=10),
        # dict(trials=5, max_epochs=50, num_layers=100),
        dict(trials=5, max_epochs=50, num_workers=1),
        dict(trials=5, max_epochs=50, num_layers=2),
        dict(trials=5, max_epochs=50, num_workers=3),
        dict(trials=5, max_epochs=50, num_layers=8),
    ]

    results = []
    for exp in experiments:
        r = benchmark(**exp, gpus=int(gpu))
        results.append(r)
        print(r)

    print("Summary")
    for r in results:
        print(r)


"""

CPU

{'settings': {'max_epochs': 50, 'checkpointing': False, 'gpus': 0}, 'enabled': 4.936049547399999, 'disabled': 3.1311714946, 'relative': 1.5764226124032747}
{'settings': {'max_epochs': 50, 'checkpointing': True, 'gpus': 0}, 'enabled': 5.308698838400001, 'disabled': 3.405054769600002, 'relative': 1.5590641553832107}
{'settings': {'max_epochs': 50, 'input_size': 32, 'gpus': 0}, 'enabled': 5.170791892000002, 'disabled': 3.238187119, 'relative': 1.5968168922853412}
{'settings': {'max_epochs': 50, 'input_size': 64, 'gpus': 0}, 'enabled': 4.702903000800001, 'disabled': 3.1402921944000015, 'relative': 1.4976004491513755}
{'settings': {'max_epochs': 50, 'input_size': 128, 'gpus': 0}, 'enabled': 4.8013775162000005, 'disabled': 3.046343913800007, 'relative': 1.5761114477093843}
{'settings': {'max_epochs': 50, 'num_layers': 1, 'gpus': 0}, 'enabled': 4.680550298399998, 'disabled': 3.0216551166000043, 'relative': 1.549002158680041}
{'settings': {'max_epochs': 50, 'num_layers': 10, 'gpus': 0}, 'enabled': 5.5193667141999985, 'disabled': 3.8447418285999957, 'relative': 1.4355623759033496}
{'settings': {'max_epochs': 50, 'num_layers': 100, 'gpus': 0}, 'enabled': 14.518200393199994, 'disabled': 12.242298266399995, 'relative': 1.1859048094789848}

GPU

{'settings': {'max_epochs': 50, 'checkpointing': False, 'gpus': 1}, 'enabled': 9.800579199939966, 'disabled': 7.798804482072592, 'relative': 1.2566771256375164}
{'settings': {'max_epochs': 50, 'checkpointing': True, 'gpus': 1}, 'enabled': 8.576479919999837, 'disabled': 7.909054584056139, 'relative': 1.0843874990177917}
{'settings': {'max_epochs': 50, 'input_size': 32, 'gpus': 1}, 'enabled': 8.521520897746086, 'disabled': 7.7378443226218225, 'relative': 1.1012784106851519}
{'settings': {'max_epochs': 50, 'input_size': 64, 'gpus': 1}, 'enabled': 8.524269503355026, 'disabled': 7.623479735851288, 'relative': 1.1181599215470532}
{'settings': {'max_epochs': 50, 'input_size': 128, 'gpus': 1}, 'enabled': 8.451044300943613, 'disabled': 7.7648034483194355, 'relative': 1.088378393244803}
{'settings': {'max_epochs': 50, 'num_layers': 1, 'gpus': 1}, 'enabled': 8.428268907219172, 'disabled': 7.724652874469757, 'relative': 1.091087074614691}
{'settings': {'max_epochs': 50, 'num_layers': 10, 'gpus': 1}, 'enabled': 12.593942693620921, 'disabled': 11.209050392359496, 'relative': 1.1235512601678925}
{'settings': {'max_epochs': 50, 'num_layers': 100, 'gpus': 1}, 'enabled': 39.223459555208684, 'disabled': 36.51386210918427, 'relative': 1.0742073637108598

"""
