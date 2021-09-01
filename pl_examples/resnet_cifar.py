import os
import sys
import time

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from pl_examples import _DATASETS_PATH, cli_lightning_logo
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI

DEFAULT_CMD_LINE = (
    "--trainer.max_epochs=1",
    "--trainer.limit_train_batches=15",
    "--trainer.limit_val_batches=15",
    "--trainer.profiler=pytorch",
    f"--trainer.gpus={int(torch.cuda.is_available())}",
)


class ModelToProfile(LightningModule):
    def __init__(self, name: str = "resnet50"):
        super().__init__()
        self.model = getattr(models, name)(pretrained=True)
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        inputs = batch[0]
        return self.model(inputs)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, num_workers):
        super().__init__()
        self.num_workers = num_workers

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def train_dataloader(self, *args, **kwargs):
        trainset = torchvision.datasets.CIFAR10(
            root=_DATASETS_PATH, train=True, download=True, transform=self.transform
        )
        return torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    def val_dataloader(self, *args, **kwargs):
        valset = torchvision.datasets.CIFAR10(root=_DATASETS_PATH, train=False, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=0)


def run(enabled=False, max_epochs=20, checkpointing=False, num_workers=0, **trainer_args):
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = str(int(enabled))
    start = time.monotonic()

    dm = CIFAR10DataModule(num_workers=num_workers)
    model = ModelToProfile()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        logger=False,
        checkpoint_callback=checkpointing,
        limit_train_batches=10,
        **trainer_args,
    )
    trainer.fit(model, datamodule=dm)

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
        # dict(trials=5, max_epochs=2, checkpointing=False),
        # dict(trials=5, max_epochs=2, checkpointing=True),
        dict(trials=5, max_epochs=2, num_workers=8),
        dict(trials=5, max_epochs=2, num_workers=3),
        dict(trials=5, max_epochs=2, num_workers=2),
        dict(trials=5, max_epochs=2, num_workers=1),
        dict(trials=5, max_epochs=2, num_workers=0),
    ]

    results = []
    for exp in experiments:
        r = benchmark(**exp, gpus=int(gpu))
        results.append(r)
        print(r)

    print("Summary")
    for r in results:
        print(r)

# if __name__ == "__main__":
#    dm, model, trainer = cli_main()
#
# def cli_main():
#     if len(sys.argv) == 1:
#         sys.argv += DEFAULT_CMD_LINE
#
#     LightningCLI(ModelToProfile, CIFAR10DataModule)


# if __name__ == "__main__":
#     cli_lightning_logo()
#     cli_main()
