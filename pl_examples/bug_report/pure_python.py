import torch
import torch.distributed
import os


class BoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.batch_norm = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.batch_norm(self.layer(x))


def run(local_rank):
    device = torch.device("cuda", local_rank)
    model = BoringModel()

    torch.distributed.init_process_group("gloo", world_size=2, rank=local_rank)
    torch.cuda.set_device(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    script = torch.jit.script(model)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    run(local_rank)
