import torch

from pytorch_lightning.utilities.meta import init_meta_context


class BaseModule(torch.nn.Module):
    pass


class MyModule(BaseModule):
    def __init__(self):
        super().__init__()
        self.lins = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=1, out_features=1), torch.nn.Linear(in_features=1, out_features=2)]
        )


with init_meta_context():
    my_module = MyModule()
    assert isinstance(my_module, MyModule)
    assert my_module.lins[0].weight.device.type == "meta"
    assert my_module.lins[1].weight.shape == torch.Size([2, 1])

my_module.materialize()
assert my_module.lins[0].weight.device.type == "cpu"
