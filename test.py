import torch

from pytorch_lightning.utilities.meta import init_meta, init_meta_context


class BaseModule(torch.nn.Module):
    pass


class MyModule(BaseModule):
    def __init__(self):
        super().__init__()
        self.nn = torch.nn.Linear(in_features=1, out_features=1)


with init_meta_context():
    my_module = torch.nn.Linear(in_features=1, out_features=1)

    breakpoint()

    my_module = torch.nn.Linear(in_features=1, out_features=2)

    my_module.materialize()
    print(id(my_module))
    assert isinstance(my_module, torch.nn.Linear)
    print(my_module.weight)
