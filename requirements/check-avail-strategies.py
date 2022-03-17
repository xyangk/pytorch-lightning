import bagua  # noqa: F401
import deepspeed  # noqa: F401
import fairscale  # noqa: F401
import horovod.torch

# TODO(akihironitta): Uncomment once the horovod installation issue is resolved (#12314)
# returns an error code
# assert horovod.torch.nccl_built()
