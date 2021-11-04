# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import threading
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterator, Optional, Set, Type

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.modules.container import ModuleDict, ModuleList, Sequential

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_10

if _TORCH_GREATER_EQUAL_1_10:
    from torch._C import _DisableTorchDispatch  # type: ignore[attr-defined]

    ####################################################################
    # BELOW: TAKEN FROM https://github.com/pytorch/pytorch/pull/66317. #
    # TODO: Removed once merged and released on PyTorch side           #
    ####################################################################

    @contextmanager
    def enable_python_mode(cls) -> Iterator[None]:
        if not hasattr(cls, "__torch_dispatch__"):
            raise ValueError("The class passed to enable_python_mode " "must have a __torch_dispatch__ classmethod")
        if not isinstance(cls, type) or not issubclass(cls, (torch.Tensor,)):
            raise ValueError("The argument passed to enable_python_mode " "must be the type of a Tensor subclass")
        torch._C._enter_python_mode(cls)
        try:
            yield
        finally:
            torch._C._exit_python_mode()

    _tls = threading.local()
    _tls.in_call = False

    @contextmanager
    def _no_dispatch() -> Iterator[None]:
        """Temporarily disables the Python dispatch mode."""
        guard = _DisableTorchDispatch()  # noqa F841
        try:
            yield
        finally:
            del guard

    def _handle_arange(func, args, kwargs):
        kwargs["device"] = torch.device("cpu")
        return torch.empty_like(func(*args, **kwargs), device="meta")

    def _handle_tril(func, args, kwargs):
        if args and isinstance(args[0], Tensor):
            return torch.empty_like(args[0], device="meta")

        return NotImplemented

    class _MetaContext(Tensor):
        _op_handlers: Dict[Callable, Callable] = {}

        @classmethod
        def _ensure_handlers_initialized(cls) -> None:
            if cls._op_handlers:
                return

            cls._op_handlers.update(
                {
                    torch.ops.aten.arange: _handle_arange,
                    torch.ops.aten.tril: _handle_tril,
                }
            )

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            cls._ensure_handlers_initialized()

            op_handler: Optional[Callable]

            try:
                op_handler = cls._op_handlers[func]
            except KeyError:
                op_handler = None

            with _no_dispatch():
                if op_handler:
                    result = op_handler(func, args, kwargs)
                    if result is not NotImplemented:
                        return result

                if "device" in kwargs:
                    kwargs["device"] = torch.device("meta")

                obj = func(*args, **kwargs)
                print(obj)
                return obj

    def init_meta(module_fn: Callable[..., Module], *args, **kwargs) -> Module:
        def create_instance(args=None, module=None, kwargs=None) -> Module:
            if module:
                # module.__init__(*args, **kwargs)
                return module
            return module_fn(*args, **kwargs)

        if _tls.in_call:
            module = create_instance(args=(), kwargs={})
        else:
            _tls.in_call = True
            try:
                with enable_python_mode(_MetaContext):
                    module = create_instance(args=(), kwargs={})
            finally:
                _tls.in_call = False

        module.materialize = partial(
            create_instance, module=module, args=args, kwargs=kwargs)  # type: ignore[assignment]

        return module

    def is_meta_init() -> bool:
        """Indicates whether the module is being instantiated by ``init_meta()``."""
        return _tls.in_call

    ####################################################################
    # ABOVE: TAKEN FROM https://github.com/pytorch/pytorch/pull/66317. #
    # TODO: Removed once merged and released on PyTorch side           #
    ####################################################################

else:

    def init_meta(*_, **__):
        if not _TORCH_GREATER_EQUAL_1_10:
            return MisconfigurationException("`init_meta` is supported from PyTorch 1.10.0")


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls: Type[nn.Module]) -> Set[nn.Module]:
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


def recursively_setattr(root_module: nn.Module, prefix: str, materialized_module: nn.Module) -> None:
    *path, name = prefix.split(".")
    for p in path:
        root_module = getattr(root_module, p)

    try:
        index = int(name)
        root_module[index] = materialized_module
    except ValueError:
        setattr(root_module, name, materialized_module)


def materialize_module(root_module: nn.Module) -> nn.Module:
    """This utility performs an in-place operation by materialize a module and its children."""
    if not _TORCH_GREATER_EQUAL_1_10:
        return root_module

    materialize_fn = getattr(root_module, "materialize", None)
    if materialize_fn and not isinstance(root_module, (Sequential, ModuleList, ModuleDict)):
        return materialize_fn()

    for name, child in root_module.named_children():
        materialize_fn = getattr(child, "materialize", None)
        if not materialize_fn or isinstance(child, (Sequential, ModuleList, ModuleDict)):
            materialize_module(child)
        else:
            setattr(child, name, materialize_fn())
    return root_module


# cache subclasses to optimize the search when resetting the meta device later on.
__STORAGE_META__ = {}

__CREATED_MODULES__ = set()


def _unset_meta_device(from_created: bool = False) -> None:
    """Replace all meta module by their original version."""
    if not _TORCH_GREATER_EQUAL_1_10:
        raise MisconfigurationException("`init_meta` is supported from PyTorch 1.10.0")

    if from_created:
        values = [__STORAGE_META__[key] for key in __CREATED_MODULES__]
    else:
        values = __STORAGE_META__.values()

    for mods, subclass, _ in values:
        for mod in mods:
            setattr(mod, subclass.__name__, subclass)


def _set_meta_device_populated(from_created: bool = False) -> None:
    """Replace all meta module by their original version."""
    if not _TORCH_GREATER_EQUAL_1_10:
        raise MisconfigurationException("`init_meta` is supported from PyTorch 1.10.0")

    if from_created:
        values = [__STORAGE_META__[key] for key in __CREATED_MODULES__]
    else:
        values = __STORAGE_META__.values()

    for mods, subclass, meta_class in values:
        for mod in mods:
            setattr(mod, subclass.__name__, meta_class)


def _wrap_init(f: Callable, x=None, y=None) -> Callable:
    @functools.wraps(f)
    def wrapper(module: Any, *args, **kwargs) -> None:
        nonlocal x
        nonlocal y
        # f(module, *x, **y)

    return wrapper


def _set_meta_device() -> None:
    """Replace all torch.nn.Module by their meta replacement."""

    if not _TORCH_GREATER_EQUAL_1_10:
        raise MisconfigurationException("`init_meta` is supported from PyTorch 1.10.0")

    # Author note: This can be optimized further by searching all subclasses at once.
    # Its time complexity is O(n*m) where n is the number of all subclasses if there's no multiple inheritance
    # and m the number of all subclasses belonging to its subclass module.

    for subclass in get_all_subclasses(torch.nn.modules.module.Module):

        if isinstance(subclass, (Sequential, ModuleList, ModuleDict)):
            continue

        # Create a class subclassing current `subclass` overriding its new method.
        # this will enable use to use `torch.distributed.nn.utils.init_meta` to create a `meta`
        # version of the current subclass module
        class _MetaClass(subclass):
            def __new__(cls, *args, **kwargs):
                meta_new = cls.__new__
                cls.__new__ = cls._class_new
                ori_init = cls.__init__
                cls.__init__ = _wrap_init(cls.__init__, x=args, y=kwargs)
                obj = init_meta(cls, *args, **kwargs)
                print(id(obj))
                cls.__init__ = ori_init
                cls.__new__ = meta_new
                return obj

        if not hasattr(subclass, "_class_new"):
            subclass._class_new = subclass.__new__
            subclass.__new__ = _MetaClass.__new__


@contextmanager
def init_meta_context() -> Generator:
    rank_zero_warn(
        "Be aware this feature is highly experimental and there are a number of weird edge cases "
        "where it can internal assert and/or crash. A more stable version is to be expected from PyTorch 1.11."
    )
    _set_meta_device()
    yield
    _unset_meta_device()
