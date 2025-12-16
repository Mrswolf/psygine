__all__ = ["PatchND"]

from typing import Sequence, Tuple, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module, inplace_add
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

def _slidingsteps(nsamples: int, nwin: int, nover: int) -> Tuple[NDArray, NDArray]:
    if nwin > nsamples:
        raise ValueError(f"nwin={nwin} is bigger than nsamples={nsamples}...")
    if nover >= nwin:
        raise ValueError(f"nover={nover} is bigger than or equal to nwin={nwin}...")

    step = nwin - nover
    starts = np.arange(0, nsamples - nwin + 1, step, dtype=int)
    return starts


class PatchND(LinearOperator):
    def __init__(
        self,
        dims: InputDimsLike,
        n_window: Union[int, Sequence[int]],
        n_stride: Union[int, Sequence[int]] = 1,
        axes: Union[int, Sequence[int]] = -1,
        dtype: DTypeLike = "float64",
        name: str = "P",
    ) -> None:
        dims: Tuple[int, ...] = _value_or_sized_to_tuple(dims)
        n_window: Tuple[int, ...] = _value_or_sized_to_tuple(n_window)
        n_stride: Tuple[int, ...] = _value_or_sized_to_tuple(n_stride)
        axes: Tuple[int, ...] = _value_or_sized_to_tuple(axes)

        axes = tuple([np.mod(axis, len(dims)) for axis in axes])

        if len(n_window) == 1:
            n_window = n_window * len(axes)
        if len(n_stride) == 1:
            n_stride = n_stride * len(axes)
        if len(n_window) != len(axes):
            raise ValueError(f"n_window={n_window} does not match axes={axes}")
        if len(n_stride) != len(axes):
            raise ValueError(f"n_stride={n_stride} does not match axes={axes}")
        if len(axes) > len(dims):
            raise ValueError(f"the number of axes({axes}) is bigger than dims({dims})")

        # nwindows = [dim for dim in dims]
        nwindows = [1] * len(dims)
        # nstrides = [dim for dim in dims]
        nstrides = [1] * len(dims)

        for i, axis in enumerate(axes):
            nwindows[axis] = n_window[i]
            nstrides[axis] = n_stride[i]

        self.nwindows = nwindows
        self.nstrides = nstrides
        self.axes = axes
        self._patch_gidx = [
                    _slidingsteps(dim, window, window - stride)
                    for dim, window, stride in zip(dims, nwindows, nstrides)
                ]
        self.patch_gidx = np.stack(
            np.meshgrid(
                *self._patch_gidx,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, len(dims))
        self.patch_lidx = np.stack(
            np.meshgrid(*[np.arange(window) for window in nwindows], indexing="ij"),
            axis=-1,
        ).reshape(-1, len(dims))
        patch_idx = np.reshape(
            self.patch_gidx.reshape(-1, 1, len(dims)) + self.patch_lidx, (-1, len(dims))
        )
        self.patch_idx = np.ravel_multi_index(patch_idx.T, dims)

        dimsd = [len(pgidx) for pgidx in self._patch_gidx] + self.nwindows

        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name
        )

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.take(x, self.patch_idx)
        return y

    @reshaped
    def _rmatvec(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros(self.dims, dtype=self.dtype)
        if ncp == np:
            ncp.add.at(y.ravel(), self.patch_idx, x.ravel())
        else:
            y = inplace_add(x.ravel(), y.ravel(), self.patch_idx)
        return y
