# -*- coding: utf-8 -*-
#
# Authors: swolf <swolfforever@gmail.com>
# Date: 2025/02/27
# License: MIT License
"""ADMM-CSNet.

Modified from https://github.com/yangyan92/Pytorch_ADMM-CSNet?tab=readme-ov-file
PWL is from https://github.com/PiotrDabkowski/torchpwl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ADMMCSNet",
    "ADMMCSNetOri",
]


def get_monotonicity(monotonicity, num_channels):
    if isinstance(monotonicity, (int, float)):
        if not monotonicity in (-1, 0, 1):
            raise ValueError("monotonicity must be one of -1, 0, +1")
        return monotonicity * torch.ones(num_channels)
    else:
        if not (
            isinstance(monotonicity, torch.Tensor)
            and list(monotonicity.shape) == [num_channels]
        ):
            raise ValueError(
                "monotonicity must be either an int or a tensor with shape [num_channels]"
            )
        if not torch.all(
            torch.eq(monotonicity, 0)
            | torch.eq(monotonicity, 1)
            | torch.eq(monotonicity, -1)
        ).item():
            raise ValueError("monotonicity must be one of -1, 0, +1")
        return monotonicity.float()


class BasePWL(torch.nn.Module):
    def __init__(self, num_breakpoints):
        super(BasePWL, self).__init__()
        if not num_breakpoints >= 1:
            raise ValueError(
                "Piecewise linear function only makes sense when you have 1 or more breakpoints."
            )
        self.num_breakpoints = num_breakpoints

    def slope_at(self, x):
        dx = 1e-3
        return -(self.forward(x) - self.forward(x + dx)) / dx


def calibrate1d(x, xp, yp):
    """
    x: [N, C]
    xp: [C, K]
    yp: [C, K]
    """
    x_breakpoints = torch.cat(
        [x.unsqueeze(2), xp.unsqueeze(0).repeat((x.shape[0], 1, 1))], dim=2
    )
    num_x_points = xp.shape[1]
    sorted_x_breakpoints, x_indices = torch.sort(x_breakpoints, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points),
            torch.tensor(num_x_points - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(
        torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1
    )
    start_x = torch.gather(
        sorted_x_breakpoints, dim=2, index=start_idx.unsqueeze(2)
    ).squeeze(2)
    end_x = torch.gather(
        sorted_x_breakpoints, dim=2, index=end_idx.unsqueeze(2)
    ).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points),
            torch.tensor(num_x_points - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(x.shape[0], -1, -1)
    start_y = torch.gather(
        y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)
    ).squeeze(2)
    end_y = torch.gather(
        y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)
    ).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x + 1e-7)
    return cand


class Calibrator(torch.nn.Module):
    def __init__(self, keypoints, monotonicity, missing_value=11.11):
        """
        Calibrates input to the output range of [-0.5*monotonicity, 0.5*monotonicity].
        The output is always monotonic with respect to the input.
        Recommended to use Adam for training. The calibrator is initalized as a straight line.

        value <= keypoint[0] will map to -0.5*monotonicity.
        value >= keypoint[-1] will map to 0.5*monotonicity.
        value == missing_value will map to a learnable value (within the standard output range).
        Each channel is independently calibrated and can have its own keypoints.
        Note: monotonicity and keypoints are not trainable, they remain fixed, only the calibration output at
        each keypoint is trainable.

        keypoints: tensor with shape [C, K], where K > 2
        monotonicity: tensor with shape [C]
        missing_value: float
        """
        super(Calibrator, self).__init__()
        xp = torch.tensor(keypoints, dtype=torch.float32)
        self.register_buffer("offset", xp[:, :1].clone().detach())
        self.register_buffer("scale", (xp[:, -1:] - self.offset).clone().detach())
        xp = (xp - self.offset) / self.scale
        self.register_buffer("keypoints", xp)
        self.register_buffer(
            "monotonicity", torch.tensor(monotonicity, dtype=torch.float32).unsqueeze(0)
        )
        self.missing_value = missing_value
        yp = xp[:, 1:] - xp[:, :-1]
        # [C, K - 1]
        self.yp = torch.nn.Parameter(yp, requires_grad=True)
        # [1, C]
        self.missing_y = torch.nn.Parameter(
            torch.zeros_like(xp[:, 0]).unsqueeze(0), requires_grad=True
        )

    def forward(self, x):
        """Calibrates input x tensor. x has shape [BATCH_SIZE, C]."""
        missing = torch.zeros_like(x) + torch.tanh(self.missing_y) / 2.0
        yp = torch.cumsum(torch.abs(self.yp) + 1e-9, dim=1)
        xp = self.keypoints
        last_val = yp[:, -1:]
        yp = torch.cat([torch.zeros_like(last_val), yp / last_val], dim=1)
        x_transformed = torch.clamp((x - self.offset) / self.scale, 0.0, 1.0)
        calibrated = calibrate1d(x_transformed, xp, yp) - 0.5
        return self.monotonicity * torch.where(
            x == self.missing_value, missing, calibrated
        )


class BasePWLX(BasePWL):
    def __init__(self, num_channels, num_breakpoints, num_x_points):
        super(BasePWLX, self).__init__(num_breakpoints)
        self.num_channels = num_channels
        self.num_x_points = num_x_points
        # self.x_positions = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_x_points))
        self.x_positions = torch.Tensor(self.num_channels, self.num_x_points)
        self._reset_x_points()

    def _reset_x_points(self):
        # torch.nn.init.normal_(self.x_positions, std=0.000001)
        # torch.nn.init.zeros_(self.x_positions)
        self.x_positions = (
            torch.linspace(-1, 1, self.num_x_points)
            .unsqueeze(0)
            .expand(self.num_channels, self.num_x_points)
        )

    def get_x_positions(self):
        return self.x_positions

    def get_sorted_x_positions(self):
        return torch.sort(self.get_x_positions(), dim=1)[0]

    def get_spreads(self):
        sorted_x_positions = self.get_sorted_x_positions()
        return (torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions)[
            :, :-1
        ]

    def unpack_input(self, x):
        shape = list(x.shape)
        if len(shape) == 2:
            return x
        elif len(shape) < 2:
            raise ValueError(
                "Invalid input, the input to the PWL module must have at least 2 dimensions with channels at dimension dim(1)."
            )
        assert shape[1] == self.num_channels, (
            "Invalid input, the size of dim(1) must be equal to num_channels (%d)"
            % self.num_channels
        )
        x = torch.transpose(x, 1, len(shape) - 1)
        assert x.shape[-1] == self.num_channels
        return x.reshape(-1, self.num_channels)

    def repack_input(self, unpacked, old_shape):
        old_shape = list(old_shape)
        if len(old_shape) == 2:
            return unpacked
        transposed_shape = old_shape[:]
        transposed_shape[1] = old_shape[-1]
        transposed_shape[-1] = old_shape[1]
        unpacked = unpacked.view(*transposed_shape)
        return torch.transpose(unpacked, 1, len(old_shape) - 1)


class BasePointPWL(BasePWLX):
    def get_y_positions(self):
        raise NotImplementedError()

    def forward(self, x):
        old_shape = x.shape
        x = self.unpack_input(x)
        cand = calibrate1d(x, self.get_x_positions(), self.get_y_positions())
        return self.repack_input(cand, old_shape)


class PointPWL(BasePointPWL):
    def __init__(self, num_channels, num_breakpoints):
        super(PointPWL, self).__init__(
            num_channels, num_breakpoints, num_x_points=num_breakpoints + 1
        )
        self.y_positions = torch.nn.Parameter(
            torch.Tensor(self.num_channels, self.num_x_points)
        )
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        with torch.no_grad():
            self.y_positions.copy_(self.get_sorted_x_positions())

    def get_x_positions(self):
        return self.x_positions

    def get_y_positions(self):
        return self.y_positions


class MonoPointPWL(BasePointPWL):
    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(MonoPointPWL, self).__init__(
            num_channels, num_breakpoints, num_x_points=num_breakpoints + 1
        )
        self.y_starts = torch.nn.Parameter(torch.Tensor(self.num_channels))
        self.y_deltas = torch.nn.Parameter(
            torch.Tensor(self.num_channels, self.num_breakpoints)
        )
        self.register_buffer(
            "monotonicity", get_monotonicity(monotonicity, num_channels)
        )
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        with torch.no_grad():
            sorted_x_positions = self.get_sorted_x_positions()
            mono_mul = torch.where(
                torch.eq(self.monotonicity, 0.0),
                torch.tensor(1.0, device=self.monotonicity.device),
                self.monotonicity,
            )
            self.y_starts.copy_(sorted_x_positions[:, 0] * mono_mul)
            spreads = self.get_spreads()
            self.y_deltas.copy_(spreads * mono_mul.unsqueeze(1))

    def get_x_positions(self):
        return self.x_positions

    def get_y_positions(self):
        starts = self.y_starts.unsqueeze(1)
        deltas = torch.where(
            torch.eq(self.monotonicity, 0.0).unsqueeze(1),
            self.y_deltas,
            torch.abs(self.y_deltas) * self.monotonicity.unsqueeze(1),
        )
        return torch.cat([starts, starts + torch.cumsum(deltas, dim=1)], dim=1)


class BaseSlopedPWL(BasePWLX):
    def get_biases(self):
        raise NotImplementedError()

    def get_slopes(self):
        raise NotImplementedError()

    def forward(self, x):
        old_shape = x.shape
        x = self.unpack_input(x)
        bs = x.shape[0]
        sorted_x_positions = self.get_sorted_x_positions().to(device=x.device)
        skips = torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions
        slopes = self.get_slopes()
        skip_deltas = skips * slopes[:, 1:]
        biases = self.get_biases().unsqueeze(1)
        cumsums = torch.cumsum(skip_deltas, dim=1)[:, :-1]

        betas = torch.cat([biases, biases, cumsums + biases], dim=1)
        breakpoints = torch.cat(
            [sorted_x_positions[:, 0].unsqueeze(1), sorted_x_positions], dim=1
        )

        # find the index of the first breakpoint smaller than x
        # TODO(pdabkowski) improve the implementation
        s = x.unsqueeze(2) - sorted_x_positions.unsqueeze(0)
        # discard larger breakpoints
        s = torch.where(s < 0, torch.tensor(float("inf"), device=x.device), s)
        b_ids = torch.where(
            sorted_x_positions[:, 0].unsqueeze(0) <= x,
            torch.argmin(s, dim=2) + 1,
            torch.tensor(0, device=x.device),
        ).unsqueeze(2)

        selected_betas = torch.gather(
            betas.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        selected_breakpoints = torch.gather(
            breakpoints.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        selected_slopes = torch.gather(
            slopes.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        cand = selected_betas + (x - selected_breakpoints) * selected_slopes
        return self.repack_input(cand, old_shape)


class PWL(BaseSlopedPWL):
    r"""Piecewise Linear Function (PWL) module.

    The module takes the Tensor of (N, num_channels, ...) shape and returns the processed Tensor of the same shape.
    Each entry in the input tensor is processed by the PWL function. There are num_channels separate PWL functions,
    the PWL function used depends on the channel.

    The x coordinates of the breakpoints are initialized randomly from the Gaussian with std of 2. You may want to
    use your own custom initialization depending on the use-case as the optimization is quite sensitive to the
    initialization of breakpoints. As long as your data is normalized (zero mean, unit variance) the default
    initialization should be fine.

    Arguments:
        num_channels (int): number of channels (or features) that this PWL should process. Each channel
            will get its own PWL function.
        num_breakpoints (int): number of PWL breakpoints. Total number of segments constructing the PWL is
            given by num_breakpoints + 1. This value is shared by all the PWL channels in this module.
    """

    def __init__(self, num_channels, num_breakpoints):
        super(PWL, self).__init__(
            num_channels, num_breakpoints, num_x_points=num_breakpoints
        )
        self.slopes = torch.nn.Parameter(
            torch.Tensor(self.num_channels, self.num_breakpoints + 1)
        )
        self.biases = torch.nn.Parameter(torch.Tensor(self.num_channels))
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        torch.nn.init.ones_(self.slopes)
        self.slopes.data[:, : (self.num_breakpoints + 1) // 2] = 0.0
        print()
        with torch.no_grad():
            self.biases.copy_(torch.zeros_like(self.biases))

    def get_biases(self):
        return self.biases

    def get_x_positions(self):
        return self.x_positions

    def get_slopes(self):
        return self.slopes


class MonoPWL(PWL):
    r"""Piecewise Linear Function (PWL) module with the monotonicity constraint.

    The module takes the Tensor of (N, num_channels, ...) shape and returns the processed Tensor of the same shape.
    Each entry in the input tensor is processed by the PWL function. There are num_channels separate PWL functions,
    the PWL function used depends on the channel. Each PWL is guaranteed to have the requested monotonicity.

    The x coordinates of the breakpoints are initialized randomly from the Gaussian with std of 2. You may want to
    use your own custom initialization depending on the use-case as the optimization is quite sensitive to the
    initialization of breakpoints. As long as your data is normalized (zero mean, unit variance) the default
    initialization should be fine.

    Arguments:
        num_channels (int): number of channels (or features) that this PWL should process. Each channel
            will get its own PWL function.
        num_breakpoints (int): number of PWL breakpoints. Total number of segments constructing the PWL is
            given by num_breakpoints + 1. This value is shared by all the PWL channels in this module.
        monotonicity (int, Tensor): Monotonicty constraint, the monotonicity can be either +1 (increasing),
            0 (no constraint) or -1 (decreasing). You can provide either an int to set the constraint
            for all the channels or a long Tensor of shape [num_channels]. All the entries must be in -1, 0, +1.
    """

    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(MonoPWL, self).__init__(
            num_channels=num_channels, num_breakpoints=num_breakpoints
        )
        self.register_buffer(
            "monotonicity", get_monotonicity(monotonicity, self.num_channels)
        )
        with torch.no_grad():
            mono_mul = torch.where(
                torch.eq(self.monotonicity, 0.0),
                torch.tensor(1.0, device=self.monotonicity.device),
                self.monotonicity,
            )
            self.biases.copy_(self.biases * mono_mul)

    def get_slopes(self):
        return torch.where(
            torch.eq(self.monotonicity, 0.0).unsqueeze(1),
            self.slopes,
            torch.abs(self.slopes) * self.monotonicity.unsqueeze(1),
        )


class ADMMCSNetOri(nn.Module):
    def __init__(
        self,
        mask,
        in_channels=1,
        out_channels=128,
        kernel_size=5,
        num_breakpoints=101,
    ):
        super().__init__()

        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.re_org_layer = ReconstructionOriginalLayer(self.rho, self.mask)
        self.conv1_layer = ConvolutionLayer1(in_channels, out_channels, kernel_size)
        self.nonlinear_layer = NonlinearLayer(out_channels, num_breakpoints)
        self.conv2_layer = ConvolutionLayer2(out_channels, in_channels, kernel_size)
        self.min_layer = MinusLayer()
        self.multiple_org_layer = MultipleOriginalLayer(self.gamma)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho, self.mask)
        self.add_layer = AdditionalLayer()
        self.multiple_update_layer = MultipleUpdateLayer(self.gamma)
        self.re_final_layer = ReconstructionFinalLayer(self.rho, self.mask)
        layers = []

        layers.append(self.re_org_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_org_layer)

        for i in range(8):
            layers.append(self.re_update_layer)
            layers.append(self.add_layer)
            layers.append(self.conv1_layer)
            layers.append(self.nonlinear_layer)
            layers.append(self.conv2_layer)
            layers.append(self.min_layer)
            layers.append(self.multiple_update_layer)

        layers.append(self.re_update_layer)
        layers.append(self.add_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_update_layer)

        layers.append(self.re_final_layer)

        self.cs_net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1_layer.conv.weight = torch.nn.init.normal_(
            self.conv1_layer.conv.weight, mean=0, std=1
        )
        self.conv2_layer.conv.weight = torch.nn.init.normal_(
            self.conv2_layer.conv.weight, mean=0, std=1
        )
        self.conv1_layer.conv.weight.data = self.conv1_layer.conv.weight.data * 0.025
        self.conv2_layer.conv.weight.data = self.conv2_layer.conv.weight.data * 0.025

    def forward(self, x):
        y = torch.mul(x, self.mask)
        x = self.cs_net(y)
        x = torch.fft.ifft2(y + (1 - self.mask) * torch.fft.fft2(x))
        return x


# reconstruction original layers
class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionOriginalLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        mask = self.mask
        denom = torch.add(mask.to(device=x.device), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).to(device=x.device)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)

        orig_output2 = torch.mul(x, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        # define data dict
        cs_data = dict()
        cs_data["input"] = x
        cs_data["conv1_input"] = orig_output3
        return cs_data


# reconstruction middle layers
class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionUpdateLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x["minus_output"]
        multiple_output = x["multi_output"]
        input = x["input"]
        mask = self.mask
        number = torch.add(
            input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output))
        )
        denom = torch.add(mask.to(device=input.device), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).to(device=input.device)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x["re_mid_output"] = orig_output3
        return x


# reconstruction middle layers
class ReconstructionFinalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x["minus_output"]
        multiple_output = x["multi_output"]
        input = x["input"]
        mask = self.mask
        number = torch.add(
            input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output))
        )
        denom = torch.add(mask.to(device=input.device), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).to(device=input.device)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x["re_final_output"] = orig_output3
        return x["re_final_output"]


# multiple original layer
class MultipleOriginalLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleOriginalLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        org_output = x["conv1_input"]
        minus_output = x["minus_output"]
        output = torch.mul(self.gamma, torch.sub(org_output, minus_output))
        x["multi_output"] = output
        return x


# multiple middle layer
class MultipleUpdateLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleUpdateLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        multiple_output = x["multi_output"]
        re_mid_output = x["re_mid_output"]
        minus_output = x["minus_output"]
        output = torch.add(
            multiple_output,
            torch.mul(self.gamma, torch.sub(re_mid_output, minus_output)),
        )
        x["multi_output"] = output
        return x


# convolution layer
class ConvolutionLayer1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=1,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        conv1_input = x["conv1_input"]
        real = self.conv(conv1_input.real)
        imag = self.conv(conv1_input.imag)
        output = torch.complex(real, imag)
        x["conv1_output"] = output
        return x


# convolution layer
class ConvolutionLayer2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=1,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        nonlinear_output = x["nonlinear_output"]
        real = self.conv(nonlinear_output.real)
        imag = self.conv(nonlinear_output.imag)
        output = torch.complex(real, imag)

        x["conv2_output"] = output
        return x


# nonlinear layer
class NonlinearLayer(nn.Module):
    def __init__(self, in_channels, num_breakpoints):
        super(NonlinearLayer, self).__init__()
        self.pwl = PWL(num_channels=in_channels, num_breakpoints=num_breakpoints)

    def forward(self, x):
        conv1_output = x["conv1_output"]
        y_real = self.pwl(conv1_output.real)
        y_imag = self.pwl(conv1_output.imag)
        output = torch.complex(y_real, y_imag)
        x["nonlinear_output"] = output
        return x


# minus layer
class MinusLayer(nn.Module):
    def __init__(self):
        super(MinusLayer, self).__init__()

    def forward(self, x):
        minus_input = x["conv1_input"]
        conv2_output = x["conv2_output"]
        output = torch.sub(minus_input, conv2_output)
        x["minus_output"] = output
        return x


# addtional layer
class AdditionalLayer(nn.Module):
    def __init__(self):
        super(AdditionalLayer, self).__init__()

    def forward(self, x):
        mid_output = x["re_mid_output"]
        multi_output = x["multi_output"]
        output = torch.add(mid_output, multi_output)
        x["conv1_input"] = output
        return x


class EfficientPWL(nn.Module):
    def __init__(self, in_channels, num_cpnts):
        super().__init__()
        self.num_cpnts = num_cpnts
        self.register_buffer("delta_t", torch.tensor([2 / (num_cpnts - 1)]))
        self.register_buffer("cpnts", torch.linspace(-1, 1, steps=num_cpnts))
        self.slopes = nn.Parameter(
            torch.Tensor(in_channels, num_cpnts + 1), requires_grad=True
        )
        self.biases = nn.Parameter(torch.Tensor(in_channels), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        # torch.nn.init.normal_(self.slopes)
        # torch.nn.init.normal_(self.biases)
        torch.nn.init.ones_(self.slopes)
        self.slopes.data[:, : (self.num_cpnts + 1) // 2] = 0.0
        torch.nn.init.zeros_(self.biases)

    def forward(self, x):
        with torch.no_grad():
            ind1 = torch.searchsorted(self.cpnts, x, side="right")
            ind2 = ind1 - 1
            ind2[ind2 < 0] = 0

        cumbias = torch.cumsum(self.slopes[:, 1:-1] * self.delta_t, dim=1)
        cumbias = torch.cat(
            [self.biases.view(-1, 1), cumbias + self.biases.view(-1, 1)], dim=1
        )

        cpnts = self.cpnts[ind2]
        slopes = torch.gather(
            self.slopes.unsqueeze(0).expand(x.shape[0], -1, -1),
            2,
            ind1.view(*x.shape[:2], -1),
        ).view(*x.shape)
        biases = torch.gather(
            cumbias.unsqueeze(0).expand(x.shape[0], -1, -1),
            2,
            ind2.view(*x.shape[:2], -1),
        ).view(*x.shape)
        x = (x - cpnts) * slopes + biases
        return x


class ReconLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rho = nn.Parameter(
            torch.tensor([0.1], dtype=torch.float32), requires_grad=True
        )

    def forward(self, mask, y, z=None, beta=None):
        """
        Parameters:
            mask: (H, W)
            y: (B, 1, H, W)
            z: (B, 1, H, W)
            beta: (B, 1, H, W)

        Returns:
            x: (B, 1, H, W)
        """
        if z is not None or beta is not None:
            b = mask * y + self.rho * torch.fft.fft2(z - beta)
        else:
            b = mask * y

        denom = torch.add(mask, self.rho)  # rho*I + S, where S is the smapling mask
        # avoid division by zero
        # (rho*I + S)^-1 * S^H*y
        x = torch.where(denom != 0, b / denom, 0)
        # F^H*(rho*I + S)^-1 * S^H*y
        x = torch.fft.ifft2(x)
        return x


class MultiplierLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, x, z, beta=None):
        """
        Parameters:
            x: (B, 1, H, W)
            z: (B, 1, H, W)
            beta: (B, 1, H, W)
        """
        if beta is not None:
            return beta + self.gamma * (x - z)
        else:
            return self.gamma * (x - z)


class ProximalLayer(nn.Module):
    def __init__(self, in_chan, L, kernel_size, num_breakpoints=101, Nt=1):
        super().__init__()
        self.Nt = Nt
        self.miu1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.conv1 = nn.Conv2d(
            in_chan,
            L,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv2d(
            L,
            in_chan,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.nonlinear_layer = EfficientPWL(L, num_breakpoints)

    def forward(self, z):
        z0 = z.clone()

        for _ in range(self.Nt):
            real = self.conv1(z.real)
            real = self.nonlinear_layer(real)
            real = self.conv2(real)

            imag = self.conv1(z.imag)
            imag = self.nonlinear_layer(imag)
            imag = self.conv2(imag)

            c = torch.complex(real, imag)
            z = self.miu1 * z + (1 - self.miu1) * z0 - c

        return z


class ADMMCSNet(nn.Module):

    def __init__(self, mask, L, kernel_size=5, num_breakpoints=101, Nt=1, Ns=9):
        super().__init__()
        self.Ns = Ns
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))

        self.proximal_layer = ProximalLayer(
            1, L, kernel_size, num_breakpoints=num_breakpoints, Nt=Nt
        )
        self.recon_layer = ReconLayer()
        self.multiplier_layer = MultiplierLayer()
        self.reset_parameters()

    def reset_parameters(self):
        self.proximal_layer.conv1.weight = nn.init.normal_(
            self.proximal_layer.conv1.weight, mean=0, std=1
        )
        self.proximal_layer.conv2.weight = nn.init.normal_(
            self.proximal_layer.conv2.weight, mean=0, std=1
        )
        self.proximal_layer.conv1.weight.data = (
            self.proximal_layer.conv1.weight.data * 0.025
        )
        self.proximal_layer.conv2.weight.data = (
            self.proximal_layer.conv2.weight.data * 0.025
        )

    def forward(self, y):
        x = self.recon_layer(self.mask, y)
        z = self.proximal_layer(x)
        beta = self.multiplier_layer(x, z)

        for _ in range(self.Ns):
            x = self.recon_layer(self.mask, y, z, beta)
            z = self.proximal_layer(x + beta)
            beta = self.multiplier_layer(x, z, beta)

        x = self.recon_layer(self.mask, y, z, beta)
        x = torch.fft.ifft2(y + (1 - self.mask) * torch.fft.fft2(x))
        return x
