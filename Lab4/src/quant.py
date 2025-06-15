import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# Round function
class ste_round(torch.autograd.Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # For straight-through estimation, we pass the gradient straight through
        return grad_output


# Get percnetile min max
def get_percentile_min_max(
    input, lower_percentile, upper_percentile, output_tensor=False
):
    """
    Calculate the percentile max and min values in a given tensor
    Parameters:
    ----------
    input: tensor
        the tensor to calculate percentile max and min
    lower_percentile: float
        if 0.1, means we return the value of the smallest 0.1% value in the
        tensor as percentile min
    upper_percentile: float
        if 99.9, means we return the value of the largest 0.1% value in the
        tensor as percentile max
    output_tensor: bool, default False
        if True, this function returns tensors, otherwise it returns values
    """
    input_length = input.shape[0]

    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)

    upper_bound = torch.kthvalue(input, k=upper_index).values

    if lower_percentile == 0:
        lower_bound = upper_bound * 0
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def uniform_quantize(input, scale, zero_point):
    """
    Quantize floating point input tensor to integers with the given scaling
    factor and zeropoint.
    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # Quantize: First scale and zero_point the input, then round to nearest integer
    output = torch.round(input / scale + zero_point)
    return output


def get_moving_avg_min_max(
    x,
    prev_x_min,
    prev_x_max,
    act_percentile=0.0,
    act_range_momentum=0.95,
    is_symmetric=False,
):
    if act_percentile == 0.0:
        x_min = x.data.min()
        x_max = x.data.max()
    elif is_symmetric:
        x_min, x_max = get_percentile_min_max(
            x.detach().view(-1),
            100 - act_percentile,
            act_percentile,
            output_tensor=True,
        )
    # Note that our asymmetric quantization is implemented using scaled unsigned
    # integers without zero_points, that is to say our asymmetric quantization
    # should always be after ReLU, which makes the minimum value to be always 0.
    # As a result, if we use percentile mode for asymmetric quantization, the
    # lower_percentile will be set to 0 in order to make sure the final x_min is 0.
    elif not is_symmetric:
        x_min, x_max = get_percentile_min_max(
            x.detach().view(-1), 0, act_percentile, output_tensor=True
        )

    # Initialization
    if prev_x_min is None or prev_x_max is None:
        new_x_min = x_min
        new_x_max = x_max
    elif prev_x_min == prev_x_max:
        new_x_min = prev_x_min + x_min
        new_x_max = prev_x_max + x_max
    # use momentum to update the quantization range
    elif act_range_momentum == -1:
        new_x_min = min(prev_x_min, x_min)
        new_x_max = max(prev_x_max, x_max)
    else:
        new_x_min = prev_x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        new_x_max = prev_x_max * act_range_momentum + x_max * (1 - act_range_momentum)
    return new_x_min, new_x_max


# Function for symmetric quantization
class SymmetricQuantFunction(torch.autograd.Function):
    """
    Class to quantize the given floating-point values using quantization with
    given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None, specified_zero_point=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of QuantFunction requires pre-calculated scaling factor.
        The current hardware support requires quantization to use scaled unsigned integers
        without zero_point, so quantization is for activations after ReLU, and zero_point is set to 0.
        specified_scale: pre-calculated scaling factor for the tensor x
        specified_zero_point: pre-calculated zero_point for the tensor x
        """
        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError(
                "The QuantFunction requires a pre-calculated scaling factor"
            )
        ctx.scale = scale
        ctx.k = k
        ctx.save_for_backward(x)

        # For symmetric quantization, zero point should be zero.
        zero_point = 0

        # Quantize
        output = uniform_quantize(x, scale, zero_point)
        
        # Clamp to [0, 2^k - 1] for k-bit quantization
        output = torch.clamp(output, 0, 2**k - 1)
        
        # Dequantize
        output = (output - zero_point) * scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        k = ctx.k
        scale = ctx.scale
        
        # Compute gradient mask for clipping
        grad_mask = (x >= 0) & (x <= (2**k - 1) * scale)
        grad_output = grad_output * grad_mask.float()
        
        return grad_output, None, None, None


# Function for asymmetric quantization
class AsymmetricQuantFunction(torch.autograd.Function):
    """
    Class to quantize the given floating-point values using quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None, specified_zero_point=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of QuantFunction requires pre-calculated scaling factor.
        The current hardware support requires quantization to use scaled unsigned integers
        without zero_point, so quantization is for activations after ReLU, and zero_point is set to 0.
        specified_scale: pre-calculated scaling factor for the tensor x
        specified_zero_point: pre-calculated zero_point for the tensor x
        """
        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError(
                "The QuantFunction requires a pre-calculated scaling factor"
            )

        if specified_zero_point is not None:
            zero_point = specified_zero_point
        else:
            raise ValueError("The QuantFunction requires a pre-calculated zero point")

        # Save values for backward pass
        ctx.save_for_backward(x)
        ctx.scale = scale
        ctx.zero_point = zero_point
        ctx.k = k

        # Quantize
        x_scaled = x / scale + zero_point
        x_clipped = torch.clamp(x_scaled, 0, 2**k - 1)
        x_rounded = torch.round(x_clipped)
        
        # Dequantize
        output = (x_rounded - zero_point) * scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Get saved tensors and parameters
        x, = ctx.saved_tensors
        scale = ctx.scale
        zero_point = ctx.zero_point
        k = ctx.k

        # Compute gradient mask for the quantization range
        x_scaled = x / scale + zero_point
        grad_mask = (x_scaled >= 0) & (x_scaled <= 2**k - 1)
        
        # Apply the mask to the gradient
        grad_input = grad_output * grad_mask.float()
        
        # Return gradients for all inputs (x, k, scale, zero_point)
        return grad_input, None, None, None


class QConfig(object):
    """
    Type that contains configurations for quantization
    and stores calculated quantization parameters
    """

    def __init__(self, quant_bits=8, is_symmetric=True):
        self.quant_bits = quant_bits
        self.is_symmetric = is_symmetric
        self.quant_mode = "symmetric" if self.is_symmetric else "asymmetric"
        self.quantize_function = (
            SymmetricQuantFunction.apply
            if self.is_symmetric
            else AsymmetricQuantFunction.apply
        )

        # States accumulators
        self.prev_scale = None
        self.prev_zeropoint = None
        self.prev_min = None
        self.prev_max = None

    def __repr__(self):
        s = (
            f"quant_bits={self.quant_bits}, quant_mode={self.quant_mode}, "
            + f"prev_scale={self.prev_scale}, prev_zeropoint={self.prev_zeropoint}, "
            + f"prev_min={self.prev_min}, prev_max={self.prev_max} "
        )
        return s

    def get_quantization_params(self, saturation_min, saturation_max):
        """
        Calculate scale and zero point given saturation_min and saturation_max
        """
        with torch.no_grad():
            if self.is_symmetric:
                # For symmetric quantization:
                # - Use the maximum absolute value as the scaling factor
                # - Zero point is always 0
                max_abs = max(abs(saturation_min), abs(saturation_max))
                scale = max_abs / ((2**(self.quant_bits - 1)) - 1)
                zero_point = 0
            else:
                # For asymmetric quantization:
                # - Scale covers the entire range
                # - Zero point shifts the range to start from 0
                scale = (saturation_max - saturation_min) / (2**self.quant_bits - 1)
                zero_point = -torch.round(saturation_min / scale)

        return scale, zero_point

    def quantize_with_params(self, x, scale, zero_point, fake_quantize=False):
        """
        Calculate quantized value given float value, scale, and zero point
        """
        x_q = self.quantize_function(x, self.quant_bits, scale, zero_point)
        if fake_quantize:
            x_q = (x_q - zero_point) * scale
        return x_q

    def quantize_with_min_max(
        self, x, saturation_min, saturation_max, fake_quantize=False
    ):
        """
        Calculate quantized value given float value, saturation_min, and saturation_max
        """
        # Compute scale and zeropoint for quantization
        scale, zero_point = self.get_quantization_params(saturation_min, saturation_max)
        # Update and store min and max
        self.prev_min = saturation_min
        self.prev_max = saturation_max
        # Update and store computed scale and zero_point
        self.prev_scale = scale
        self.prev_zeropoint = zero_point

        x_q = self.quantize_with_params(
            x, scale, zero_point, fake_quantize=fake_quantize
        )
        return x_q

    def quantize_with_prev_params(self, x, fake_quantize=False):
        """
        Calculate quantized value using scale and zero point calculated in previous quantization
        """
        assert (
            self.prev_scale is not None and self.prev_zeropoint is not None
        ), "no params saved"
        x_q = self.quantize_with_params(
            x, self.prev_scale, self.prev_zeropoint, fake_quantize=fake_quantize
        )
        return x_q

    def copy(self):
        """
        Create a new QConfig instance with the same quant_bits and is_symmetric setting
        """
        return QConfig(self.quant_bits, self.is_symmetric)


def quantize_activations(x, qconfig, is_moving_avg=False, fake_quantize=False):
    """
    Return the quantized activations (x) calculated using given qconfig.
    Moving average method is used to calculate min and max when 'is_moving_avg'
    is set to True. 'is_moving_avg' is usually set to True during training, and
    set to False during testing and validation
    """
    x_transform = x.data.detach()
    prev_x_min, prev_x_max = qconfig.prev_min, qconfig.prev_max

    if is_moving_avg:
        x_min, x_max = get_moving_avg_min_max(
            x_transform,
            prev_x_min,
            prev_x_max,
            act_percentile=99.9,
            is_symmetric=qconfig.is_symmetric,
        )

    else:
        x_min, x_max = get_moving_avg_min_max(
            x_transform,
            None,
            None,
            act_percentile=99.9,
            is_symmetric=qconfig.is_symmetric,
        )

    # Get quantized activations and update scale, zero_point, min, and max of qconfig
    x_q = qconfig.quantize_with_min_max(x, x_min, x_max, fake_quantize=fake_quantize)
    return x_q


def quantize_weights_bias(w, qconfig, fake_quantize=False):
    """
    Quantize weights and biases.
    Parameters:
    ----------
    w: weights or biases to be quantized
    qconfig: configurations for quantization
    fake_quantize: whether to perform fake quantization
    """
    # Get min and max values
    w_min = w.data.min()
    w_max = w.data.max()

    # Calculate quantization parameters
    scale, zero_point = qconfig.get_quantization_params(w_min, w_max)

    # Store the scale and zero_point for future use
    qconfig.prev_scale = scale
    qconfig.prev_zeropoint = zero_point
    qconfig.prev_min = w_min
    qconfig.prev_max = w_max

    # Perform quantization
    w_q = qconfig.quantize_with_params(w, scale, zero_point, fake_quantize)
    return w_q


def conv2d_uniform_quantized(module, x, a_qconfig=None, w_qconfig=None, b_qconfig=None):
    """
    Quantized convolution/linear operation
    Parameters:
    ----------
    module: nn.Conv2d or nn.Linear module to be quantized
    x: input tensor
    a_qconfig: configurations for activation quantization
    w_qconfig: configurations for weight quantization
    b_qconfig: configurations for bias quantization
    """
    # Quantize weights if w_qconfig is provided
    if w_qconfig is not None:
        weight_q = quantize_weights_bias(module.weight, w_qconfig)
    else:
        weight_q = module.weight

    # Quantize bias if b_qconfig is provided and module has bias
    if module.bias is not None and b_qconfig is not None:
        bias_q = quantize_weights_bias(module.bias, b_qconfig)
    else:
        bias_q = module.bias

    # Perform convolution or linear operation with quantized weights and bias
    if isinstance(module, nn.Conv2d):
        output = F.conv2d(
            x,
            weight_q,
            bias_q,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
    else:  # Linear layer
        output = F.linear(x, weight_q, bias_q)

    # Quantize activations if a_qconfig is provided
    if a_qconfig is not None:
        output = quantize_activations(output, a_qconfig)

    return output


class QuantWrapper(nn.Module):
    def __init__(self, module, a_qconfig, w_qconfig, b_qconfig):
        super(QuantWrapper, self).__init__()
        self.a_qconfig = a_qconfig
        self.w_qconfig = w_qconfig
        self.b_qconfig = b_qconfig
        self.module = module

    def __repr__(self):
        s = super().__repr__()[:-1]
        s = (
            f"{s}"
            + f"\t(activation): {self.a_qconfig} \n"
            + f"\t(weight): {self.w_qconfig} \n"
            + f"\t(bias): {self.b_qconfig} \n)"
        )
        return s

    def forward(self, x):
        out = conv2d_uniform_quantized(
            self.module, x, self.a_qconfig, self.w_qconfig, self.b_qconfig
        )
        return out


def quantize_model(float_model, a_qconfig=None, w_qconfig=None, b_qconfig=None):
    """
    Making a float_model aware of quantization during training and inference
    """
    qat_model = copy.deepcopy(float_model)
    if a_qconfig is None or w_qconfig is None:
        return qat_model

    for name, layer in qat_model.named_children():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            quant_wrapper = QuantWrapper(
                layer, a_qconfig.copy(), w_qconfig.copy(), b_qconfig.copy()
            )
            setattr(qat_model, f"{name}", quant_wrapper)

    return qat_model


def dequantize_model(qat_model):
    """
    Remove all QuantWrappers from qat_model without updating weights and
    activation to quantized values
    """
    float_model = copy.deepcopy(qat_model)
    for name, layer in float_model.named_children():
        if isinstance(layer, QuantWrapper):
            print(f"Layer to be deprepared: QuantWrapper: {name}")
            setattr(float_model, f"{name}", layer.module)
    return float_model
