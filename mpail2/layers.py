import torch
import torch.nn as nn
from typing import Callable, List


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    elif act_name == "mish":
        return torch.nn.Mish()
    elif act_name == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def mlp_factory(
    *,
    input_dim: int,
    hidden_dims: List[int],
    activation: Callable,
    output_dim: int,
    use_layer_norm: bool = False,
    disable_output_bias: bool = False,
    override_last_layer_norm: bool = False,
    override_last_layer_activation: bool = False,
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.nn.Sequential:
    activation = resolve_nn_activation(activation)
    layers = []

    def add_layer(*, layers, layer, use_layer_norm, layer_dim=None):
        layers.append(layer)
        if use_layer_norm and layer_dim is not None:
            layers.append(nn.LayerNorm(layer_dim))

    # Input layer
    add_layer(
        layers=layers, layer=nn.Linear(input_dim, hidden_dims[0]),
        use_layer_norm=use_layer_norm, layer_dim=hidden_dims[0]
    )
    layers.append(activation)

    # Hidden layers
    for layer_index in range(len(hidden_dims)):
        is_last = layer_index == len(hidden_dims) - 1
        out_dim = output_dim if is_last else hidden_dims[layer_index + 1]
        # Don't apply normalization to output layer unless override_last_layer_norm is True
        add_layer(
            layers=layers,
            layer=nn.Linear(
                hidden_dims[layer_index], out_dim,
                bias=not (is_last and disable_output_bias)
            ),
            use_layer_norm=(not is_last and use_layer_norm) or (is_last and override_last_layer_norm),
            layer_dim=out_dim
        )
        if not is_last:
            layers.append(activation)

    if override_last_layer_activation:
        layers.append(activation)

    model = nn.Sequential(*layers)
    model.to(device=device, dtype=dtype)
    return model


def cnn_factory(
    *,
    H: int,
    W: int,
    C: int,
    output_dim: int,
    out_channels: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    activation: str = "silu",
    hidden_dims: List[int] = [],
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
    override_last_layer_norm: bool = False,
    override_last_layer_activation: bool = False,
    **kwargs
) -> torch.nn.Sequential:

    num_layers = len(kernel_sizes)
    assert num_layers == len(strides), "kernel_sizes and strides must have the same length"

    layers = []

    # Convolutional layers
    in_channels = C
    for i in range(num_layers):
        kernel_size = kernel_sizes[i]
        stride = strides[i]
        padding = 0

        layers.append(nn.Conv2d(in_channels, out_channels[i], kernel_size, stride, padding))

        layers.append(resolve_nn_activation(activation))

        in_channels = out_channels[i]

    layers.append(nn.Flatten())

    # Compute the size of the flattened feature map
    with torch.no_grad():
        dummy_input = torch.zeros(1, C, H, W)
        dummy_output = nn.Sequential(*layers)(dummy_input)
        flattened_size = dummy_output.shape[1]

    in_dim = flattened_size
    for i, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, dim))
        layers.append(resolve_nn_activation(activation))
        in_dim = dim

    # Final linear layer to project to output_dim
    layers.append(nn.Linear(in_dim, output_dim))

    if override_last_layer_norm:
        layers.append(nn.LayerNorm(output_dim))

    if override_last_layer_activation:
        layers.append(resolve_nn_activation(activation))

    model = nn.Sequential(*layers)
    model.to(device=device, dtype=dtype)
    return model

def identity_factory(
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.nn.Identity:
    model = nn.Identity()
    model.to(device=device, dtype=dtype)
    return model