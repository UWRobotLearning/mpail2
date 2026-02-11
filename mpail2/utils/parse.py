import torch
import importlib
import os
import yaml

from typing import Type

def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data

def resolve_optim(opt: str):
    opt_lower = opt.lower()
    if opt_lower == "adam":
        return torch.optim.Adam
    elif opt_lower == "adamw":
        return torch.optim.AdamW
    elif opt_lower == "sgd":
        return torch.optim.SGD
    elif opt_lower == "rmsprop":
        return torch.optim.RMSprop
    else:
        raise ValueError(f"Invalid optimizer: {opt}")

# Resolve classes using module if not already a class type
def resolve_obj(class_key: str) -> type:
    """
    Resolve a class from a configuration dictionary.

    Args:
        cfg (dict): Configuration dictionary.
        class_key (str): Key in the configuration dictionary that contains the class name.
        module (str): Module name where the class is located.

    Returns:
        type: Resolved class type.
    """
    if isinstance(class_key, str):
        return resolve_from_str(class_key)

    return class_key

def resolve_from_str(full_path: str):
    """
    Resolves a class or function from a module using a string in the format
    'module.submodule...finalmodule:class_or_func_name' and instantiates or calls it.

    Args:
        full_path (str): The full path string (e.g., 'module.submodule...finalmodule:class_or_func_name').
        *args: Positional arguments to pass to the class constructor or function.
        **kwargs: Keyword arguments to pass to the class constructor or function.

    Returns:
        object: The instantiated class or the result of the function call.
    """
    # Split the string into module path and class/function name
    module_path, class_or_func_name = full_path.split(":")

    # Import the module dynamically
    module = importlib.import_module(module_path)

    # Resolve the class or function
    resolved = getattr(module, class_or_func_name)

    # If it's a class, instantiate it; if it's a function, call it
    if callable(resolved):
        return resolved
    else:
        raise TypeError(f"{class_or_func_name} is not callable in module {module_path}")


def dataclass_from_yaml_recurse(cls, yaml):
    '''
    Resolves a (potentially hierarchical) dataclass from a yaml file.
    The fields must be correct in the given class.
    Args:
        cls: The dataclass to resolve.
        yaml: The yaml file to resolve from.
    Returns:
        The resolved dataclass.
    '''
    # Check if the class is a dataclass
    if hasattr(cls, "__dataclass_fields__"):
        # Create a dictionary to hold the resolved values
        resolved_dict = {}
        for field_name, field_type in cls.__dataclass_fields__.items():
            # Check if the field type is a dataclass
            if hasattr(field_type.type, "__dataclass_fields__"):
                # Recursively resolve the dataclass
                resolved_dict[field_name] = dataclass_from_yaml_recurse(field_type.type, yaml[field_name])
            else:
                # Resolve the field value from the yaml file
                resolved_dict[field_name] = yaml[field_name]
        return cls(**resolved_dict)
    else:
        return cls(yaml)