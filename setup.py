"""Installation script for the 'wheeledlab' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "torch",
    "numpy>=1.20",  # Required for numpy.typing (used by gymnasium)
    "pyyaml",
    "toml",
    "gymnasium",
    "gymnasium[mujoco]",  # MuJoCo environments
    "tqdm",
    "wandb",
    "imageio",
    "imageio-ffmpeg",  # For video encoding
    "av",  # PyAV for video encoding
    "matplotlib",
    "scipy",
]

# Installation operation
setup(
    name="mpail2",
    packages=["mpail2", "mpail2.configs", "mpail2.utils", "gymnasium_mpail"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
