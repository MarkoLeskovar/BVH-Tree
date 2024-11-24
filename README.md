# AABB-Tree

> Axis-Aligned Bounding Box Tree for 3D surface meshes written in Python.

`AABB-Tree` is Python + Numba implementation of static axis-aligned bounding box tree for fast distance queries on 3D surface meshes.

## Features

- Add text.

## Project structure

- [´aabbtree´](src/aabbtree) - Main module for AABB-Tree.
- [´scripts´](examples) - Scripts which serve as examples on how to use `AABB-Tree`.

## Installation from source

The `AABB-Tree` is tested and works on Windows 11 and Linux Debian 12. It should also work on any MacOS systems.
The project can be simply installed trough `pip`. It is recommended to install the project in
a separated [virtual environment](https://docs.python.org/3/library/venv.html) via `venv` module.

1. Clone the repository to a folder.
```sh
git https://github.com/MarkoLeskovar/AABB-Tree
```

2. Navigate to the folder.
```sh
 cd AABB-Tree
```

3. Create a virtual environment with a name ".venv".
```sh
python3 -m venv .venv
```

4. **Linux** - activate virtual environment.
```sh
source .venv/bin/activate
```

4. **Windows** - activate virtual environment.
```sh
source .venv/Scripts/activate
```

5. For local development install `AABB-Tree` in editable mode.
```sh
pip install -e .
```

## Quick programming guide

Add text.
