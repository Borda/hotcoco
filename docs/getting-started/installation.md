# Installation

## Python

```bash
pip install hotcoco
```

Verify the installation:

```python
from hotcoco import COCO
print("hotcoco installed successfully")
```

hotcoco ships with type stubs (`.pyi`) and a `py.typed` marker, so autocomplete, hover docs, and type checking work out of the box in VS Code, PyCharm, and other editors. numpy is installed automatically as a dependency.

!!! tip "You don't need the images"
    Evaluation reads only the JSON annotation and result files — never the image files themselves. There's no need to download an image set to get started.

## CLI

The `coco` command ships with the Python package — `pip install hotcoco` is all you need.

For a standalone binary with no Python dependency, install the Rust CLI:

```bash
cargo install hotcoco-cli
```

This installs the `coco-eval` binary, which does evaluation only.

## Rust library

```bash
cargo add hotcoco
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
hotcoco = "1.0"
```

Full API documentation is on [docs.rs](https://docs.rs/hotcoco).

## Building from source

To build hotcoco yourself — or to run the benchmarks and parity checks against COCO val2017 — see [CONTRIBUTING.md](https://github.com/derekallman/hotcoco/blob/main/CONTRIBUTING.md).
