# Platform Support

PySCF runs natively on Linux and macOS. Windows users have two clear paths.

| Platform | Status | Notes |
| --- | --- | --- |
| **Linux / macOS** | :material-check-circle:{ .green } Full | PySCF installs natively via conda or pip |
| **WSL (Windows)** | :material-check-circle:{ .green } Full | Ubuntu WSL environment — follows the Linux path exactly |
| **Windows native** | :material-alert-circle:{ .yellow } Partial | UI, structure search, and 3D visualization work; PySCF calculations require the Apptainer container |

!!! info "Windows users"
    The easiest path is [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)
    with Ubuntu. Install conda inside WSL and follow the standard
    [installation](installation.md). Alternatively, use the
    [Apptainer container](https://github.com/The-Schultz-Lab/QuantUI/blob/main/apptainer/README.md)
    which bundles the complete environment in a single file.

## GPU offload

Optional NVIDIA GPU acceleration via [gpu4pyscf](https://github.com/pyscf/gpu4pyscf)
requires a CUDA-capable GPU and a compatible driver stack. Set
`QUANTUI_DISABLE_GPU=1` to force CPU execution.

Verify GPU wiring after install:

```bash
quantui gpu check
```

See the [CLI reference](CLI.md#quantui-gpu-check) for sample output and troubleshooting.
