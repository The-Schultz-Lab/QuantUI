# Platform Support

PySCF runs natively on Linux and macOS. Native Windows can use a guarded
PyFock subset or move to WSL/Apptainer for the full feature set.

| Platform | Status | Notes |
| --- | --- | --- |
| **Linux / macOS** | :material-check-circle:{ .green } Full | PySCF installs natively via conda or pip |
| **WSL (Windows)** | :material-check-circle:{ .green } Full | Ubuntu WSL environment — follows the Linux path exactly |
| **Windows native** | :material-alert-circle:{ .yellow } Partial | PyFock: neutral closed-shell PBE/def2 single points; WSL/Apptainer: full PySCF feature set |

!!! info "Windows users"
    Install `quantui[pyfock,ase,app]` under Python 3.11 for native PBE single
    points. QuantUI's engine picker hides unsupported controls. For hybrids,
    ions, radicals, optimizations, spectra, solvent, checkpointing, GPU, and
    orbital analysis, use [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)
    with Ubuntu or the
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
