# Platform Support

| Platform | Status | Notes |
| --- | --- | --- |
| **Linux / macOS** | :material-check-circle:{ .green } Full | PySCF via conda or pip |
| **WSL (Windows)** | :material-check-circle:{ .green } Full | Same as Linux |
| **Windows native** | :material-alert-circle:{ .yellow } Partial | PyFock: neutral closed-shell PBE single points and optimizations |

!!! info "Windows users"
    `quantui[pyfock,ase,app]` (Python 3.11) covers PBE single points and optimizations. For everything else use [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) or the [Apptainer container](https://github.com/The-Schultz-Lab/QuantUI/blob/main/apptainer/README.md).

## GPU offload

Optional, via [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) on a CUDA GPU (`quantui[gpu-cuda12x]` or `quantui[gpu-cuda13x]`, matching your driver). `QUANTUI_DISABLE_GPU=1` forces CPU. PyFock has its own CuPy GPU path (`quantui[pyfock,pyfock-gpu-cuda12x]` or `-cuda13x`). Check them with:

```bash
quantui gpu check                 # gpu4pyscf
quantui gpu check --engine pyfock # PyFock
```

See the [CLI reference](CLI.md#quantui-gpu-check).
