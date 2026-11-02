# Platform Support

| Platform | Status | Notes |
| --- | --- | --- |
| **Linux / macOS** | :material-check-circle:{ .green } Full | PySCF via conda or pip |
| **WSL (Windows)** | :material-check-circle:{ .green } Full | Same as Linux |
| **Windows native** | :material-alert-circle:{ .yellow } Partial | PyFock: neutral closed-shell PBE single points and optimizations |

!!! info "Windows users"
    `quantui[pyfock,ase,app]` (Python 3.11) covers PBE single points and optimizations. For everything else use [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) or the [Apptainer container](https://github.com/The-Schultz-Lab/QuantUI/blob/main/apptainer/README.md).

## GPU offload

Optional, via [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) on a CUDA GPU. `QUANTUI_DISABLE_GPU=1` forces CPU. Check it with:

```bash
quantui gpu check
```

See the [CLI reference](CLI.md#quantui-gpu-check).
