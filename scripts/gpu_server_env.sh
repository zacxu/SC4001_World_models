#!/bin/bash
# Source after creating .envs/wm312 using linux_gpu_server_setup.md.
# GPU selection remains entirely controlled by CUDA_VISIBLE_DEVICES.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Use: source scripts/gpu_server_env.sh' >&2
    exit 2
fi

_wm_activate_server() {
    local wm_project wm_conda_base wm_ffmpeg
    wm_project="${WM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
    if [[ ! -f "$wm_project/pyproject.toml" ]]; then
        printf '%s\n' 'WM_ROOT must point to the project directory containing pyproject.toml.' >&2
        return 1
    fi
    export WM_ROOT="$wm_project"
    if command -v conda >/dev/null 2>&1; then
        wm_conda_base=$(conda info --base) || return 1
        source "$wm_conda_base/etc/profile.d/conda.sh" || return 1
    elif [[ -r "$WM_ROOT/.tools/miniforge3/etc/profile.d/conda.sh" ]]; then
        source "$WM_ROOT/.tools/miniforge3/etc/profile.d/conda.sh" || return 1
    else
        printf '%s\n' 'Conda is unavailable. Complete linux_gpu_server_setup.md first.' >&2
        return 1
    fi
    conda activate "$WM_ROOT/.envs/wm312" || return 1
    if [[ -x "$WM_ROOT/.tools/tmux/bin/tmux" ]]; then
        case ":$PATH:" in
            *":$WM_ROOT/.tools/tmux/bin:"*) ;;
            *) export PATH="$WM_ROOT/.tools/tmux/bin:$PATH" ;;
        esac
    fi
    wm_ffmpeg=$(command -v ffmpeg) || {
        printf '%s\n' 'ffmpeg is missing from the active environment; complete the setup guide.' >&2
        return 1
    }
    case ":${LD_LIBRARY_PATH-}:" in
        *":$CONDA_PREFIX/lib:"*) ;;
        *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    export WORLD_MODELS_ACC="$WM_ROOT/.tools/acc-linux/acc"
    export IMAGEIO_FFMPEG_EXE="$wm_ffmpeg"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
}

if _wm_activate_server; then
    unset -f _wm_activate_server
else
    unset -f _wm_activate_server
    return 1
fi
