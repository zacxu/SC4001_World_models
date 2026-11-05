#!/bin/bash
# Source this file before starting Python in each ASPIRE2A GPU allocation.
# --install downloads only NVIDIA user-space compatibility libraries; no driver
# or system package is installed. PBS jobs must use activation without --install.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Use: source scripts/aspire2a_cuda.sh [--install]' >&2
    exit 2
fi

_wm_aspire2a_cuda_main() {
    local wm_mode="${1:-}" wm_root wm_tools wm_cache wm_driver_output
    local wm_name wm_driver wm_branch wm_first_driver='' wm_native=0
    local wm_archive='cuda_compat-linux-x86_64-595.71.05_cuda13.2-archive.tar.xz'
    local wm_sha='7825774c062ef3b5ce82726a90a013b7090ff955627b16b1e5a6a6ea4457c1bf'
    local wm_url='https://developer.download.nvidia.com/compute/cuda/redist/cuda_compat/linux-x86_64'
    local wm_stage='' wm_libfile wm_libdir wm_target wm_item wm_relative
    local wm_old_path="${LD_LIBRARY_PATH-}" wm_rest wm_part wm_clean='' wm_have=0
    local wm_manifest='' wm_required wm_found

    if (( $# > 1 )) || [[ -n "$wm_mode" && "$wm_mode" != '--install' ]]; then
        printf '%s\n' 'Use: source scripts/aspire2a_cuda.sh [--install]' >&2
        return 2
    fi
    if [[ -z "${WM_ROOT:-}" || ! -d "$WM_ROOT" ]]; then
        printf '%s\n' 'Set WM_ROOT to the existing WM1 project directory first.' >&2
        return 1
    fi
    wm_root=$(readlink -f -- "$WM_ROOT") || return 1
    wm_tools="$wm_root/.tools"
    wm_cache="$wm_tools/cuda-compat-13.2.2"

    if ! wm_driver_output=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits); then
        printf '%s\n' 'nvidia-smi failed. Run this inside a PBS GPU allocation.' >&2
        return 1
    fi
    if [[ -z "$wm_driver_output" ]]; then
        printf '%s\n' 'No GPU was reported; request a PBS GPU allocation first.' >&2
        return 1
    fi
    while IFS=, read -r wm_name wm_driver; do
        wm_name="${wm_name#"${wm_name%%[![:space:]]*}"}"
        wm_name="${wm_name%"${wm_name##*[![:space:]]}"}"
        wm_driver="${wm_driver#"${wm_driver%%[![:space:]]*}"}"
        wm_driver="${wm_driver%"${wm_driver##*[![:space:]]}"}"
        if [[ ! "$wm_name" =~ (^|[[:space:]])A100([[:space:]-]|$) ]] ||
           [[ ! "$wm_driver" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
            printf 'Unsupported GPU/driver for this ASPIRE2A helper: %s, %s\n' "$wm_name" "$wm_driver" >&2
            return 1
        fi
        if [[ -n "$wm_first_driver" && "$wm_driver" != "$wm_first_driver" ]]; then
            printf '%s\n' 'Mixed driver versions were reported; stop and inspect the allocation.' >&2
            return 1
        fi
        wm_first_driver="$wm_driver"
        wm_branch="${wm_driver%%.*}"
        if (( 10#$wm_branch >= 595 )); then
            wm_native=1
        else
            case "$wm_branch" in
                535|570|580|590) ;;
                *) printf 'Driver branch %s is outside this helper\047s supported compatibility matrix.\n' "$wm_branch" >&2; return 1 ;;
            esac
        fi
    done <<< "$wm_driver_output"

    if (( ! wm_native )); then
        for wm_item in sha256sum find readlink; do
            if ! command -v "$wm_item" >/dev/null 2>&1; then
                printf 'Required command is unavailable: %s\n' "$wm_item" >&2
                return 1
            fi
        done
        if [[ ! -d "$wm_cache" ]]; then
            if [[ -e "$wm_cache" || -L "$wm_cache" ]]; then
                printf 'Installation path already exists but is not a directory: %s\n' "$wm_cache" >&2
                return 1
            fi
            if [[ "$wm_mode" != '--install' ]]; then
                printf '%s\n' 'Compatibility libraries are missing. In the setup allocation run: source scripts/aspire2a_cuda.sh --install' >&2
                return 1
            fi
            for wm_item in curl tar mktemp mv; do
                if ! command -v "$wm_item" >/dev/null 2>&1; then
                    printf 'Required installation command is unavailable: %s\n' "$wm_item" >&2
                    return 1
                fi
            done
            mkdir -p -- "$wm_tools" || return 1
            wm_stage=$(mktemp -d "$wm_tools/.cuda-compat-13.2.2.install.XXXXXXXX") || return 1
            printf 'Downloading NVIDIA compatibility libraries to %s\n' "$wm_stage"
            if ! curl --fail --location --retry 3 --output "$wm_stage/$wm_archive" "$wm_url/$wm_archive"; then
                printf 'Download failed. Partial files remain in %s; no environment was changed.\n' "$wm_stage" >&2
                return 1
            fi
            if ! printf '%s  %s\n' "$wm_sha" "$wm_stage/$wm_archive" | sha256sum --check --status; then
                printf 'NVIDIA archive checksum failed. Files remain in %s; no environment was changed.\n' "$wm_stage" >&2
                return 1
            fi
            if ! tar -xJf "$wm_stage/$wm_archive" --strip-components=1 -C "$wm_stage"; then
                printf 'Archive extraction failed. Files remain in %s.\n' "$wm_stage" >&2
                return 1
            fi
        fi

        wm_target="${wm_stage:-$wm_cache}"
        wm_libfile=$(find "$wm_target" -name 'libcuda.so.1' -print) || return 1
        if [[ -z "$wm_libfile" || "$wm_libfile" == *$'\n'* ]]; then
            printf 'Expected exactly one libcuda.so.1 under %s. Preserve this directory and inspect the installation.\n' "$wm_target" >&2
            return 1
        fi
        wm_libdir="${wm_libfile%/*}"
        # Resolve required libraries so broken links, stubs and links outside the
        # private installation cannot silently replace the CUDA driver library.
        for wm_required in 'libcuda.so.1' 'libnvidia-ptxjitcompiler.so.*' 'libnvidia-nvvm.so.*'; do
            wm_found=0
            for wm_item in "$wm_libdir"/$wm_required; do
                [[ -e "$wm_item" || -L "$wm_item" ]] || continue
                wm_relative=$(readlink -f -- "$wm_item") || return 1
                if [[ "$wm_relative" != "$wm_target"/* || ! -f "$wm_relative" || ! -s "$wm_relative" ]]; then
                    printf 'Invalid compatibility library: %s\n' "$wm_item" >&2
                    return 1
                fi
                wm_found=1
            done
            if (( ! wm_found )); then
                printf 'Required compatibility library is missing: %s/%s\n' "$wm_libdir" "$wm_required" >&2
                return 1
            fi
        done
        wm_relative=$(readlink -f -- "$wm_libfile") || return 1
        if (( $(wc -c < "$wm_relative") < 1048576 )); then
            printf 'libcuda.so.1 appears to be a linker stub: %s\n' "$wm_relative" >&2
            return 1
        fi

        if [[ -n "$wm_stage" ]]; then
            wm_relative="${wm_libdir#"$wm_stage/"}"
            printf '%s\n' "$wm_relative" > "$wm_stage/library-directory.txt" || return 1
            printf '%s\n' "$wm_sha" > "$wm_stage/archive.sha256" || return 1
            if ! (cd "$wm_stage" && find "$wm_relative" -type f -exec sha256sum -- {} +) > "$wm_stage/libraries.sha256"; then
                return 1
            fi
            if ! mv -T -n -- "$wm_stage" "$wm_cache" || [[ -d "$wm_stage" ]]; then
                printf 'Could not publish installation. Files remain in %s; inspect concurrent/partial installations.\n' "$wm_stage" >&2
                return 1
            fi
        fi
        if [[ ! -f "$wm_cache/archive.sha256" || ! -f "$wm_cache/libraries.sha256" || ! -f "$wm_cache/library-directory.txt" ]]; then
            printf 'Incomplete compatibility installation: %s. Preserve it for inspection; do not activate it.\n' "$wm_cache" >&2
            return 1
        fi
        wm_manifest=$(cat "$wm_cache/archive.sha256") || return 1
        if [[ "$wm_manifest" != "$wm_sha" ]] ||
           ! printf '%s  %s\n' "$wm_sha" "$wm_cache/$wm_archive" | sha256sum --check --status ||
           ! (cd "$wm_cache" && sha256sum --check --status libraries.sha256); then
            printf 'Compatibility installation checksum failed: %s. No environment was changed.\n' "$wm_cache" >&2
            return 1
        fi
        wm_relative=$(cat "$wm_cache/library-directory.txt") || return 1
        wm_libdir=$(readlink -f -- "$wm_cache/$wm_relative") || return 1
        if [[ "$wm_libdir" != "$wm_cache"/* || ! -e "$wm_libdir/libcuda.so.1" ]]; then
            printf 'Invalid compatibility library directory under %s.\n' "$wm_cache" >&2
            return 1
        fi
    fi

    # Remove only this project's previous compatibility entry. Preserve the
    # remaining search path, including intentional empty components.
    if [[ -n "$wm_old_path" ]]; then
        wm_rest="$wm_old_path"
        while :; do
            wm_part="${wm_rest%%:*}"
            case "$wm_part" in
                "$wm_cache"|"$wm_cache"/*) ;;
                *)
                    if (( wm_have )); then wm_clean+=":"; fi
                    wm_clean+="$wm_part"
                    wm_have=1
                    ;;
            esac
            [[ "$wm_rest" == *:* ]] || break
            wm_rest="${wm_rest#*:}"
        done
    fi
    if (( wm_native )); then
        if [[ "$wm_clean" != "$wm_old_path" ]]; then
            if (( wm_have )); then export LD_LIBRARY_PATH="$wm_clean"; else unset LD_LIBRARY_PATH; fi
        fi
        printf 'CUDA 13.2: using native NVIDIA driver %s.\n' "$wm_first_driver"
    else
        if (( wm_have )); then export LD_LIBRARY_PATH="$wm_libdir:$wm_clean"; else export LD_LIBRARY_PATH="$wm_libdir"; fi
        printf 'CUDA 13.2 compatibility activated for A100 driver %s.\n' "$wm_first_driver"
    fi
    return 0
}

# Every required operation checks its result. Keep cleanup reachable under
# set -e; a returned failure still lets the calling PBS script stop normally.
if _wm_aspire2a_cuda_main "$@"; then
    unset -f _wm_aspire2a_cuda_main
    return 0
else
    unset -f _wm_aspire2a_cuda_main
    return 1
fi
