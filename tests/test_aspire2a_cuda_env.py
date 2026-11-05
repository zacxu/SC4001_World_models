"""Exercise ASPIRE2A's shell activation without CUDA or network access."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aspire2a_cuda.sh"
PINNED_SHA = "7825774c062ef3b5ce82726a90a013b7090ff955627b16b1e5a6a6ea4457c1bf"
BASH = shutil.which("bash")
if os.name == "nt":
    # Avoid the optional WSL launcher, which may exist without a Linux distro.
    BASH = next((str(path) for path in (
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files/Git/usr/bin/bash.exe"),
    ) if path.is_file()), None)


def bash_path(path: Path) -> str:
    value = path.as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@unittest.skipUnless(BASH, "Bash is required for shell integration checks")
class Aspire2aCudaEnvironmentTests(unittest.TestCase):
    def setUp(self):
        # Git Bash uses a restricted Windows token in some local runners; put
        # its writable fixtures in the already-authorized workspace.
        temporary = tempfile.TemporaryDirectory(prefix=".aspire2a CUDA tests ", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "WM1 with spaces"
        self.project.mkdir()
        self.mock_bin = self.root / "mock commands"
        self.mock_bin.mkdir()
        self.archive = self.root / "fixture.tar.xz"
        with tarfile.open(self.archive, "w:xz") as archive:
            for name, value in (
                ("compat/libcuda.so.1", b"mock CUDA library\n" * 65536),
                ("compat/libnvidia-ptxjitcompiler.so.595.71.05", b"mock PTX\n"),
                ("compat/libnvidia-nvvm.so.4", b"mock NVVM\n"),
            ):
                info = tarfile.TarInfo("nvidia-fixture/" + name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
        self.environment = os.environ.copy()
        self.environment.update({
            "FIXTURE_ARCHIVE": bash_path(self.archive),
            "FIXTURE_SHA": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            "MOCK_GPU": "NVIDIA A100-SXM4-40GB, 570.124.06",
            "MOCK_CURL_FAILURE": "0",
            "MOCK_CHECKSUM_FAILURE": "0",
            "MOCK_TAR_FAILURE": "0",
            "MOCK_LOG": bash_path(self.root / "commands.log"),
            "CUDA_VISIBLE_DEVICES": "GPU-pbs-assignment",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "LD_LIBRARY_PATH": "/existing/conda/lib:/cluster/lib",
        })
        self.write_mock("nvidia-smi", """
if [[ "$*" != '--query-gpu=name,driver_version --format=csv,noheader,nounits' ]]; then exit 91; fi
if [[ "${MOCK_SMI_FAILURE:-0}" == 1 ]]; then exit 1; fi
printf '%s\\n' "$MOCK_GPU"
""")
        self.write_mock("curl", """
printf '%s\\n' curl >> "$MOCK_LOG"
output=''
while (( $# )); do
    case "$1" in
        --output) output="$2"; shift 2 ;;
        *) shift ;;
    esac
done
[[ -n "$output" ]] || exit 92
if [[ "$MOCK_CURL_FAILURE" == 1 ]]; then printf partial > "$output"; exit 22; fi
cp -- "$FIXTURE_ARCHIVE" "$output"
""")
        self.write_mock("sha256sum", f"""
# Only the fixed NVIDIA archive digest is substituted, and only for exact
# fixture bytes. Installed-library integrity uses the real implementation.
if (( $# == 2 )) && [[ "$1" == --check && "$2" == --status ]]; then
    IFS= read -r receipt || exit 93
    digest="${{receipt%%  *}}"
    target="${{receipt#*  }}"
    [[ "$digest" == '{PINNED_SHA}' ]] || exit 94
    [[ "${{target##*/}}" == cuda_compat-linux-x86_64-595.71.05_cuda13.2-archive.tar.xz ]] || exit 95
    [[ "$MOCK_CHECKSUM_FAILURE" != 1 ]] || exit 1
    actual=$(/usr/bin/sha256sum -- "$target") || exit 96
    [[ "${{actual%% *}}" == "$FIXTURE_SHA" ]]
else
    exec /usr/bin/sha256sum "$@"
fi
""")
        self.write_mock("tar", """
printf '%s\\n' tar >> "$MOCK_LOG"
[[ "$MOCK_TAR_FAILURE" != 1 ]] || exit 2
exec /usr/bin/tar "$@"
""")

    def write_mock(self, name: str, body: str):
        path = self.mock_bin / name
        path.write_text("#!/bin/bash\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_shell(self, body: str, **environment):
        prefix = (
            "mock_bin=$(cd " + shlex.quote(bash_path(self.mock_bin)) + " && pwd)\n"
            'export PATH="$mock_bin:$PATH"\n'
            "export WM_ROOT=" + shlex.quote(bash_path(self.project)) + "\n"
            'export WM_ROOT=$(readlink -f -- "$WM_ROOT")\n'
            "WM_TEST_SCRIPT=" + shlex.quote(bash_path(SCRIPT)) + "\n"
        )
        return subprocess.run(
            [BASH, "--noprofile", "--norc", "-c", prefix + body],
            env={**self.environment, **environment}, text=True,
            capture_output=True, timeout=30,
        )

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def install(self):
        result = self.run_shell('source "$WM_TEST_SCRIPT" --install')
        self.assert_ok(result)
        return result

    def test_direct_execution_explains_that_script_must_be_sourced(self):
        result = self.run_shell('bash "$WM_TEST_SCRIPT" --install')
        self.assertEqual(result.returncode, 2)
        self.assertIn("source scripts/aspire2a_cuda.sh", result.stderr)
        self.assertFalse((self.project / ".tools").exists())

    def test_missing_installation_fails_without_altering_environment(self):
        result = self.run_shell('''
before=$LD_LIBRARY_PATH
if source "$WM_TEST_SCRIPT"; then exit 71; fi
[[ "$LD_LIBRARY_PATH" == "$before" ]] || exit 72
[[ "$CUDA_VISIBLE_DEVICES" == GPU-pbs-assignment ]] || exit 73
[[ "$CUDA_DEVICE_ORDER" == PCI_BUS_ID ]] || exit 74
''')
        self.assert_ok(result)
        self.assertIn("--install", result.stderr)
        self.assertFalse((self.project / ".tools").exists())

    def test_install_and_activation_support_spaces_and_preserve_pbs_assignment(self):
        result = self.run_shell('''
source "$WM_TEST_SCRIPT" --install || exit 71
expected="$WM_ROOT/.tools/cuda-compat-13.2.2/compat"
[[ "$LD_LIBRARY_PATH" == "$expected:/existing/conda/lib:/cluster/lib" ]] || exit 72
[[ "$CUDA_VISIBLE_DEVICES" == GPU-pbs-assignment ]] || exit 73
[[ "$CUDA_DEVICE_ORDER" == PCI_BUS_ID ]] || exit 74
source "$WM_TEST_SCRIPT" || exit 75
[[ "$LD_LIBRARY_PATH" == "$expected:/existing/conda/lib:/cluster/lib" ]] || exit 76
''')
        self.assert_ok(result)
        self.assertEqual((self.root / "commands.log").read_text().splitlines(), ["curl", "tar"])

    def test_shell_options_and_positional_arguments_are_preserved(self):
        result = self.run_shell('''
set -euo pipefail
set -- caller argument
before_options=$-
before_pipefail=$(set -o | /usr/bin/grep '^pipefail')
source "$WM_TEST_SCRIPT" --install
[[ "$-" == "$before_options" ]]
[[ "$(set -o | /usr/bin/grep '^pipefail')" == "$before_pipefail" ]]
[[ "$#" == 2 && "$1" == caller && "$2" == argument ]]
! declare -F _wm_aspire2a_cuda_main
''')
        self.assert_ok(result)

    def test_path_empty_components_are_preserved_and_activation_is_idempotent(self):
        result = self.run_shell('''
source "$WM_TEST_SCRIPT" --install || exit 71
first=$LD_LIBRARY_PATH
source "$WM_TEST_SCRIPT" || exit 72
[[ "$LD_LIBRARY_PATH" == "$first" ]] || exit 73
[[ "$first" == "$WM_ROOT/.tools/cuda-compat-13.2.2/compat:/one::/two:" ]] || exit 74
''', LD_LIBRARY_PATH="/one::/two:")
        self.assert_ok(result)

    def test_unset_search_path_gets_only_compatibility_directory(self):
        result = self.run_shell('''
unset LD_LIBRARY_PATH
set -u
source "$WM_TEST_SCRIPT" --install || exit 71
[[ "$LD_LIBRARY_PATH" == "$WM_ROOT/.tools/cuda-compat-13.2.2/compat" ]] || exit 72
''')
        self.assert_ok(result)

    def test_native_driver_removes_only_this_projects_obsolete_compatibility_path(self):
        result = self.run_shell('''
export LD_LIBRARY_PATH="$WM_ROOT/.tools/cuda-compat-13.2.2/compat:/other/project/compat:/conda/lib"
source "$WM_TEST_SCRIPT" --install || exit 71
[[ "$LD_LIBRARY_PATH" == /other/project/compat:/conda/lib ]] || exit 72
''', MOCK_GPU="NVIDIA A100-SXM4-40GB, 595.71.05")
        self.assert_ok(result)
        self.assertFalse((self.root / "commands.log").exists())
        self.assertIn("native", result.stdout)

    def test_unsupported_gpu_driver_and_missing_allocation_leave_environment_unchanged(self):
        for values in (
            {"MOCK_GPU": "NVIDIA GeForce RTX 3090, 570.124.06"},
            {"MOCK_GPU": "NVIDIA A100-SXM4-40GB, 525.1.0"},
            {"MOCK_GPU": ""},
            {"MOCK_SMI_FAILURE": "1"},
            {"MOCK_GPU": "NVIDIA A100-SXM4-40GB, 570.124.06\nNVIDIA A100-SXM4-40GB, 570.999.00"},
        ):
            with self.subTest(values=values):
                result = self.run_shell('''
before=$LD_LIBRARY_PATH
if source "$WM_TEST_SCRIPT" --install; then exit 71; fi
[[ "$LD_LIBRARY_PATH" == "$before" ]] || exit 72
[[ "$CUDA_VISIBLE_DEVICES" == GPU-pbs-assignment ]] || exit 73
''', **values)
                self.assert_ok(result)
                self.assertFalse((self.project / ".tools").exists())

    def test_download_checksum_and_extraction_failures_preserve_partial_files(self):
        for name, message in (
            ("MOCK_CURL_FAILURE", "Download failed"),
            ("MOCK_CHECKSUM_FAILURE", "checksum failed"),
            ("MOCK_TAR_FAILURE", "extraction failed"),
        ):
            with self.subTest(failure=name):
                result = self.run_shell('''
before=$LD_LIBRARY_PATH
if source "$WM_TEST_SCRIPT" --install; then exit 71; fi
[[ "$LD_LIBRARY_PATH" == "$before" ]] || exit 72
[[ "$CUDA_VISIBLE_DEVICES" == GPU-pbs-assignment ]] || exit 73
''', **{name: "1"})
                self.assert_ok(result)
                self.assertIn(message, result.stderr)
                self.assertFalse((self.project / ".tools" / "cuda-compat-13.2.2").exists())
                self.assertTrue(list((self.project / ".tools").glob(".cuda-compat-13.2.2.install.*")))

    def test_installed_library_tampering_fails_before_environment_change(self):
        self.install()
        library = self.project / ".tools" / "cuda-compat-13.2.2" / "compat" / "libnvidia-nvvm.so.4"
        library.write_bytes(b"modified library")
        result = self.run_shell('''
before=$LD_LIBRARY_PATH
if source "$WM_TEST_SCRIPT"; then exit 71; fi
[[ "$LD_LIBRARY_PATH" == "$before" ]] || exit 72
''')
        self.assert_ok(result)
        self.assertIn("checksum failed", result.stderr)

    def test_archive_tampering_after_installation_is_detected(self):
        self.install()
        cache = self.project / ".tools" / "cuda-compat-13.2.2"
        next(cache.glob("*.tar.xz")).write_bytes(b"corrupted archive")
        result = self.run_shell('''
before=$LD_LIBRARY_PATH
if source "$WM_TEST_SCRIPT"; then exit 71; fi
[[ "$LD_LIBRARY_PATH" == "$before" ]] || exit 72
''')
        self.assert_ok(result)
        self.assertIn("checksum failed", result.stderr)

    def test_incomplete_existing_installation_is_preserved_and_not_replaced(self):
        cache = self.project / ".tools" / "cuda-compat-13.2.2"
        cache.mkdir(parents=True)
        marker = cache / "preserve-for-inspection"
        marker.write_text("incomplete")
        result = self.run_shell('''
before=$LD_LIBRARY_PATH
if source "$WM_TEST_SCRIPT" --install; then exit 71; fi
[[ "$LD_LIBRARY_PATH" == "$before" ]] || exit 72
''')
        self.assert_ok(result)
        self.assertEqual(marker.read_text(), "incomplete")
        self.assertFalse((self.root / "commands.log").exists())


if __name__ == "__main__":
    unittest.main()
