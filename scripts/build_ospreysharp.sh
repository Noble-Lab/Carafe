#!/usr/bin/env bash
#
# Build self-contained Osprey executables for bundling with the Carafe
# distribution. Osprey is the MacCoss-lab DIA search tool that lives in the
# ProteoWizard (pwiz) tree; Carafe drives it as an external search engine.
#
# For each requested .NET runtime identifier (RID) this script runs
#   dotnet publish -c Release -f net8.0 -r <rid> --self-contained ...
# and stages the published output under <output-root>/<rid>/ so Carafe's
# resolveOspreyBinary() can locate Osprey(.exe) next to the jar (or under
# ~/.carafe/osprey/<rid>/) at runtime with no separate .NET runtime install.
#
# Usage:
#   scripts/build_ospreysharp.sh [rid ...]
#
# Environment overrides:
#   OSPREY_CSPROJ  Path to Osprey.csproj
#                  (default: ~/GitHub-Repo/ProteoWizard/pwiz/pwiz_tools/Osprey/Osprey/Osprey.csproj)
#   OUTPUT_ROOT    Where to stage per-RID output
#                  (default: <repo>/target/osprey)
#
# Examples:
#   scripts/build_ospreysharp.sh                # build the default RID for this host
#   scripts/build_ospreysharp.sh win-x64 linux-x64 osx-arm64
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OSPREY_CSPROJ="${OSPREY_CSPROJ:-${HOME}/GitHub-Repo/ProteoWizard/pwiz/pwiz_tools/Osprey/Osprey/Osprey.csproj}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/target/osprey}"
TARGET_FRAMEWORK="net8.0"

# Default RID: infer from the host when none is given on the command line.
default_rid() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    case "${os}" in
        Linux)  echo "linux-x64" ;;
        Darwin)
            case "${arch}" in
                arm64) echo "osx-arm64" ;;
                *)     echo "osx-x64" ;;
            esac ;;
        MINGW*|MSYS*|CYGWIN*) echo "win-x64" ;;
        *)      echo "linux-x64" ;;
    esac
}

if ! command -v dotnet >/dev/null 2>&1; then
    echo "ERROR: 'dotnet' SDK not found on PATH. Install the .NET 8 SDK first." >&2
    exit 1
fi

if [[ ! -f "${OSPREY_CSPROJ}" ]]; then
    echo "ERROR: Osprey.csproj not found at: ${OSPREY_CSPROJ}" >&2
    echo "       Set OSPREY_CSPROJ to the correct path." >&2
    exit 1
fi

RIDS=("$@")
if [[ ${#RIDS[@]} -eq 0 ]]; then
    RIDS=("$(default_rid)")
fi

echo "Osprey csproj : ${OSPREY_CSPROJ}"
echo "Output root        : ${OUTPUT_ROOT}"
echo "Target framework   : ${TARGET_FRAMEWORK}"
echo "RIDs               : ${RIDS[*]}"
echo

for rid in "${RIDS[@]}"; do
    dest="${OUTPUT_ROOT}/${rid}"
    echo "==> Publishing Osprey for ${rid} -> ${dest}"
    rm -rf "${dest}"
    mkdir -p "${dest}"
    # Do NOT use PublishSingleFile: Osprey's blib writer uses System.Data.SQLite,
    # whose native SQLite.Interop.dll is located via the managed assembly's own directory.
    # Under single-file publish Assembly.Location is empty, so the SQLiteConnection ctor
    # throws ArgumentNullException (Parameter 'path1') when it opens the output .blib.
    # A folder publish ships SQLite.Interop.dll as a real sibling file and works.
    dotnet publish "${OSPREY_CSPROJ}" \
        -c Release \
        -f "${TARGET_FRAMEWORK}" \
        -r "${rid}" \
        --self-contained true \
        -p:PublishSingleFile=false \
        -p:Platform=x64 \
        -o "${dest}"

    # Make the published binary executable on Unix RIDs.
    if [[ "${rid}" != win-* ]]; then
        if [[ -f "${dest}/Osprey" ]]; then
            chmod +x "${dest}/Osprey"
        fi
    fi
    echo "    done: ${dest}"
done

echo
echo "Staged Osprey builds under ${OUTPUT_ROOT}."
echo "Copy the appropriate <rid>/ directory next to the Carafe jar (beside lib/)"
echo "as 'osprey/<rid>/' so resolveOspreyBinary() finds it, or let the jpackage"
echo "step include it via --input."
