#!/usr/bin/env bash
set -euo pipefail

PREFIX="${CONDA_INSTALL_PREFIX:-$HOME/miniforge3}"

for candidate in "$PREFIX" "$HOME/miniforge3" "$HOME/miniconda3" \
                 "$HOME/anaconda3" /opt/conda; do
    if [ -x "$candidate/bin/conda" ]; then
        echo "conda is already installed at $candidate -- nothing to do."
        echo "Next: bash quick_reproduce.sh"
        exit 0
    fi
done
if command -v conda >/dev/null 2>&1; then
    echo "conda is already on PATH ($(command -v conda)) -- nothing to do."
    echo "Next: bash quick_reproduce.sh"
    exit 0
fi

case "$(uname -s)" in
    Linux)  os=Linux ;;
    Darwin) os=MacOSX ;;
    *) echo "Unsupported OS $(uname -s); install Miniforge manually:" >&2
       echo "  https://github.com/conda-forge/miniforge#install" >&2
       exit 1 ;;
esac
case "$(uname -m)" in
    x86_64|amd64)  arch=x86_64 ;;
    aarch64)       arch=aarch64 ;;
    arm64)         arch=arm64 ;;
    *) echo "Unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${os}-${arch}.sh"
INSTALLER="$(mktemp -t miniforge-XXXXXX.sh)"
trap 'rm -f "$INSTALLER"' EXIT

have_downloader() {
    command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1
}

if ! have_downloader; then
    SUDO=""
    if [ "$(id -u)" != 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
    echo "[install_conda] neither curl nor wget found; installing curl"
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update -qq \
            && $SUDO env DEBIAN_FRONTEND=noninteractive \
                   apt-get install -y -qq curl ca-certificates
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y -q curl ca-certificates
    elif command -v yum >/dev/null 2>&1; then
        $SUDO yum install -y -q curl ca-certificates
    elif command -v apk >/dev/null 2>&1; then
        $SUDO apk add --no-cache curl ca-certificates
    elif command -v pacman >/dev/null 2>&1; then
        $SUDO pacman -Sy --noconfirm curl ca-certificates
    elif command -v zypper >/dev/null 2>&1; then
        $SUDO zypper --non-interactive install curl ca-certificates
    fi
fi
if ! have_downloader; then
    echo "Could not install a downloader automatically. Install curl, then re-run:" >&2
    echo "  sudo apt-get install -y curl ca-certificates   # or your package manager" >&2
    exit 1
fi

echo "[install_conda] downloading $URL"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$INSTALLER"
else
    wget -qO "$INSTALLER" "$URL"
fi

echo "[install_conda] installing into $PREFIX (unattended)"
bash "$INSTALLER" -b -p "$PREFIX"

export PATH="$PREFIX/bin:$PATH"

for sh in bash zsh; do
    command -v "$sh" >/dev/null 2>&1 && conda init "$sh" >/dev/null 2>&1 || true
done

version="$(conda --version 2>/dev/null || echo "conda (version check failed)")"
echo
echo "[install_conda] done: $version"
echo "Next: bash quick_reproduce.sh"
