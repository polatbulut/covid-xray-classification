#!/usr/bin/env bash
#
# Download and extract the COVID-19 Radiography Database from Kaggle.
#
# Requires the Kaggle CLI and API credentials in ~/.kaggle/kaggle.json:
#   pip install -e ".[data]"
#   https://www.kaggle.com/docs/api#authentication
#
# Usage: scripts/download_data.sh [destination directory]   (default: data/raw)

set -euo pipefail

readonly DATASET="tawsifurrahman/covid19-radiography-database"
readonly ARCHIVE="covid19-radiography-database.zip"
readonly DEST="${1:-data/raw}"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "error: the 'kaggle' CLI is not installed. Run: pip install -e \".[data]\"" >&2
    exit 1
fi

if [[ -d "${DEST}" ]] && [[ -n "$(ls -A "${DEST}" 2>/dev/null)" ]]; then
    echo "${DEST} already exists and is not empty; nothing to do."
    echo "Delete it first if you want to re-download."
    exit 0
fi

mkdir -p "${DEST}"

# Remove the archive on any exit path, successful or not.
cleanup() { rm -f "${ARCHIVE}"; }
trap cleanup EXIT

echo "Downloading ${DATASET}..."
kaggle datasets download -d "${DATASET}" --force

echo "Extracting into ${DEST}..."
unzip -q -o "${ARCHIVE}" -d "${DEST}"

echo
echo "Done. Contents of ${DEST}:"
ls -1 "${DEST}"
echo
echo "Next: covid-xray split --raw-dir ${DEST}"
