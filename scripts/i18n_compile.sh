#!/usr/bin/env bash
# Compile .ts → .qm. The runtime loader (`lib/i18n.install_translator`)
# reads .qm out of lib/i18n/qm/.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

LRELEASE="${LRELEASE:-uv run --no-sync pyside6-lrelease}"

mkdir -p lib/i18n/qm
for ts in lib/i18n/aglaia_*.ts; do
    base="$(basename "$ts" .ts)"
    qm="lib/i18n/qm/${base}.qm"
    echo "lrelease: $ts → $qm"
    $LRELEASE "$ts" -qm "$qm"
done
echo "done."
