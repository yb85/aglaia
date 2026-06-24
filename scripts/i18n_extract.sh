#!/usr/bin/env bash
# Re-scan the GUI source for self.tr() / QCoreApplication.translate
# calls and update the .ts source catalogues. Run after wrapping new
# strings.
#
# After editing translations in Qt Linguist (`pyside6-linguist`), run
# scripts/i18n_compile.sh to produce the .qm binaries.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

LUPDATE="${LUPDATE:-uv run --no-sync pyside6-lupdate}"

# In-scope sources: GUI package + the GUI-surfaced strings in workers
# (toast messages, error dialogs surfaced via log_queue, ProcessMonitor
# emissions consumed by MainWindow).
SOURCES=()
while IFS= read -r f; do SOURCES+=("$f"); done < <(find aglaia/gui -name "*.py" -not -path "*/__pycache__/*")
while IFS= read -r f; do SOURCES+=("$f"); done < <(find aglaia/workers -name "*.py" -not -path "*/__pycache__/*")

# Output catalogues — one per supported locale.
TS_FILES=(
    aglaia/i18n/aglaia_en_US.ts
    aglaia/i18n/aglaia_fr_FR.ts
)

echo "lupdate: scanning ${#SOURCES[@]} files → ${TS_FILES[*]}"
$LUPDATE "${SOURCES[@]}" -ts "${TS_FILES[@]}" -no-obsolete
echo "done."
