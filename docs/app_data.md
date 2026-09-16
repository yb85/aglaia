# APP_DATA — per-user directories & config DB

Aglaïa keeps all per-user state outside the repo, resolved through
[`platformdirs`](https://github.com/tox-dev/platformdirs) (vendor
`bibli.cc`, app `Aglaia`). Implemented in `aglaia/app_data/__init__.py`.

## Directories

| Helper | macOS default | Holds |
|---|---|---|
| `app_data_dir()` | `~/Library/Application Support/Aglaia` | config DB, `pipelines/`, `models/`, `plugins/` |
| `cache_dir()` | `~/Library/Caches/Aglaia` | disposable caches (safe to delete) |
| `log_dir()` | `~/Library/Logs/Aglaia` | one rotated log per session |
| `models_dir()` | `<APP_DATA>/models` | downloaded ML weights (Surya, EAST, DBNet…) |
| `pipelines_dir()` | `<APP_DATA>/pipelines` | user pipeline YAMLs |
| `plugins_dir(kind)` | `<APP_DATA>/plugins/{processors,ocr,destinations}` | plugins, dropped in or installed from the registry (see [processors.md](./processors.md), [destinations.md](./destinations.md)) |
| `config_db_path()` | `<APP_DATA>/aglaia-config.db` | SQLite config + recent projects |

Linux uses the XDG dirs (`~/.local/share/Aglaia`, `~/.cache/Aglaia`,
`~/.local/state/Aglaia/log`); Windows nests under `%APPDATA%\bibli.cc\Aglaia`.

`models_dir()` is user-overridable (Settings → Models, key `models_dir`):
empty → `<APP_DATA>/models`; relative → resolved against APP_DATA;
absolute → used as-is. It lives under APP_DATA (not cache) so explicitly
downloaded weights survive a cache purge.

## Environment overrides

For tests / portable installs:

- `AGLAIA_APP_DATA_DIR` — overrides `app_data_dir()`
- `AGLAIA_LOG_DIR` — overrides `log_dir()`
- `AGLAIA_CACHE_DIR` — overrides `cache_dir()`

## Config DB (`aglaia-config.db`)

SQLite, created + seeded on first connect (`aglaia/app_data/db.py`). Tables:

- `config(key TEXT PK, value TEXT)` — JSON-encoded values; `get(key,
  default)` / `set(key, value)`. Canonical keys (`KEY_*`): theme,
  language, OCR defaults, export defaults, worker count, OCR DPI,
  confidence gate, models dir, … `bootstrap()` seeds missing keys from
  `config/config_default.yaml` + `BUILTIN_DEFAULTS`.
- `recent_projects(path PK, name, opened_at, scan_count)` — startup picker.
- `plugins(path PK, kind, sha256, status, added_at)` — accepted drop-in
  plugins (trust gate; see [processors.md](./processors.md)).

This is distinct from the **project** store — each `.agl` project is its
own SQLite DB (see [storage.md](./storage.md)). Secrets (the Mistral API
key, a plugin's password or token) never go in this DB (see
[ocr.md](./ocr.md)).

## Where a secret is written

| Run | Store |
|---|---|
| the GUI (`aglaia` / `aglaia gui`) | the **OS keychain** via `keyring` |
| every other command (`run`, `ocr`, `plugins`, `server`, `setup`, …) | **`<APP_DATA>/.env`**, mode 0600 |

A keychain is a *session* service: on Linux it needs a logged-in desktop to
unlock it, so a key stored there is unreadable to the cron job, the ssh
session or the systemd unit that has to use it — and the failure looks like a
wrong key, which sends the user to rotate one that was fine. A headless
install therefore keeps its secrets in a file owned by the account that runs
the batch. `aglaia/cli` sets this for every command but `gui`
(`secrets.use_plaintext_store`); `AGLAIA_SECRETS_PLAINTEXT=0` or `=1` forces
either store.

**Reading is unaffected and checks all of them**: the environment variable
first, then `.env`, then the keychain — so a key stored by the GUI keeps
working in a batch on the same machine, and nothing already stored becomes
unreachable. Plugin secrets are namespaced per plugin: the keychain account
is `<slug>:<key>`, the `.env` line is
`AGLAIA_PLUGIN_<SLUG>_<KEY>=…`.
