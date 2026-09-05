# Destinations — sending an export somewhere

A **destination** is somewhere a finished export goes: a calibre library, a
Kindle mailbox, a private corpus. It is the third plugin kind, beside
`processors` and `ocr`, and the first one built on the plugin API designed in
[plugin-store.md](./plugin-store.md).

Three ship with the app, in `aglaia/plugins/destinations/<slug>/`, in exactly
the layout the plugin registry will use — a manifest, one top-level module, a
README. They are working destinations and simultaneously the worked example
every submitted plugin is measured against.

| Slug | Sends via | Accepts |
|---|---|---|
| `send-to-calibre` | calibre content server, `POST /cdb/add-book/…` | pdf, md, txt |
| `send-to-kindle` | SMTP, as a MIME attachment | pdf, epub, txt, md, docx |
| `send-to-corpus` | Corpus `POST /book/upload` | pdf, md, txt, epub, html |

## Why one kind and not two

The obvious shape is "email destinations" and "API destinations". The wire
says otherwise:

| | calibre | kindle | corpus |
|---|---|---|---|
| Transport | HTTP | SMTP | HTTP |
| Auth | Basic **or Digest** | SMTP credentials | `X-API-Key` header |
| Payload | **raw body**, parameters in the **URL path** | MIME attachment | **multipart** fields |
| Metadata | calibre reads it from the file | subject line | explicit form fields |
| Result | JSON `book_id` \| `duplicates` | SMTP 250 | 201 / 200 / 400 / 413 |

calibre and corpus are both HTTP and share nothing else. A common `send()`
across them would be a signature and a shrug, and the abstraction would have to
be broken open by the first destination that does not fit.

What *is* common is everything around the transport — the settings schema, the
credential storage, the check/send split, the result shape, the GUI that
renders all of it — and that is exactly what `Destination` carries.

## The contract

```python
from aglaia.plugin_api import (
    Destination, Field, BookMeta, SendResult, CheckResult, register_destination)

@register_destination
class MyDestination(Destination):
    name = "send-to-somewhere"       # registry key; matches the plugin slug
    display = "Somewhere"            # menus
    description = "One line."
    accepts = ("pdf",)               # export formats it can take

    CONFIG_FIELDS = (Field("url", "Server URL", "str", "", required=True),)
    SECRET_FIELDS = (Field("token", "API token", "secret", "", required=True),)

    def check(self) -> CheckResult: ...
    def send(self, path: Path, meta: BookMeta) -> SendResult: ...
```

**`Field`** describes a setting so the host can render a form without knowing
what the destination is — the same idea as the OCR tab reading engine
capability flags instead of hard-coding engine names. `kind` is one of `str`,
`int`, `bool`, `choice`, `secret`. A `secret` field is rendered masked and
stored in the keychain; everything else goes to the plugin's own settings file.

**`self.conf(key)`** reads a setting, falling back to the field's declared
default — so a fresh install reads `587`, not `None`, and a plugin does not
repeat its defaults in two places. **`self.secret(key)`** reads a credential.
**`missing_settings()`** returns the labels of required fields still empty, so
the host can say what is needed instead of letting the send fail on the far end
for a reason it could have named locally.

**`check()` is separate from `send()`** so a configuration can be proved
without pushing a book into a library — which matters most for Kindle, where a
test document lands somewhere the user cannot tidy from the desktop.

**`SendResult.ok` is not a verdict on the user's intent.** A document the
destination already had is `ok=True, already_there=True`. Flattening that into
a failure teaches people to ignore failures.

## Storage

Per plugin, under `<APP_DATA>/plugins/data/<slug>/`:

* `config.db` — a single `kv` table, this destination's settings.
* `files/` — scratch (`ctx.data_dir`).

Secrets go to the OS keychain under service `aglaia.plugin`, username
`<slug>\x1f<key>`. The slug is bound by the host at construction, and `\x1f`
cannot occur in an accepted key, so no plugin can name its way into another's
namespace *through the API*. With no keychain available they fall back to the
plugin's own config file, and `ctx.secrets.available` reports which — see
[plugin-store.md §1](./plugin-store.md) for why that is not a security
boundary.

`config.all()` hides host-reserved rows, because it is what a settings form
reads and a password does not belong in one.

## The three, and what each gets wrong if you are not careful

### calibre

`POST {base}{prefix}/cdb/add-book/{job_id}/{add_duplicates}/{filename}/{library_id}`

The file goes in the **raw request body** — not multipart. Send multipart and
calibre stores a book whose contents are a MIME envelope. Every parameter is a
**path segment**, so each is percent-quoted with an empty safe list: an
unquoted `/` in a title reshapes the route and a space breaks the request line.
The filename is what calibre names the book, so the document's title is sent
rather than `project_003_A.pdf`.

The route needs database write access: the server must run `--enable-auth` with
a user that is not restricted to a read-only library. calibre defaults to
**digest** auth; `--auth-mode=basic` is the manual's advice behind a TLS
reverse proxy. Both are supported, because guessing wrong yields a 401 that
says nothing about which was expected — and `check()` says exactly that instead
of picking one.

A `duplicates` reply is reported as *already there*, not as an error.

### kindle

Plain SMTP with the document attached. Two of Amazon's rules are enforced
*after* the SMTP transaction has already succeeded, so both are handled before
it starts:

* **The sender must be on Amazon's approved list.** Mail from an unrecognised
  address is dropped in silence — SMTP says 250, nothing arrives, nothing
  reports a failure. The success message therefore promises only what happened
  (the mail was accepted for delivery) and names the approved-sender list.
* **There is a size ceiling** around 50 MB. An oversized file is refused
  *before the connection is opened*, naming the limit and the actual size.

For Gmail, iCloud and Outlook the password must be an **app-specific** one.
That is the single most common first-send failure, so the field says so and
`_explain()` turns `SMTPAuthenticationError` into that sentence rather than
into the server's text.

### corpus

`POST {base}/book/upload`, multipart, `X-API-Key` header. Aglaïa already knows
the document's metadata, so it sends it — retyping a title into a web form is
the tedium this destination exists to remove. Only fields that have a value are
sent: an empty field means *erase this* to the corpus's metadata route, and the
same habit here would overwrite a harvested record with nothing.

The API's four outcomes stay four:

| code | meaning | reported |
|---|---|---|
| 201 | added | success, with the id |
| 200 | same title and author already in base | already there |
| 400 | extension not admitted, or empty | refused, with the reason |
| 413 | over 2 GiB | refused, naming the limit |

Inadmissible extensions are refused locally against the API's own list — a
round trip to be told "no" is a round trip wasted.

## Where they appear

An exporter is an exporter. An installed export plugin gets **the same card as
PDF and Markdown**, in the same list, selected the same way, run by the same
**Export** button (card key `send:<slug>`). It was briefly a separate *Send to*
strip below the button with its own send buttons, which meant two ways to start
an export and two shapes of control for one idea.

A destination that accepts more than one format Aglaïa can write gets a small
**Export as** picker in the card's extras; one that accepts a single format
resolves silently. Formats Aglaïa cannot produce never appear — Kindle accepts
`epub` and `docx`, and the card offers neither.

A destination that is not configured says **"Not set up yet — needs …"** on the
card, and pressing Export opens its settings rather than running an export that
would be thrown away.

**The export goes to a private staging directory, not to a file the user
names.** A file that exists only to be handed to calibre is a courier, not a
deliverable: asking for a folder and a filename for something the user will
never open is a dialog for nothing, and it invites the one failure a courier
must not have — overwriting last week's export because both are called
`Book.pdf`. Each send gets a fresh `mkdtemp`, so two sends of the same book
cannot collide. The *filename* is kept exactly as the save dialog would have
proposed it, because it is not incidental: Kindle attaches the file under that
name, and calibre reads a book title out of it. The courier is deleted once the
plugin has had it, whether the send succeeded or not.

A normal export is untouched: still asked for, still kept, still revealed in
the Finder.

There is also a **plug icon** in the right-hand rail, above Settings, that
opens the Plugins tab.

## Reporting a failure

`CheckResult` carries a `kind` alongside its message, because "could not
connect" and "wrong password" have nothing in common except the word *failed*,
and a user who reads only the first two words should already be looking in the
right place. Modelled on Thunderbird's taxonomy:

| `kind` | Shown as | Means |
|---|---|---|
| `CheckResult.NETWORK` | Cannot connect | could not reach it at all |
| `CheckResult.AUTH` | Wrong credentials | reached it; it rejected them |
| `CheckResult.PERMISSION` | Not allowed | signed in; not permitted to do this |
| `CheckResult.SERVER` | Server problem | reached it; it reported a problem |
| `CheckResult.CONFIG` | Missing setting | a setting is wrong, before any I/O |
| `CheckResult.UNKNOWN` | Failed | genuinely cannot tell |

`UNKNOWN` is a real member, not a gap. A taxonomy without one gets quietly
widened until every failure is `NETWORK`.

**`message` is for the user; `detail` is for the log.** The host shows one and
logs the other. A plugin that concatenates them produces a dialog nobody can
read and a log line nobody can search — so the status code, the exception type
and the server's own words go in `detail`. This is checked by
`tests/plugins/test_destinations.py`.

Check the settings before touching the network. A round trip that can only fail
costs the user the wait and then reports "could not connect", which sends them
to look at their server instead of at the empty field.

See [ui-writing.md](ui-writing.md) for how to word any of it.

## Listing them

```bash
aglaia list destinations
```

prints each destination, what it accepts, and either `ready` or the settings it
is still missing.

## Writing your own

Drop a directory in `<APP_DATA>/plugins/destinations/<slug>/` with an
`aglaia-plugin.toml` and one module. Discovery requires the manifest's `slug`
to equal the directory name — the directory decides, because the slug also
decides the keychain namespace and the settings file, and that is not a thing
to guess at. `requires.api` must match the host's `plugin_api.API_VERSION`; a
mismatch is refused with both numbers named, rather than loading and failing
later somewhere less obvious.

Import from `aglaia.plugin_api` and nowhere else under `aglaia`.
