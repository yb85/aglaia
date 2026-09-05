# Writing for the user

Aglaïa's user-facing text is written for the person using it. Everything about
*how the program works* — what failed to import, which class was not decorated,
what the exception type was — belongs in the log.

That split is the whole guide. Everything below is how to hold it.

The rules come from measuring ~35 000 real strings from thirty open-source
applications praised for their craft, plus the four projects that publish a
written copy standard (Mozilla Photon and Acorn, Element Compound, the Rust
compiler's diagnostics guide). Sources are named throughout; where a rule is
just taste, it says so.

---

## 0. The one rule

> **The UI says what happened to the user. The log says what happened in the
> program.**

Not two catalogues — two *registers*. The failure is almost never that a log
line is literally reused in a dialog; Aglaïa has zero of those. The failure is
that UI text gets written in log voice: mechanism first, lowercase fragment,
exception spliced in, an identifier the reader cannot act on.

Here is the defect in its purest form, shipped in `destinations.load_error()`
and shown in a `QMessageBox`:

> *"it imported but registered no destination — is the class decorated with
> `@register_destination`, and does its `name` match the plugin slug?"*

Every word is true and it is addressed to the wrong person. Whoever reads it
installed a plugin from the registry; they cannot decorate anything. The
decorator question belongs in the log and in the plugin `CONTRIBUTING.md`. The
dialog should say the plugin is broken, name it, and offer to remove it.

The test, applied to any string before it ships:

**Could the reader act on this sentence?** If the answer needs them to open the
source, it is a log line.

Transmission is the cautionary tale: its GUI catalogue and its `libtransmission`
log strings are one file, so `Couldn't bind IPv6 socket {address}: {error}
({error_code})` sits beside genuinely excellent copy, and both get translated.

### What the log gets

Everything the UI is not allowed to keep: exception type and message, module
and function, the slug, the URL, the return code, the byte counts, the timings.
Write log lines for yourself at 3 a.m. six months from now. Be as technical as
you like. Nobody is translating them.

---

## 1. Length

Measured across the corpus. Two bands, and almost nothing sits between them.

| Where | Words | Hard cap |
|---|---|---|
| Progress / busy caption | **2–3** | 6 |
| Field label | **1–3** | 5 |
| Button | **1–2** | 3 |
| Error, inline | **6–9** | **80 characters** |
| Confirmation body | **13–20** | 2 sentences |
| Field help / tooltip | **10–15** | 25, and only for a knob the label cannot carry |
| Long-form (disclosure panel, docs) | **40–90** | — |

The 80-character error cap is Element Compound's published rule, and our
measurements corroborate it: median errors run 5–8 words across every app
measured, p90 lands at 15–17.

Two calibration points worth keeping in mind:

- **Proton Bridge configures an entire IMAP/SMTP client in 29 help strings, none
  over 27 words.** Its whole port panel is seven words: *"Changes require
  reconfiguration of your email client."* Not a definition of a port — the
  consequence of changing it.
- **Tooltips are the exception to brevity, and the best apps are the longest.**
  Anki's median scheduler tooltip is 25 words with a p90 of 109; HandBrake keeps
  a separate `ResourcesTooltips.resx` whose max is 216. When a parameter cannot
  be reasoned about from its label, the tooltip stops being a label and becomes
  inline documentation. Aglaïa's dewarp arch/tilt, Wolf window and DPI clamps
  are exactly this case. **Long is allowed when the length is carrying real
  information; it is never allowed for rationale.**

Leave a third of every label's width free. English grows by up to 50 % in
French and German (Mozilla L10n).

---

## 2. The message has slots

The highest-value finding in the research, and it is structural rather than
stylistic.

Elm's snippet renderer takes a **pair** — `(preHint, postHint)`, what happened
and what to do, with the evidence rendered *between* them. You cannot construct
an Elm error that omits either half: the type demands both. Homebrew's
`Finding::Remediation.new(text:, commands: [])` does the same for `brew doctor`.

> If your message type is a `str`, you will ship messages that are only a cause.

Aglaïa's slots:

| Slot | Job | Length |
|---|---|---|
| **Title** | Names the failure, not the subsystem | 1–5 words |
| **What happened** | The user's situation | 6–9 words |
| **Evidence** | The path, the value, the page — indented, monospace, never inline prose | — |
| **What to do** | The only slot allowed to propose a change | 1 sentence |
| **Note** | Context they should know and cannot act on | 1 sentence |

Rust states the separation outright: *"The error or warning portion should not
suggest how to fix the problem, only the 'help' sub-diagnostic should."* And
`help` and `note` are not interchangeable — `help` is something to try, `note`
is something to know. Elm uses `Hint:` 50 times and `Note:` 110 times and never
confuses them.

**Omit a slot rather than pad it.** Deno's hint is an `Option`; the line is
simply absent when no pattern matched. Nothing in the corpus writes "please try
again" to fill a hole.

In a 300 px sidebar the slots become: bold one-liner → the value or button →
a disclosure triangle. Write the status label to Rust's *label* budget (median
5 words) and the disclosure panel to Elm's *report* budget (60–120 words).

---

## 3. Case and punctuation

**Sentence case for everything we draw.** Labels, buttons, headings, card
titles, error text. Aglaïa is already at **93.3 %** (235 of 252 short labels) —
close to Calibre, the most disciplined app measured at 96.6 %.

*Exception: the native macOS menu bar.* Title case there, because Apple's HIG
says so and the OS convention is more visible than ours. `About Aglaïa`,
`Close Project`, `Quit Aglaïa` are correct. `Full Calibration`,
`Model Downloader`, `Activate Capture` are not — those are ours to draw.

The industry moved: Mozilla's Acorn reverses Photon on this, Element Compound
gives the reasons (*"It looks cleaner. It's easier to use consistently. It's
easier for some people to read (e.g. people with dyslexia)."*). What actually
matters is that the choice is total — the apps stuck at 30–60 % are the ones
where the convention was never written down.

**Terminal punctuation:** none on anything ≤4 words; a period on anything that
is a full sentence. The corpus measures 3–9 % vs 60–81 %. This is the most
consistent rule found anywhere.

**No colon on field labels.** `Port`, not `Port:`. Proton Bridge, Thunderbird
for Android, Bitwarden, Nextcloud and Joplin all agree; only Thunderbird's older
desktop file uses colons, and its own rewrite dropped them.

**Ellipsis is `…` (U+2026), never `...`.** On a button it means a dialog opens;
ending a status it means work is in progress. We have 66 correct and 3 wrong.

**Never concatenate translated fragments**, and never translate a partial
sentence. Element's smell test: *if a string does not begin with a capital, or
ends with `:` or a preposition, you are probably translating half a sentence.*

---

## 4. Person and voice

- **The user is "you".** Their data is "your project", "your scans".
- **Aglaïa is named, in the third person.** *"Aglaïa could not reach the
  server."* Never "I". Elm's relentless first person works because it never
  breaks character across 426 sentences; half-committing is worse than either.
- **"We" only when a person is genuinely acting** — a decision made by whoever
  maintains the registry, not by the program. *"We review every plugin before
  merging it."* Never "We couldn't open the file."
- **Buttons are imperative verbs.** `Save`, `Send`, `Install`, `Remove`,
  `Set up`, `Try again`.
- **Setting descriptions are declarative.** *"Aglaïa will re-run this page when
  a parameter changes."* The description says what will happen; the button says
  what to do. Never merge them into one string.
- **Passive is fine when the actor is genuinely irrelevant**: `Settings saved`,
  `Connection verified`.

---

## 5. Errors

**Open with the subject that failed.** `Error` belongs in a title, never in a
body.

**System failure:** *what failed, because why. What to do.*

> "The add-on could not be downloaded because of a connection failure."
> — Firefox

> "The server took too long to respond. Check your connection and try syncing
> again. If it still doesn't work, reach out to your server administrator."
> — Nextcloud

**User input:** give the fix, skip the diagnosis.

> "Port is invalid (must be 1–65535)." — Thunderbird

Not *"You entered an invalid port."* **Never assign blame** — Element Compound
and Mozilla Photon both forbid it in writing. Bitwarden's
`"Login attempt failed with incorrect password."` breaks its own standard twice:
it blames, and it confirms to an attacker that the account exists. Nextcloud's
`"Authentication error: Either username or password are wrong."` is the fix.

**Name every cause you can actually distinguish, and include an unknown
branch.** Thunderbird for Android is the model, and it maps 1:1 onto what
`Destination.check()` already returns:

| Their taxonomy | Ours |
|---|---|
| `Network error` | cannot reach the server |
| `Authentication error` | credentials rejected |
| `Server error` | reached, refused |
| `Missing server capability` | authenticated, not permitted |
| `Unknown error` | — *we lack this branch* |

Two words each, with the machine detail below under `Details:`. Never collapse
"cannot reach" into "credentials rejected"; they have different fixes.

For the "authenticated but not permitted" case, Thunderbird's prose form does
three things in 34 words — rules out the credential hypothesis, names the
responsible human, gives a next action:

> "Configuration could not be verified. If your username and password are
> correct, it's likely that the server administrator has disabled the selected
> configuration for your account. Try selecting another protocol."

**Raw technical detail is allowed — beneath a plain sentence, behind a label.**
`Details:`, `The reported error was:`. Never as the whole message.

**Say who can fix it when it is not the user.** Nextcloud does this in eight of
its fifteen top-level server errors.

**Reassure about what did *not* break.** *"Your scans are unchanged."*
Proton: *"Email clients stay connected to Bridge."*

**Generate the fix from the user's own input.** fish re-escapes your broken
token into `Did you mean \`set foo 'ba nana'\`?`. In a GUI this is stronger than
in a terminal, because the fix can be a button. Not *"adjust the dewarp
settings"* but **`Set arch to 0.35`**.

Rust grades suggestions by confidence, which is worth copying as widget
behaviour: machine-applicable → a one-click button; has-placeholders →
pre-filled but editable; maybe-incorrect → prose only.

**Do not suggest with a question.** Rust bans "did you mean" outright:
*compare "did you mean: `Foo`" with "there is a struct with a similar name:
`Foo`".* A dialog may ask a question; a persistent label that asks one is
nagging, and it re-renders on every repaint.

**Refusals have their own shape:** *refusing to X because Y; to do it anyway,
Z.* Four unrelated projects converged on it independently.

---

## 6. Settings and field help

**Explain the consequence, never the mechanism.**

> "Changes require reconfiguration of your email client." — Proton Bridge
> "Enabling will reduce call quality." — Signal
> "Attention: DNS settings might not go into effect immediately." — Mullvad

**Boolean → one sentence.** 56 of VS Code's 381 settings descriptions are
literally `Controls whether …`. Non-boolean → a noun phrase naming the unit.

**Units go in the label, in parentheses.** `Timeout (seconds)`,
`Output folder (absolute path)`. Never a separate line saying "in seconds".

**Ranges use an en dash, inline.** `must be 1–65535`.

**Always spell out the sentinel value.** `Set to 0 for no limit.`
`Leave empty to use the default.` Four words that prevent a support question.

**Give a literal example instead of describing a format.** The most transferable
technique found:

> "Example: https://bitwarden.company.com" — Bitwarden
> "Example: .mozilla.org, .net.nz, 192.168.1.0/24" — Firefox
> placeholder `john.doe@example.com` — Thunderbird

**State the default only where the widget does not show it.** VS Code omits it
97 % of the time because the settings control renders it; rustc and Ghostty
state it 45 % and 60 % of the time because they have no widget. Aglaïa's
settings panel shows the control — so do not repeat the default in prose.

**Do not explain why the default is the default.** This is the single most
common way developer reasoning leaks into the UI, and the corpus is nearly
unanimous against it — the "default is X *because* Y" construction appears a few
dozen times in 35 000 strings. It reads as fascinating to whoever chose the
default and as noise to everyone else. (Where it genuinely earns its place, the
information is a *constraint the user shares* — Calibre's *"The default is the
size required for Adobe Digital Editions"* — not a note on our reasoning.)

**Enum members get one line each.** The parent description names the dimension
being chosen, not the members.

**Restart requirements, platform limits and runtime-effect caveats each get
their own trailing sentence.** Never buried mid-paragraph.

---

## 7. Secrets

**Say where it goes, in one sentence, once.**

> `<what>` is `<protection>` and stored `<where>`.

> "Your backup is end-to-end encrypted and stored on your computer." — Signal
> "Your credentials will only be stored locally on your computer." — Thunderbird

Nine to twelve words. No threat model, no rationale.

**Where a secret is stored is the host's fact, not the plugin's.** It is
identical for every plugin, so it is said once above the secret fields — not
repeated in each plugin's field help. And it must be *true*: it reads the live
keyring state, so on a machine with no keychain it says the value is written as
plain text instead of claiming a keychain that is not there. Zotero does this
unprompted after a failed encryption — *"Your credentials remain stored
unencrypted on disk."* — and silently degrading instead is the wrong call.

**We say "keychain", one word, on every platform.** Zotero maintains three
per-platform variants so it can say "Keychain", "Credential Manager" and
"keyring service such as GNOME Keyring or KWallet". That is defensible for a
cross-platform product; Aglaïa is macOS-first, and three strings per message to
translate is not worth it here. **This is a deliberate departure, not an
oversight.**

**Disambiguate a password with a parenthetical inside the instruction, not a
banner.** Proton Bridge solves our exact Kindle problem in six words:

> "Use the password below (not your Proton password), when adding your Proton
> account to %1."

Its password *label* also mutates to "Use this password" and turns
warning-coloured, with no separate warning string at all.

**Link to the thing that produces the credential** rather than explaining what
it is. Nextcloud: *"Click here to request an app password from the web
interface."* Our Kindle plugin should link to Amazon's approved-senders page.

---

## 8. Destructive actions

**Consequence first, then the question.** 252 confirmation strings across nine
apps agree, against the common modern advice to drop "Are you sure":

> "The selected book will be **deleted** and the files removed from your calibre
> library. Are you sure?" — Calibre

**Escalate ceremony with blast radius**, and encode the risk in the *default*:

| Tier | Mechanism | In Aglaïa |
|---|---|---|
| Irreversible | type the object's name | wiping a project's derived images |
| Recoverable but costly | `y/N`, default **no** | discarding manual per-page tuning |
| Trivially redone | `y/N`, default **yes** | re-running a step |

**Fork the wording, not just the behaviour**, on recoverability. VS Code:
*delete* + "You can restore this file from the Trash" versus *permanently
delete* + "This action is irreversible!" Use the OS's own noun.

**The button repeats the verb.** `Move to Trash`, `Delete permanently`,
`Discard tuning` — never `OK`.

**Warn about the second-order loss.** *"4 of these pages have hand-tuned dewarp
settings, which will be lost."*

**Name the residue you did not remove.** *"The OCR text was kept; only the page
images were regenerated."*

**ALL CAPS is a budget, spent once**, on genuinely unrecoverable loss.

**Put the off-switch inside the confirmation.** A confirmation nobody can retire
becomes noise, and noise gets clicked through.

**Leave a one-line record of the decision afterwards.** Silence after a
destructive confirmation is indistinguishable from a no-op.

---

## 9. Third-party code

**State the risk once, flatly. List the mitigations. Then let them proceed.**
Joplin's plugin-security screen is the model: one comparative sentence — *"Like
any software you install, plugins can potentially cause security issues or data
loss."* — then three concrete mitigations, then `Enable plugin support`. No
scare copy, no "are you sure".

**Attribute a third party's claims to the third party.** Firefox refuses to
vouch, inside the sentence:

> "The developer says this extension collects: …"

Ours: *"The plugin author says this plugin sends …"*

**Permission lines are verb-first, second-person object, zero API identifiers.**
Element: *"See messages posted to this room"*, *"Change the topic of this
room"*.

---

## 10. Never write these in the UI

A published do-not-use list is the artefact most projects lack; Mozilla's
`word-list` is the model. Ours:

| Never | Because |
|---|---|
| An environment variable name (`CORPUS_APP_KEY`, `MISTRAL_API_KEY`) | The user did not set it and cannot find it |
| An HTTP header (`X-API-Key`, `Content-Type`) | Wire detail |
| A class, function or decorator (`@register_destination`, `ImageBuffer`) | Addressed to a developer |
| A module or file path in our source | ditto |
| An exception type spliced into prose | This is log voice |
| `.env`, `sys.path`, `sqlite`, "the registry", "the host" | Internal vocabulary |
| A `uv sync --extra …` command | The user installed a `.dmg` |
| Rationale for a default | Interesting to us, noise to them |
| "An error occurred" with no subject | Says nothing |
| `page(s)`, `1 files` | Write the plural helper instead |

**But do name anything the user can act on**: a protocol they are copying from
another settings page (`SMTP`, `STARTTLS`, `IMAP`, `PDF`, `OCR` — bare, never
expanded; expanding them makes matching *harder*), a file they can open, a
folder they can find, a menu item in another app, a pipeline YAML key they edit.

Name our surfaces the way the user sees them — "the scans column", "the import
panel" — never `IntegratedProcessingChain`.

And introduce our *own* coinages before using them. Element: *"We call the
places where you can host your account 'homeservers'."* Aglaïa's are "branch",
"node", "layout", "span" — every one needs this on first use, or replacing.

---

## 11. Grammar is infrastructure

Elm ships `args n`, `intToOrdinal` (with the 11/12/13 exception) and `commaSep`,
so nothing it prints ever says `1 arguments`. Write the helpers once:

```python
plural(n, "page", "pages")     # never "page(s)"
oxford_list(names)             # "a, b, and c"
```

An English UI written in a French-speaking household needs these more than most.

---

## 12. Where Aglaïa stands

Measured over 939 `tr()` strings, 21 plugin `Field(help=…)` strings and 41
plugin result messages:

| | |
|---|---|
| Short labels in sentence case | 93.3 % ✅ |
| Strings splicing a raw exception into the UI | 3 ⚠️ |
| Strings shared between log and UI | 0 ✅ |
| `…` vs `...` | 66 / 3 ⚠️ |
| Straight vs smart apostrophes | 32 / 0 — pick one |
| Median `tr()` length | 3 words ✅ |
| Median plugin field help | 9 words ✅ |
| Strings naming an env var, header, path or flag | 12 ❌ |

The sweep is tracked in **#138**. The catalogue is in better shape than it
looked; the defect is concentrated in the failure paths and the advanced
settings, which is exactly where the corpus says every project's defects
concentrate.

---

## Checklist

Before shipping a user-visible string:

1. Could the reader **act** on it? If acting requires reading our source, it is
   a log line.
2. Does it name an env var, header, class, module or exception type? Move that
   to the log.
3. Does it explain **why** rather than **what**? Delete the why, or make it a
   code comment.
4. Is there a **fix slot**, and is it a specific value or gesture rather than a
   category?
5. Sentence case, no colon on labels, `…` not `...`, period only on full
   sentences.
6. Within the word budget for its kind (§1)?
7. Is it a **whole sentence**, not a fragment to be concatenated?
