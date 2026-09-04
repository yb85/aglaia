# SPDX-License-Identifier: MIT
"""Mail a finished export to a Kindle — or to any address.

Amazon's "Send to Kindle" is an email inbox: mail a document to a device's
`@kindle.com` address and it appears on the device. There is no API and no key,
which makes this the simplest destination to describe and the easiest to get
subtly wrong, because two of its rules are enforced by Amazon *after* the SMTP
transaction has already succeeded:

* **The sender must be on the approved list.** A mail from an address Amazon
  does not recognise is dropped in silence. SMTP says 250, the document never
  arrives, and nothing anywhere reports a failure. So the success message says
  what it can actually promise — that the mail was accepted for delivery — and
  the field help says where the approved list lives.
* **There is a size ceiling** (about 50 MB). Discovering it after a ten-minute
  upload is a bad way to learn it, so an oversized file is refused before the
  connection is opened, naming the limit and the actual size.

Credentials go in the keychain; everything else in the plugin's own settings.
For Gmail and iCloud the password must be an app-specific one — that is the
single most common reason a first send fails, so the field says so rather than
leaving the user to read an SMTP error about it.
"""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from aglaia.plugin_api import (
    BookMeta, CheckResult, Destination, Field, SendResult, register_destination,
)

#: Extension → (maintype, subtype) for the attachment. Anything unlisted goes
#: as application/octet-stream, which Amazon rejects — better to say so.
_MIME = {
    ".pdf": ("application", "pdf"),
    ".epub": ("application", "epub+zip"),
    ".txt": ("text", "plain"),
    ".md": ("text", "markdown"),
    ".docx": ("application",
              "vnd.openxmlformats-officedocument.wordprocessingml.document"),
}


@register_destination
class KindleDestination(Destination):
    name = "send-to-kindle"
    display = "Export to Kindle by email"
    description = "Mail the export to a Kindle address, or to anyone."
    accepts = ("pdf", "epub", "txt", "md", "docx")

    CONFIG_FIELDS = (
        Field("recipient", "Send to", "str", "", required=True,
              placeholder="your-device@kindle.com",
              help="The device's Send-to-Kindle address. Any address works — "
                   "this is just email."),
        Field("sender", "From", "str", "", required=True,
              placeholder="you@example.org",
              help="Must be on Amazon's approved sender list, or the mail is "
                   "accepted and then silently discarded. Amazon account → "
                   "Preferences → Personal Document Settings."),
        Field("smtp_host", "SMTP server", "str", "", required=True,
              placeholder="smtp.gmail.com"),
        Field("smtp_port", "Port", "int", 587,
              help="587 for STARTTLS, 465 for implicit SSL, 25 unencrypted."),
        Field("security", "Encryption", "choice", "starttls",
              choices=("starttls", "ssl", "none"),
              help="STARTTLS on 587 is the usual answer."),
        Field("subject_template", "Subject", "str", "{title}",
              help="{title} {author} {year} {filename} are substituted. "
                   "Amazon ignores the subject; a human recipient will not."),
        Field("body_template", "Message", "str",
              "Sent by Aglaïa.\n\n{title}\n{author}\n",
              help="Same substitutions."),
        Field("max_attachment_mb", "Size limit (MB)", "int", 45,
              help="Refuse before uploading. Amazon's ceiling is around 50; "
                   "45 leaves room for the MIME encoding overhead."),
    )
    SECRET_FIELDS = (
        Field("smtp_user", "SMTP username", "secret", "", required=True,
              help="Usually the full email address."),
        Field("smtp_password", "SMTP password", "secret", "", required=True,
              help="For Gmail, iCloud or Outlook this must be an "
                   "APP-SPECIFIC password, not your account password. Your "
                   "normal password will be refused."),
    )

    # ── plumbing ──────────────────────────────────────────────────────
    def _fill(self, template: str, path: Path, meta: BookMeta) -> str:
        return str(template or "").format(
            title=meta.title or path.stem,
            author=meta.author or "",
            year=meta.year or "",
            filename=path.name,
        )

    def _connect(self) -> smtplib.SMTP:
        """An open, authenticated connection. Raises on any failure — the
        callers turn that into a message, and they want the exception type."""
        host = str(self.conf("smtp_host") or "").strip()
        port = int(self.conf("smtp_port") or 587)
        security = str(self.conf("security") or "starttls")
        if security == "ssl":
            smtp: smtplib.SMTP = smtplib.SMTP_SSL(
                host, port, timeout=60, context=ssl.create_default_context())
        else:
            smtp = smtplib.SMTP(host, port, timeout=60)
            if security == "starttls":
                smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        user, pwd = self.secret("smtp_user"), self.secret("smtp_password")
        if user:
            smtp.login(user, pwd)
        return smtp

    def _explain(self, e: Exception) -> str:
        """SMTP errors are precise; the point is to say what to change."""
        if isinstance(e, smtplib.SMTPAuthenticationError):
            return ("The server rejected the username or password. For Gmail, "
                    "iCloud or Outlook this must be an app-specific password, "
                    "not your account password.")
        if isinstance(e, smtplib.SMTPNotSupportedError):
            return (f"The server does not support that: {e}. Try a different "
                    f"encryption setting — 587/STARTTLS or 465/SSL.")
        if isinstance(e, (smtplib.SMTPConnectError, OSError)):
            return (f"Cannot reach {self.conf('smtp_host')}:"
                    f"{self.conf('smtp_port')} — {type(e).__name__}. Check the "
                    f"server, the port, and whether the network allows it.")
        if isinstance(e, smtplib.SMTPRecipientsRefused):
            return f"The server refused the recipient: {e.recipients}"
        return f"{type(e).__name__}: {e}"

    # ── check ─────────────────────────────────────────────────────────
    def check(self) -> CheckResult:
        """Connect, negotiate, log in, hang up. Proves host, port, encryption
        and credentials without putting a test document in anyone's library —
        which matters here, because a Kindle library is a place you cannot
        tidy from the desktop."""
        missing = self.missing_settings()
        if missing:
            return CheckResult(False, "Still needed: " + ", ".join(missing))
        try:
            smtp = self._connect()
        except Exception as e:  # noqa: BLE001 — every failure is a message
            return CheckResult(False, self._explain(e))
        try:
            smtp.quit()
        except Exception:
            pass
        sender = str(self.conf("sender") or "")
        return CheckResult(
            True, f"Signed in to {self.conf('smtp_host')} as "
                  f"{self.secret('smtp_user')}. Mail will be sent from "
                  f"{sender} — make sure Amazon has it on the approved "
                  f"sender list.")

    # ── send ──────────────────────────────────────────────────────────
    def send(self, path: Path, meta: BookMeta) -> SendResult:
        path = Path(path)
        if not path.is_file():
            return SendResult(False, f"No such file: {path}")
        missing = self.missing_settings()
        if missing:
            return SendResult(False, "Still needed: " + ", ".join(missing))

        limit_mb = int(self.conf("max_attachment_mb") or 45)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > limit_mb:
            return SendResult(
                False, f"{path.name} is {size_mb:.0f} MB, over the "
                       f"{limit_mb} MB limit. Export a smaller PDF (G4 "
                       f"bitonal is much smaller than colour) or raise the "
                       f"limit if your provider allows it.")

        maintype, subtype = _MIME.get(path.suffix.lower(),
                                      ("application", "octet-stream"))
        msg = EmailMessage()
        msg["From"] = str(self.conf("sender"))
        msg["To"] = str(self.conf("recipient"))
        msg["Subject"] = self._fill(self.conf("subject_template"), path, meta)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="aglaia.local")
        msg.set_content(self._fill(self.conf("body_template"), path, meta))
        msg.add_attachment(path.read_bytes(), maintype=maintype,
                           subtype=subtype, filename=path.name)

        try:
            smtp = self._connect()
        except Exception as e:  # noqa: BLE001
            return SendResult(False, self._explain(e))
        try:
            smtp.send_message(msg)
        except Exception as e:  # noqa: BLE001
            return SendResult(False, self._explain(e))
        finally:
            try:
                smtp.quit()
            except Exception:
                pass

        # Say what was actually achieved. Delivery is Amazon's to decide, and
        # a plugin that claims "delivered" for a mail that may be silently
        # dropped is worse than one that claims nothing.
        return SendResult(
            True, f"Mailed {path.name} ({size_mb:.1f} MB) to "
                  f"{self.conf('recipient')}. Amazon delivers only from "
                  f"approved senders — if it does not appear, check that "
                  f"{self.conf('sender')} is on the list.",
            detail={"bytes": path.stat().st_size})
