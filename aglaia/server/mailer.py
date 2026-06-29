# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Completion email (#52, slice 4).

On a finished job (when the submitter asked for `email_notif`), send an email
with download links to the PDF and Markdown. SMTP settings live in the server
`config` table (``smtp`` = {host, port, user, password, from, tls}); if unset,
sending is skipped silently.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from aglaia.server import db as sdb


def download_url(base_url: Optional[str], job_id: str, which: str, token: str) -> str:
    base = (base_url or "").rstrip("/")
    return f"{base}/download/{job_id}/{which}?token={token}"


def send_completion(db_file: Path, *, to: str, job, base_url: Optional[str]) -> bool:
    """Send the completion email. Returns False (no-op) if SMTP isn't configured."""
    with sdb.session(db_file) as conn:
        smtp = sdb.get_config(conn, sdb.CONFIG_SMTP)
    if not smtp or not smtp.get("host"):
        return False

    job_id, token = job["id"], job["download_token"]
    links = []
    if job["pdf_path"]:
        links.append(("PDF", download_url(base_url, job_id, "pdf", token)))
    if job["md_path"]:
        links.append(("Markdown", download_url(base_url, job_id, "md", token)))
    body = "Your Aglaïa scan is ready.\n\n" + "\n".join(f"{label}: {url}" for label, url in links)

    msg = EmailMessage()
    msg["Subject"] = "Your Aglaïa scan is ready"
    msg["From"] = smtp.get("from") or smtp.get("user") or "aglaia@localhost"
    msg["To"] = to
    msg.set_content(body)

    with smtplib.SMTP(smtp["host"], int(smtp.get("port", 587)), timeout=20) as server:
        if smtp.get("tls", True):
            server.starttls()
        if smtp.get("user"):
            server.login(smtp["user"], smtp.get("password", ""))
        server.send_message(msg)
    return True
