# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""`aglaia server` — the warm-pool HTTP job API (#52)."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from aglaia.server import DEFAULT_PORT


def server(
    host: Annotated[str, typer.Option("--host", help="Bind address (use 0.0.0.0 to accept LAN/remote clients).")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = DEFAULT_PORT,
    public_url: Annotated[Optional[str], typer.Option("--public-url", help="Public base URL for download links in emails, e.g. https://scan.example.com.")] = None,
) -> None:
    """Run the long-running HTTP job server (needs the `server` extra:
    `pip install \"aglaia[server]\"`)."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("The server needs the 'server' extra: pip install \"aglaia[server]\"", err=True)
        raise typer.Exit(1) from None

    from aglaia.server import db as sdb
    from aglaia.server.app import create_app

    with sdb.session() as conn:
        secret = sdb.ensure_admin_secret(conn)
        if public_url:
            sdb.set_config(conn, sdb.CONFIG_BASE_URL, public_url.rstrip("/"))
    app = create_app()
    typer.echo(f"Aglaïa server → http://{host}:{port}")
    typer.echo(f"  admin panel: http://{host}:{port}/admin?secret={secret}")
    uvicorn.run(app, host=host, port=port)
