# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/
"""Ephemeral self-signed TLS certificate for the aglaia-bridge receiver (#47).

Trust model: the QR shown on screen is the trusted side channel, the LAN is
hostile. The desktop generates a fresh per-session self-signed cert; the phone
pins it by the SHA-256 fingerprint carried in the QR. So the cert needs no CA
and no valid hostname — the fingerprint *is* the trust anchor.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


@dataclass(frozen=True)
class EphemeralCert:
    cert_path: Path
    key_path: Path
    fingerprint_sha256: str  # lowercase hex of the DER cert — what the phone pins


def generate_ephemeral_cert(*, host_ip: str | None = None) -> EphemeralCert:
    """Create a per-session self-signed cert and write it to a temp dir."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "aglaia-bridge")])
    now = datetime.datetime.now(datetime.UTC)

    san: list[x509.GeneralName] = [x509.DNSName("aglaia-bridge.local")]
    if host_ip:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(host_ip)))
        except ValueError:
            pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(hours=1))   # tolerate clock skew
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )

    der = cert.public_bytes(serialization.Encoding.DER)
    fingerprint = hashlib.sha256(der).hexdigest()

    tmp = Path(tempfile.mkdtemp(prefix="aglaia-bridge-tls-"))
    cert_path = tmp / "cert.pem"
    key_path = tmp / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return EphemeralCert(cert_path=cert_path, key_path=key_path, fingerprint_sha256=fingerprint)
