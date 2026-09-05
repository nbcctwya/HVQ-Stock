#!/usr/bin/env python3
"""Phase 2 Supervisor mail notifier (SMTP delivery only).

This is a deterministic, experiment-agnostic sender used by the Phase 2
Supervisor layer. It does NOT read the queue, does NOT understand
experiment logic, and does NOT classify failures — the Supervisor composes
the subject and body; this module only delivers them.

Notification failure is never an experiment failure: a non-zero exit code
here must not change queue status, markers, artifacts, or any runner
result. The Supervisor records "Notification: FAILED" in its report and
moves on.

Credentials come exclusively from environment variables:

    QQ_SMTP_EMAIL          sender QQ mail address
    QQ_SMTP_AUTH_CODE      QQ SMTP authorization code
    RESEARCH_NOTIFY_EMAIL  recipient address

The authorization code is never written to code, logs, CLI arguments, or
repository files, and is scrubbed from any error message.

Usage:

    python experiments/notify.py \
        --subject "[HVQ] Phase2 completed — 4 DONE / 1 FAILED" \
        --body-file /tmp/phase2_summary.txt

``--body`` is accepted as a fallback, but ``--body-file`` is the primary
interface. Exit code 0 on success; missing configuration, a missing body
file, or an SMTP failure prints a clear (credential-free) error to stderr
and exits non-zero.
"""

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465  # SSL
SMTP_TIMEOUT = 30

ENV_SENDER = "QQ_SMTP_EMAIL"
ENV_AUTH = "QQ_SMTP_AUTH_CODE"
ENV_RECIPIENT = "RESEARCH_NOTIFY_EMAIL"


class NotifyError(RuntimeError):
    pass


def load_config(env=None):
    """Read sender/auth/recipient from the environment. Never logs values."""
    env = os.environ if env is None else env
    missing = [k for k in (ENV_SENDER, ENV_AUTH, ENV_RECIPIENT) if not env.get(k)]
    if missing:
        raise NotifyError(
            "missing required environment variables: " + ", ".join(missing)
        )
    return env[ENV_SENDER], env[ENV_AUTH], env[ENV_RECIPIENT]


def read_body(body=None, body_file=None) -> str:
    if body_file:
        p = Path(body_file)
        if not p.is_file():
            raise NotifyError(f"body file not found: {p}")
        return p.read_text(encoding="utf-8")
    if body is not None:
        return body
    raise NotifyError("no body given: pass --body-file (preferred) or --body")


def _scrub(text: str, secret: str) -> str:
    """Guarantee the auth code never appears in an error message."""
    if secret:
        text = text.replace(secret, "***")
    return text


def send_mail(sender: str, auth_code: str, recipient: str,
              subject: str, body: str) -> None:
    """Send one UTF-8 mail via QQ SMTP (SSL). Raises NotifyError on failure."""
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body, charset="utf-8")
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
            server.login(sender, auth_code)
            server.send_message(msg)
    except Exception as e:  # noqa: BLE001 - report any SMTP failure scrubbed
        raise NotifyError(
            f"SMTP send failed: {_scrub(f'{type(e).__name__}: {e}', auth_code)}"
        ) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="Mail subject (UTF-8).")
    parser.add_argument("--body", default=None,
                        help="Mail body text (fallback; --body-file is preferred).")
    parser.add_argument("--body-file", default=None,
                        help="Path to a UTF-8 text file used as the mail body.")
    args = parser.parse_args(argv)

    try:
        sender, auth_code, recipient = load_config()
        body = read_body(args.body, args.body_file)
        send_mail(sender, auth_code, recipient, args.subject, body)
    except NotifyError as e:
        print(f"notify: ERROR: {e}", file=sys.stderr)
        return 1
    print(f"notify: sent to {recipient}: {args.subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
