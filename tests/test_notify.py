"""Unit tests for the Phase 2 notifier (experiments/notify.py).

All SMTP interactions are mocked; no real mail is ever sent and no real
credentials are used.
"""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

NOTIFY_PATH = Path(__file__).resolve().parent.parent / "experiments" / "notify.py"
spec = importlib.util.spec_from_file_location("notify", NOTIFY_PATH)
notify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notify)

ENV = {
    "QQ_SMTP_EMAIL": "sender@qq.com",
    "QQ_SMTP_AUTH_CODE": "secret-auth-code-123",
    "RESEARCH_NOTIFY_EMAIL": "receiver@example.com",
}


class LoadConfigTest(unittest.TestCase):
    def test_reads_all_three_env_vars(self):
        sender, auth, recipient = notify.load_config(dict(ENV))
        self.assertEqual(sender, "sender@qq.com")
        self.assertEqual(auth, "secret-auth-code-123")
        self.assertEqual(recipient, "receiver@example.com")

    def test_missing_each_env_var_fails(self):
        for key in ENV:
            env = {k: v for k, v in ENV.items() if k != key}
            with self.assertRaises(notify.NotifyError) as cm:
                notify.load_config(env)
            self.assertIn(key, str(cm.exception))

    def test_empty_env_var_counts_as_missing(self):
        env = dict(ENV, QQ_SMTP_AUTH_CODE="")
        with self.assertRaises(notify.NotifyError):
            notify.load_config(env)

    def test_main_fails_without_env_vars(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = notify.main(["--subject", "s", "--body", "b"])
        self.assertEqual(rc, 1)
        self.assertIn("QQ_SMTP", err.getvalue())


class ReadBodyTest(unittest.TestCase):
    def test_body_file_is_read_as_utf8(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("实验 001 完成：IC 0.0352\n")
            path = f.name
        self.assertEqual(notify.read_body(body_file=path), "实验 001 完成：IC 0.0352\n")

    def test_missing_body_file_fails(self):
        with self.assertRaises(notify.NotifyError) as cm:
            notify.read_body(body_file="/tmp/definitely-not-there-xyz.txt")
        self.assertIn("body file not found", str(cm.exception))

    def test_main_fails_when_body_file_missing(self):
        with mock.patch.dict("os.environ", dict(ENV), clear=True):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = notify.main(
                    ["--subject", "s", "--body-file", "/tmp/definitely-not-there-xyz.txt"]
                )
        self.assertEqual(rc, 1)
        self.assertIn("body file not found", err.getvalue())

    def test_body_fallback_and_no_body_error(self):
        self.assertEqual(notify.read_body(body="inline"), "inline")
        with self.assertRaises(notify.NotifyError):
            notify.read_body()


class SendMailTest(unittest.TestCase):
    def _run_main(self, smtp_mock, extra_args=("--body", "正文 body")):
        with mock.patch.dict("os.environ", dict(ENV), clear=True), \
                mock.patch.object(notify.smtplib, "SMTP_SSL", smtp_mock):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = notify.main(["--subject", "[HVQ] 完成 — 测试", *extra_args])
        return rc, out.getvalue(), err.getvalue()

    def test_smtp_success_path(self):
        server = mock.MagicMock()
        smtp = mock.MagicMock(return_value=server)
        rc, out, _ = self._run_main(smtp)

        self.assertEqual(rc, 0)
        smtp.assert_called_once_with(
            notify.SMTP_HOST, notify.SMTP_PORT, timeout=notify.SMTP_TIMEOUT
        )
        ctx = server.__enter__.return_value
        ctx.login.assert_called_once_with("sender@qq.com", "secret-auth-code-123")
        self.assertEqual(ctx.send_message.call_count, 1)
        msg = ctx.send_message.call_args[0][0]
        self.assertEqual(msg["From"], "sender@qq.com")
        self.assertEqual(msg["To"], "receiver@example.com")
        self.assertEqual(msg["Subject"], "[HVQ] 完成 — 测试")
        self.assertIn("正文 body", msg.get_content())
        self.assertIn("sent to receiver@example.com", out)

    def test_smtp_failure_returns_nonzero(self):
        smtp = mock.MagicMock(side_effect=OSError("connection refused"))
        rc, _, err = self._run_main(smtp)
        self.assertEqual(rc, 1)
        self.assertIn("SMTP send failed", err)

    def test_auth_code_never_leaks_into_error(self):
        # Even if the SMTP layer echoes the credential in its exception,
        # the reported error must scrub it.
        smtp = mock.MagicMock(
            side_effect=Exception("login rejected for secret-auth-code-123")
        )
        rc, out, err = self._run_main(smtp)
        self.assertEqual(rc, 1)
        self.assertNotIn("secret-auth-code-123", err)
        self.assertNotIn("secret-auth-code-123", out)
        self.assertIn("***", err)


if __name__ == "__main__":
    unittest.main()
