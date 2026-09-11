from __future__ import annotations

import unittest
from unittest.mock import patch

import lnl_toolbox.cli.ports as ports_module
from lnl_toolbox.cli.ports import cleanup_ports, find_lnl_listeners


class CliPortsTests(unittest.TestCase):
    def test_find_lnl_listeners_requires_an_lnl_commandline(self) -> None:
        raw = [(101, 8765, "127.0.0.1:8765"), (202, 9000, "127.0.0.1:9000")]
        commands = {101: "python web/command_console.py --port 8765", 202: "python unrelated.py"}
        with patch("lnl_toolbox.cli.ports._windows_listeners", return_value=raw), patch(
            "lnl_toolbox.cli.ports._posix_listeners", return_value=raw
        ), patch("lnl_toolbox.cli.ports._process_commandlines", return_value=commands):
            listeners = find_lnl_listeners()
        self.assertEqual([(item.pid, item.port) for item in listeners], [(101, 8765)])


    def test_cleanup_ports_dry_run_does_not_terminate(self) -> None:
        listener = (101, 8765, "127.0.0.1:8765")
        with patch("lnl_toolbox.cli.ports._windows_listeners", return_value=[listener]), patch(
            "lnl_toolbox.cli.ports._posix_listeners", return_value=[listener]
        ), patch(
            "lnl_toolbox.cli.ports._process_commandlines",
            return_value={101: "python web/command_console.py --port 8765"},
        ), patch("lnl_toolbox.cli.ports._terminate") as terminate:
            self.assertEqual(cleanup_ports(dry_run=True), 0)
        terminate.assert_not_called()

    def test_find_lnl_listeners_falls_back_to_python_on_default_port(self) -> None:
        raw = [(101, 8765, "127.0.0.1:8765")]
        with patch.object(ports_module.os, "name", "nt"), patch(
            "lnl_toolbox.cli.ports._windows_listeners", return_value=raw
        ), patch("lnl_toolbox.cli.ports._process_commandlines", return_value={}), patch(
            "lnl_toolbox.cli.ports._windows_process_images",
            return_value={101: r"F:\\Miniconda\\envs\\pytorch\\python.exe"},
        ):
            listeners = find_lnl_listeners()
        self.assertEqual([(item.pid, item.port) for item in listeners], [(101, 8765)])
