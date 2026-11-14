"""The command line parses, and every documented command exists (the tests import the CLI)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_cli_module_compiles_and_shows_help():
    out = subprocess.run([sys.executable, "-m", "quant", "--help"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize("cmd", ["fetch", "backtest", "sweep", "scan", "levels", "judas", "journal", "metalabel"])
def test_every_command_has_help(cmd):
    out = subprocess.run([sys.executable, "-m", "quant", cmd, "--help"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
