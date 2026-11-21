"""Вартість запуску CLI: важкі залежності вантажаться лише там, де вони справді потрібні."""

import subprocess
import sys


def test_importing_cli_does_not_load_heavy_modules() -> None:
    """NumPy потрібен лише пошуку й фразам, pydantic_settings лише команді serve: решта команд
    (--help, index, stats) не мають платити за них на старті (≈ 90 і ≈ 100 мс)."""
    code = (
        "import sys, findex.cli; print(sorted(m for m in ('numpy', 'pydantic_settings') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
