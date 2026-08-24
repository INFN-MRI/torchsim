"""The three-pool operator table, held to the arm that forms it per event.

Triton's CPU interpreter runs the kernels in Python over host tensors, so
these reach the plumbing without a GPU and without invalidating a compile
cache: the argument alignment of the launchers, the table's row indexing, and
the washout and recoveries the reading event applies.

They are slow -- a minute each, since the interpreter walks every element in
Python -- so they are opt-in: ``pytest -m interpreted``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent.parent
ROOT = TESTS.parent
TIMEOUT_S = 900


def _run(case: str) -> subprocess.CompletedProcess[str]:
    """One case, in a process that sets the interpreter flag before Triton."""
    environment = dict(os.environ)
    # Triton reads this at import, so it cannot be set from inside a test.
    environment["TRITON_INTERPRET"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(TESTS)]
    )
    return subprocess.run(
        [sys.executable, "-m", "utils.interpreted", case],
        capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT,
        env=environment,
    )


@pytest.mark.interpreted
@pytest.mark.parametrize("case", ["narrow", "wide", "chunked"])
def test_the_table_gives_what_forming_it_per_event_gives(case: str) -> None:
    """Both arms run the same launch; only where the operator comes from differs.

    ``narrow`` forces a table onto a train the launch-wide gate calls narrow,
    so every row takes the series and the two arms agree to the bit. ``wide``
    is the case that ships -- an inversion makes the launch decline the gate,
    and the table carries series rows beside a roots row. ``chunked`` is the
    same launch cut into chunks, which is what tells a chunk-local index for
    the cotangent table from a global one.
    """
    finished = _run(case)
    assert finished.returncode == 0, (
        f"{case} case failed:\n{finished.stdout}\n{finished.stderr}"
    )
    assert "forward" in finished.stdout, finished.stdout
