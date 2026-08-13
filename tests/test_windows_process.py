from __future__ import annotations

import os
import subprocess
import sys

import pytest

from latentslate_engine.runtime.windows_process import DisposableProcessTree


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_disposable_process_tree_reports_zero_active_processes_after_clean_exit():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.05)"])
    tree = DisposableProcessTree(process)
    try:
        assert tree.active_processes() >= 1
        assert process.wait(timeout=10) == 0
        tree.wait_for_empty(timeout=10)
        assert tree.active_processes() == 0
    finally:
        tree.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_disposable_process_tree_termination_waits_for_the_job_to_empty():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    tree = DisposableProcessTree(process)
    try:
        tree.terminate()
        process.wait(timeout=10)
        tree.wait_for_empty(timeout=10)
        assert tree.active_processes() == 0
    finally:
        tree.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_disposable_process_tree_terminates_descendant_after_root_exits():
    code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(0.1)"
    )
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    tree = DisposableProcessTree(process)
    try:
        assert process.stdout is not None
        assert int(process.stdout.readline().strip()) > 0
        assert process.wait(timeout=10) == 0
        assert tree.active_processes() >= 1
        tree.terminate()
        tree.wait_for_empty(timeout=10)
        assert tree.active_processes() == 0
    finally:
        tree.close()
