"""Tests for init script collection and execution from multiple directories (.sh and .py)."""

import sys

from ministack.app import _collect_scripts, _run_init_scripts


def test_collect_scripts_single_dir(tmp_path):
    (tmp_path / "01-seed.sh").write_text("#!/bin/sh\necho seed")
    (tmp_path / "02-setup.sh").write_text("#!/bin/sh\necho setup")
    (tmp_path / "notes.txt").write_text("not a script")

    result = _collect_scripts(str(tmp_path))
    assert len(result) == 2
    assert result[0].endswith("01-seed.sh")
    assert result[1].endswith("02-setup.sh")


def test_collect_scripts_multiple_dirs(tmp_path):
    dir1 = tmp_path / "native"
    dir2 = tmp_path / "compat"
    dir1.mkdir()
    dir2.mkdir()

    (dir1 / "01-seed.sh").write_text("#!/bin/sh\necho native")
    (dir2 / "02-extra.sh").write_text("#!/bin/sh\necho compat")

    result = _collect_scripts(str(dir1), str(dir2))
    assert len(result) == 2
    assert result[0].endswith("01-seed.sh")
    assert result[1].endswith("02-extra.sh")


def test_collect_scripts_dedup_first_dir_wins(tmp_path):
    dir1 = tmp_path / "native"
    dir2 = tmp_path / "compat"
    dir1.mkdir()
    dir2.mkdir()

    (dir1 / "01-seed.sh").write_text("#!/bin/sh\necho native")
    (dir2 / "01-seed.sh").write_text("#!/bin/sh\necho compat")

    result = _collect_scripts(str(dir1), str(dir2))
    assert len(result) == 1
    assert str(dir1) in result[0]  # native path wins


def test_collect_scripts_missing_dir(tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    (existing / "01-seed.sh").write_text("#!/bin/sh\necho hi")

    result = _collect_scripts("/nonexistent/path", str(existing))
    assert len(result) == 1
    assert result[0].endswith("01-seed.sh")


def test_collect_scripts_empty_dirs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _collect_scripts(str(empty))
    assert result == []


def test_collect_scripts_no_dirs():
    result = _collect_scripts("/nonexistent/a", "/nonexistent/b")
    assert result == []


def test_collect_scripts_alphabetical_order(tmp_path):
    (tmp_path / "03-last.sh").write_text("")
    (tmp_path / "01-first.sh").write_text("")
    (tmp_path / "02-middle.sh").write_text("")

    result = _collect_scripts(str(tmp_path))
    names = [r.split("/")[-1] for r in result]
    assert names == ["01-first.sh", "02-middle.sh", "03-last.sh"]


def test_collect_scripts_py_files(tmp_path):
    (tmp_path / "01-seed.sh").write_text("#!/bin/sh\necho seed")
    (tmp_path / "02-migrate.py").write_text("print('migrate')")

    result = _collect_scripts(str(tmp_path))
    assert len(result) == 2
    assert result[0].endswith("01-seed.sh")
    assert result[1].endswith("02-migrate.py")


def test_collect_scripts_mixed_sort_order(tmp_path):
    (tmp_path / "03-cleanup.sh").write_text("")
    (tmp_path / "01-setup.sh").write_text("")
    (tmp_path / "02-migrate.py").write_text("")

    result = _collect_scripts(str(tmp_path))
    names = [r.split("/")[-1] for r in result]
    assert names == ["01-setup.sh", "02-migrate.py", "03-cleanup.sh"]


def test_collect_scripts_ignores_non_script_files(tmp_path):
    (tmp_path / "01-seed.sh").write_text("")
    (tmp_path / "02-migrate.py").write_text("")
    (tmp_path / "readme.md").write_text("")
    (tmp_path / "config.json").write_text("")
    (tmp_path / "notes.txt").write_text("")

    result = _collect_scripts(str(tmp_path))
    assert len(result) == 2
    names = [r.split("/")[-1] for r in result]
    assert names == ["01-seed.sh", "02-migrate.py"]


def test_init_scripts_uses_correct_interpreter(tmp_path, monkeypatch):
    sh_script = tmp_path / "01-setup.sh"
    py_script = tmp_path / "02-migrate.py"
    sh_script.write_text("#!/bin/sh\necho hi")
    py_script.write_text("print('hi')")

    monkeypatch.setattr(
        'ministack.app._collect_scripts',
        lambda *a: [str(sh_script), str(py_script)]
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    monkeypatch.setattr('subprocess.run', fake_run)

    _run_init_scripts()

    assert calls[0][0] == "sh"
    assert calls[0][1] == str(sh_script)
    assert calls[1][0] == sys.executable
    assert calls[1][1] == str(py_script)
