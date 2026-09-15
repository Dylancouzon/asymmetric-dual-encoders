"""Exercise preservation of the E host's old runtime in scratch directories."""
import ast
from pathlib import Path
import subprocess
import sys


def runtime_code():
    source=Path(__file__).resolve().parents[1]/'scripts/m13_build_migrated.py'
    tree=ast.parse(source.read_text())
    return next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='runtime_link' for t in n.targets))


def test_old_runtime_is_preserved_and_linked(tmp_path):
    (tmp_path/'work').mkdir();old=tmp_path/'.venv';old.mkdir();(old/'pyvenv.cfg').write_text('synthetic');(old/'retain.txt').write_text('preserve')
    r=subprocess.run([sys.executable,'-c',runtime_code()],cwd=tmp_path,capture_output=True)
    assert r.returncode==0,r.stderr
    assert (tmp_path/'work/m13-migrated-original-runtime/retain.txt').read_text()=='preserve'
    assert (tmp_path/'.venv').readlink()==Path('/opt/m13-runtime/venv')


def test_unknown_existing_directory_refuses(tmp_path):
    (tmp_path/'work').mkdir();(tmp_path/'.venv').mkdir();(tmp_path/'.venv/retain.txt').write_text('preserve')
    r=subprocess.run([sys.executable,'-c',runtime_code()],cwd=tmp_path,capture_output=True)
    assert r.returncode!=0
    assert (tmp_path/'.venv/retain.txt').read_text()=='preserve'


def test_wrong_runtime_symlink_refuses(tmp_path):
    (tmp_path/'.venv').symlink_to('/unrelated/runtime')
    r=subprocess.run([sys.executable,'-c',runtime_code()],cwd=tmp_path,capture_output=True)
    assert r.returncode!=0
    assert (tmp_path/'.venv').readlink()==Path('/unrelated/runtime')
