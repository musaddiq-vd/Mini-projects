"""The setup program: install, update and uninstall (a fake package, a test registry key, test shortcut folders)."""

import importlib.util
import platform
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Windows", reason="Windows installer")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_SETUP_REGKEY", r"Software\JarvisSetupTest\Uninstall")
    monkeypatch.setenv("JARVIS_SETUP_RUNVALUE", "JarvisSetupTest")
    monkeypatch.setenv("JARVIS_SETUP_SHORTCUTS", str(tmp_path / "links"))
    spec = importlib.util.spec_from_file_location("setup_app_under_test", ROOT / "installer" / "setup_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    mod.registry_delete(r"Software\JarvisSetupTest\Uninstall")
    mod.registry_delete(r"Software\JarvisSetupTest")
    mod.registry_delete(mod.RUN_KEY, "JarvisSetupTest")


def fake_package(root: Path, marker: str = "v1") -> Path:
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "pythonw.exe").write_text("not really python", encoding="utf-8")
    app = root / "app"
    (app / "jarvis").mkdir(parents=True)
    (app / "jarvis" / "__init__.py").write_text(f'__version__ = "9.9.{marker[-1]}"\n', encoding="utf-8")
    (app / "jarvis_launcher.pyw").write_text("# launcher", encoding="utf-8")
    (app / "assets").mkdir()
    (app / "assets" / "jarvis.ico").write_bytes(b"\x00\x00\x01\x00")
    (app / "installer").mkdir()
    (app / "installer" / "setup_app.py").write_text("# setup", encoding="utf-8")
    (app / "marker.txt").write_text(marker, encoding="utf-8")
    return root


def test_install_update_uninstall(setup, tmp_path):
    pkg = fake_package(tmp_path / "pkg")
    target = tmp_path / "Programs" / "Jarvis"
    steps = []
    setup.install(target, autostart=True, progress=lambda pct, text: steps.append(pct), source=pkg)
    assert (target / "app" / "marker.txt").read_text(encoding="utf-8") == "v1"
    assert steps[-1] == 100 and steps == sorted(steps)
    assert (tmp_path / "links" / "start" / "Jarvis.lnk").exists() and (tmp_path / "links" / "desktop" / "Jarvis.lnk").exists()
    key = r"Software\JarvisSetupTest\Uninstall"
    assert setup.registry_get(key, "InstallLocation") == str(target.resolve())
    assert "--uninstall" in setup.registry_get(key, "UninstallString")
    assert "jarvis_launcher.pyw" in setup.registry_get(setup.RUN_KEY, "JarvisSetupTest")
    assert setup.installed_dir() == target.resolve()

    pkg2 = fake_package(tmp_path / "pkg2", "v2")  # an update: the folder is swapped, nothing old is left
    setup.install(target, desktop_icon=False, source=pkg2)
    assert (target / "app" / "marker.txt").read_text(encoding="utf-8") == "v2"
    assert setup.registry_get(setup.RUN_KEY, "JarvisSetupTest") is None  # unticked on update: removed
    assert not (tmp_path / "links" / "desktop" / "Jarvis.lnk").exists()
    assert sorted(p.name for p in target.parent.iterdir()) == ["Jarvis"]

    home = tmp_path / "home"
    home.mkdir()
    (home / "memory.json").write_text("{}", encoding="utf-8")
    setup.uninstall(target)
    assert not target.exists() and not (tmp_path / "links" / "start" / "Jarvis.lnk").exists()
    assert setup.registry_get(key, "InstallLocation") is None
    assert setup.registry_get(setup.RUN_KEY, "JarvisSetupTest") is None
    assert (home / "memory.json").exists()  # data is kept unless asked


def test_refusals(setup, tmp_path):
    pkg = fake_package(tmp_path / "pkg")
    with pytest.raises(RuntimeError, match="setup files are"):
        setup.install(pkg / "inside", source=pkg)
    broken = tmp_path / "broken"
    (broken / "app").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        setup.install(tmp_path / "x", source=broken)
    someone_else = tmp_path / "Documents"
    someone_else.mkdir()
    (someone_else / "thesis.docx").write_text("important", encoding="utf-8")
    with pytest.raises(RuntimeError, match="other files"):
        setup.install(someone_else, source=pkg)  # never replaces a folder that isn't a Jarvis install
    assert (someone_else / "thesis.docx").exists()
    with pytest.raises(RuntimeError, match="isn't installed"):
        setup.uninstall(someone_else)
    assert (someone_else / "thesis.docx").exists()


def test_delete_data_on_request(setup, tmp_path, monkeypatch):
    pkg = fake_package(tmp_path / "pkg")
    target = tmp_path / "J"
    setup.install(target, desktop_icon=False, menu_icon=False, source=pkg)
    home = tmp_path / "jhome"
    home.mkdir()
    monkeypatch.setenv("JARVIS_HOME", str(home))
    setup.uninstall(target, delete_data=True)
    assert not home.exists() and not target.exists()


def test_package_contents_are_listed():
    sys.path.insert(0, str(ROOT / "installer"))
    import build

    assert build.skip("pytest") and build.skip("scipy.libs") and build.skip("PyInstaller")
    assert not build.skip("PyQt6") and not build.skip("numpy") and not build.skip("tinytuya")
    assert {"Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"} <= build.QT_KEEP_DLL
