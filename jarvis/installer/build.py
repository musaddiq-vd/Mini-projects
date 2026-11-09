"""Build the Windows package:  .venv\\Scripts\\python installer\\build.py

Makes installer\\dist\\Jarvis-<version>-windows.zip with "Install Jarvis.cmd" inside. Nothing in it is a new,
unsigned .exe: Jarvis runs on the official, signed Python runtime from python.org, so Windows Smart App Control
(on by default on new Windows 11 PCs) lets it run without a code-signing certificate.

Steps: download the Python embeddable runtime (same version as this venv, signature checked) -> copy the
libraries Jarvis needs (dev tools and unused Qt parts left out) -> copy Jarvis -> self-test the package ->
zip it.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
DIST = HERE / "dist"
SITE = Path(sys.prefix) / "Lib" / "site-packages"
sys.path.insert(0, str(ROOT))

# packages that are only for development, or that Jarvis doesn't use
SKIP_PACKAGES = [
    "pip", "pip-*", "setuptools", "setuptools-*", "pkg_resources", "_distutils_hack", "distutils-precedence.pth",
    "wheel", "wheel-*", "pytest", "pytest-*", "_pytest", "py.py", "pluggy", "pluggy-*", "iniconfig", "iniconfig-*",
    "ruff", "ruff-*", "PyInstaller", "pyinstaller-*", "_pyinstaller_hooks_contrib", "pyinstaller_hooks_contrib-*",
    "altgraph", "altgraph-*", "pefile*", "ordlookup", "peutils.py", "pywin32_ctypes*", "win32ctypes",
    "scipy", "scipy-*", "scipy.libs", "sklearn", "scikit_learn-*", "joblib", "joblib-*", "threadpoolctl*",
    "pygments", "pygments-*", "__pycache__",
]
# Qt: only Core, Gui, Widgets, Network (+ Svg for icons) and the platform plugins are used
QT_KEEP_PYD = {"QtCore.pyd", "QtGui.pyd", "QtWidgets.pyd", "QtNetwork.pyd", "QtSvg.pyd"}
QT_KEEP_DLL = {"Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "Qt6Network.dll", "Qt6Svg.dll"}
QT_KEEP_PLUGINS = {"platforms", "styles", "imageformats", "iconengines", "tls", "generic", "networkinformation"}
QT_DROP_DLL = {"avcodec-61.dll", "avformat-61.dll", "avutil-59.dll", "swresample-5.dll", "swscale-8.dll"}
APP_FILES = ["jarvis", "plugins", "jarvis_launcher.pyw", "README.md", "requirements.txt"]


def _link_or_copy(src, dst):
    """--hardlink (local checks only): reuse the same files, which Smart App Control already knows."""
    try:
        import os

        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


COPY = _link_or_copy if "--hardlink" in sys.argv else shutil.copy2


def log(msg: str) -> None:
    print(">", msg, flush=True)


def runtime_zip(version: str) -> Path:
    url = f"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip"
    dest = BUILD / "cache" / Path(url).name
    if not dest.exists():
        log(f"downloading {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def check_signature(exe: Path) -> None:
    """The runtime must carry the Python Software Foundation's valid signature."""
    env = {**__import__("os").environ, "JARVIS_SIG_FILE": str(exe)}  # passed as data, never as script text
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "$s=Get-AuthenticodeSignature -LiteralPath $env:JARVIS_SIG_FILE; "
                          "Write-Output ($s.Status.ToString() + '|' + $s.SignerCertificate.Subject)"],
                         capture_output=True, text=True, check=False, env=env).stdout.strip()
    status, _, subject = out.partition("|")
    if status != "Valid" or "Python Software Foundation" not in subject:
        raise SystemExit(f"{exe.name} is not signed by the Python Software Foundation ({out}) - stopping.")


def skip(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in SKIP_PACKAGES)


def copy_site_packages(dest: Path) -> None:
    log("copying libraries")
    for item in sorted(SITE.iterdir()):
        if skip(item.name):
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "test"),
                            copy_function=COPY)
        else:
            COPY(item, target)
    qt = dest / "PyQt6"
    for pyd in qt.glob("*.pyd"):
        if pyd.name not in QT_KEEP_PYD and not pyd.name.startswith("sip."):
            pyd.unlink()
    for folder in ("qml", "translations", "qsci"):
        shutil.rmtree(qt / "Qt6" / folder, ignore_errors=True)
    for dll in (qt / "Qt6" / "bin").glob("*.dll"):
        if (dll.name.startswith("Qt6") and dll.name not in QT_KEEP_DLL) or dll.name in QT_DROP_DLL:
            dll.unlink()
    for plugin in (qt / "Qt6" / "plugins").iterdir():
        if plugin.name not in QT_KEEP_PLUGINS:
            shutil.rmtree(plugin, ignore_errors=True)
    for folder in ("bindings", "lupdate", "uic"):
        shutil.rmtree(qt / folder, ignore_errors=True)


def make_runtime(dest: Path, version: str) -> None:
    log("unpacking the signed Python runtime")
    with zipfile.ZipFile(runtime_zip(version)) as z:
        z.extractall(dest)
    for exe in ("python.exe", "pythonw.exe"):
        check_signature(dest / exe)
    tag = "".join(version.split(".")[:2])  # 3.11.5 -> 311
    # where Python looks for code: its own library, the libraries we copy, and Jarvis itself
    (dest / f"python{tag}._pth").write_text(
        f"python{tag}.zip\n.\nLib\\site-packages\n..\\app\nimport site\n", encoding="utf-8")
    shutil.copy2(HERE / "openwakeword_stub.py", dest / "sitecustomize.py")
    site = dest / "Lib" / "site-packages"
    site.mkdir(parents=True)
    copy_site_packages(site)


def make_app(dest: Path) -> None:
    log("copying Jarvis")
    dest.mkdir(parents=True)
    for name in APP_FILES:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            shutil.copy2(src, dest / name)
    (dest / "installer").mkdir()
    for name in ("setup_app.py", "selftest.py"):
        shutil.copy2(HERE / name, dest / "installer" / name)
    shutil.copytree(HERE / "assets", dest / "assets")


def selftest(files: Path) -> dict:
    log("self-test of the package (tools, libraries, the main window)")
    report = BUILD / "selftest.json"
    report.unlink(missing_ok=True)
    env = {**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen", "JARVIS_HOME": str(BUILD / "selftest-home")}
    env.pop("PYTHONPATH", None)  # only the package's own code and libraries may be used
    subprocess.run([str(files / "runtime" / "python.exe"), str(files / "app" / "installer" / "selftest.py"),
                    str(report)], env=env, timeout=300, check=False)
    data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"ok": False, "errors": ["no report"]}
    print("  ", json.dumps(data))
    if not data.get("ok"):
        raise SystemExit("The package failed its self-test - not making the zip.")
    return data


def main() -> None:
    from jarvis import __version__ as version

    py_version = ".".join(map(str, sys.version_info[:3]))
    subprocess.run([sys.executable, str(HERE / "make_icon.py")], check=True, cwd=ROOT)
    pkg = BUILD / f"Jarvis-{version}"
    shutil.rmtree(pkg, ignore_errors=True)
    files = pkg / "files"
    make_runtime(files / "runtime", py_version)
    make_app(files / "app")
    selftest(files)
    shutil.rmtree(files / "app" / "assets" / "__pycache__", ignore_errors=True)
    for cache in list(files.rglob("__pycache__")):  # the self-test's compiled files are not shipped
        shutil.rmtree(cache, ignore_errors=True)
    (pkg / "Install Jarvis.cmd").write_text(
        '@echo off\r\nrem Opens the Jarvis setup window (Jarvis runs on the official signed Python runtime).\r\n'
        'start "" "%~dp0files\\runtime\\pythonw.exe" "%~dp0files\\app\\installer\\setup_app.py"\r\n',
        encoding="ascii")
    (pkg / "READ ME.txt").write_text(
        f"Jarvis {version}\r\n\r\n1. Extract this zip.\r\n2. Double-click \"Install Jarvis.cmd\".\r\n"
        "3. Choose your options and press Install.\r\n\r\nNo Python or admin rights needed. Your data stays in "
        "%USERPROFILE%\\.jarvis.\r\nRemove it any time from Settings > Apps > Installed apps > Jarvis.\r\n",
        encoding="utf-8")
    DIST.mkdir(exist_ok=True)
    out = DIST / f"Jarvis-{version}-windows.zip"
    log("zipping")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(pkg.parent).as_posix())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (DIST / f"{out.name}.sha256").write_text(f"{digest}  {out.name}\n", encoding="ascii")
    size = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
    print(f"\nPackage: {out}  ({out.stat().st_size / 2**20:.0f} MB zip, {size / 2**20:.0f} MB installed)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    if "--clean" in sys.argv:
        shutil.rmtree(BUILD, ignore_errors=True)
    main()
