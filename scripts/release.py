#!/usr/bin/env python3
"""Bump every version in the project, commit it, and tag the release.

    python scripts/release.py 0.4.0
    python scripts/release.py minor --dry-run

The tag it creates is what triggers .github/workflows/release.yml, so the
workflow — not a hand-made tag from a half-updated tree — is what ships a
release. Run it from a clean checkout of the branch you mean to release.

No third-party packages: this has to work on a fresh clone.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every place a version lives. AGENTS/READMEs are prose and stay untouched.
JSON_FILES = {
    "app/package.json": lambda d: d.__setitem__("version", None),
    "app/package-lock.json": None,  # two fields, handled below
    "app/src-tauri/tauri.conf.json": lambda d: d.__setitem__("version", None),
}
CARGO_TOML = "app/src-tauri/Cargo.toml"
CARGO_LOCK = "app/src-tauri/Cargo.lock"
PYPROJECT = "sidecar/pyproject.toml"
PY_INIT = "sidecar/hadith_sidecar/__init__.py"

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class ReleaseError(RuntimeError):
    pass


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    """Write back with the file's original line endings, so diffs stay small."""
    target = ROOT / path
    raw = target.read_bytes()
    if b"\r\n" in raw:
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    target.write_bytes(text.encode("utf-8"))


def replace_once(path: str, pattern: str, replacement: str, label: str) -> None:
    text = read(path)
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ReleaseError(f"{path}: could not find {label}")
    write(path, new)


def bump_json(path: str, version: str) -> None:
    data = json.loads(read(path))
    data["version"] = version
    write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def bump_lock(version: str) -> None:
    data = json.loads(read("app/package-lock.json"))
    data["version"] = version
    root = data.get("packages", {}).get("")
    if isinstance(root, dict):
        root["version"] = version
    write("app/package-lock.json", json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def bump_cargo_lock(version: str) -> None:
    text = read(CARGO_LOCK)
    pattern = r'(\[\[package\]\]\nname = "hadith-finder"\nversion = ")[^"]+(")'
    new, count = re.subn(pattern, rf"\g<1>{version}\g<2>", text)
    if count != 1:
        raise ReleaseError(f"{CARGO_LOCK}: could not find the hadith-finder package")
    write(CARGO_LOCK, new)


def current_version() -> str:
    for path in ("app/src-tauri/tauri.conf.json", "app/package.json"):
        data = json.loads(read(path))
        value = data.get("version")
        if isinstance(value, str):
            return value
    raise ReleaseError("no version found in tauri.conf.json or package.json")


def next_version(spec: str) -> str:
    spec = spec.lstrip("v")
    match = SEMVER.match(spec)
    if match:
        return spec
    if spec in {"major", "minor", "patch"}:
        major, minor, patch = (int(n) for n in SEMVER.match(current_version()).groups())
        if spec == "major":
            return f"{major + 1}.0.0"
        if spec == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"
    raise ReleaseError(f"expected X.Y.Z, major, minor or patch, got {spec!r}")


def ensure_clean(branch: str) -> None:
    dirty = git("status", "--porcelain")
    if dirty:
        raise ReleaseError("working tree is not clean; commit or stash first")
    current = git("rev-parse", "--abbrev-ref", "HEAD")
    if current != branch:
        raise ReleaseError(f"on branch {current!r}, expected {branch!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="X.Y.Z, or major / minor / patch")
    parser.add_argument("--branch", default="main", help="branch to release (default: main)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    parser.add_argument("--no-push", action="store_true", help="commit and tag, but do not push")
    parser.add_argument("--allow-dirty", action="store_true", help="skip the clean-tree check")
    args = parser.parse_args()

    try:
        # A dry run only reads, so it may preview from a dirty tree.
        if not args.allow_dirty and not args.dry_run:
            ensure_clean(args.branch)
        old = current_version()
        new = next_version(args.version)
        if new == old:
            raise ReleaseError(f"version is already {old}")
        tag = f"v{new}"
        if tag in git("tag", "-l", tag).split():
            raise ReleaseError(f"tag {tag} already exists")

        changes = [
            "app/package.json",
            "app/package-lock.json",
            "app/src-tauri/tauri.conf.json",
            CARGO_TOML,
            CARGO_LOCK,
            PYPROJECT,
            PY_INIT,
        ]
        print(f"{old} -> {new}  (tag {tag})")
        for path in changes:
            print(f"  bump {path}")
        if args.dry_run:
            print("dry run: nothing written")
            return 0

        bump_json("app/package.json", new)
        bump_lock(new)
        bump_json("app/src-tauri/tauri.conf.json", new)
        replace_once(CARGO_TOML, r'^version = "[^"]+"', f'version = "{new}"', "[package] version")
        bump_cargo_lock(new)
        replace_once(PYPROJECT, r'^version = "[^"]+"', f'version = "{new}"', "project version")
        replace_once(PY_INIT, r'^__version__ = "[^"]+"', f'__version__ = "{new}"', "__version__")

        mismatched = [
            path
            for path in ("app/package.json", "app/src-tauri/tauri.conf.json")
            if json.loads(read(path)).get("version") != new
        ]
        if mismatched or f'version = "{new}"' not in read(CARGO_TOML):
            raise ReleaseError(f"version did not land everywhere: {mismatched or CARGO_TOML}")

        git("add", *changes)
        git("commit", "-m", f"Release {new}")
        # Annotated on purpose: `--follow-tags` skips lightweight tags, which is
        # how v0.4.0 got committed and "pushed" without its tag ever shipping.
        git("tag", "-a", tag, "-m", f"Release {new}")
        print(f"committed and tagged {tag}")

        if args.no_push:
            print(f"not pushed; run: git push origin {args.branch} {tag}")
            return 0
        git("push", "origin", args.branch, tag)
        if not git("ls-remote", "--tags", "origin", tag):
            raise ReleaseError(f"pushed {args.branch}, but {tag} never reached the remote")
        print(f"pushed {args.branch} and {tag}; the release workflow is building it")
        return 0
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
