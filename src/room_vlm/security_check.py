"""Security audit: ensure sensitive paths and secrets are not tracked by git."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Paths / patterns that must never be committed.
DENY_PATH_SUBSTRINGS = (
    "data/raw/",
    "data/extracted/",
    "data/processed/",
    "data/metadata/",
    "outputs/checkpoints/",
    "/adapter/",
    "adapter_model",
)

DENY_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".pem",
    ".key",
)

DENY_BASENAMES = {
    ".env",
    "credentials.json",
    "hf_token",
    "id_rsa",
}

# Content patterns that look like secrets.
SECRET_CONTENT_PATTERNS = (
    re.compile(r"HF_TOKEN\s*=\s*hf_", re.IGNORECASE),
    re.compile(r"HUGGING_FACE_HUB_TOKEN\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"WANDB_API_KEY\s*=\s*[a-zA-Z0-9_-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"BEGIN (RSA |OPENSSH )?PRIVATE KEY"),
)

NYU_PPM_RE = re.compile(r"(^|/)r-\d+\.\d+-\d+\.ppm$", re.IGNORECASE)


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # Not a git repo / empty — treat as no tracked files.
        return []
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _is_denied_path(path: str) -> str | None:
    name = Path(path).name
    # Synthetic fixtures are allowed (not real NYU Depth V2).
    if path.replace("\\", "/").startswith("tests/fixtures/"):
        return None
    if name in DENY_BASENAMES:
        return f"denied basename: {name}"
    if name.startswith(".env") and name != ".env.example":
        return f"denied env file: {name}"
    lower = path.lower()
    for sub in DENY_PATH_SUBSTRINGS:
        if sub in lower:
            # Allow .gitkeep placeholders only under data/
            if name == ".gitkeep":
                continue
            return f"denied path substring: {sub}"
    for ext in DENY_EXTENSIONS:
        if lower.endswith(ext):
            # Allow nothing with weight-like extensions in the repo.
            return f"denied extension: {ext}"
    if NYU_PPM_RE.search(path):
        return "looks like NYU raw RGB dump (r-*.ppm)"
    if "outputs/" in lower and name not in {".gitkeep"}:
        # outputs/** is ignored except .gitkeep; if tracked, fail.
        return "tracked file under outputs/ (only .gitkeep allowed)"
    return None


def _scan_content(repo_root: Path, rel_path: str) -> str | None:
    path = repo_root / rel_path
    if not path.is_file():
        return None
    # Skip binary / large files.
    try:
        if path.stat().st_size > 1_000_000:
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for pattern in SECRET_CONTENT_PATTERNS:
        if pattern.search(text):
            # Allow empty placeholders like HF_TOKEN= in .env.example
            if rel_path.endswith(".env.example"):
                # Still fail if a real-looking token value is present.
                if re.search(r"HF_TOKEN\s*=\s*hf_\S+", text, re.IGNORECASE):
                    return f"secret-like content matching {pattern.pattern}"
                continue
            return f"secret-like content matching {pattern.pattern}"
    return None


def audit(repo_root: Path | None = None) -> list[str]:
    """Return a list of violation messages (empty if clean)."""
    root = Path(repo_root or Path.cwd()).resolve()
    violations: list[str] = []
    for rel in _git_ls_files(root):
        reason = _is_denied_path(rel)
        if reason:
            violations.append(f"{rel}: {reason}")
            continue
        content_reason = _scan_content(root, rel)
        if content_reason:
            violations.append(f"{rel}: {content_reason}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit git-tracked files for secrets and data leaks.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: current working directory).",
    )
    args = parser.parse_args(argv)
    violations = audit(args.root)
    if violations:
        print("SECURITY CHECK FAILED. The following tracked files are not allowed:", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nRemove them from git (and keep them gitignored). "
            "Never commit NYU data, .env secrets, or model weights.",
            file=sys.stderr,
        )
        return 1
    print("Security check passed: no denied paths or secret-like content in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
