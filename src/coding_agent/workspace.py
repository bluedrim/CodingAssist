from __future__ import annotations

from pathlib import Path

from coding_agent.models import FilePatch


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: Path) -> Path:
        target = (self.root / path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return target

    def list_files(self) -> list[Path]:
        ignored = {
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            "build",
            "dist",
            "htmlcov",
        }
        files: list[Path] = []
        for path in self.root.rglob("*"):
            if any(part in ignored for part in path.parts):
                continue
            if any(part.endswith(".egg-info") for part in path.parts):
                continue
            if path.is_file():
                files.append(path.relative_to(self.root))
        return sorted(files)

    def read_text(self, path: Path) -> str:
        return self.resolve(path).read_text(encoding="utf-8")

    def exists(self, path: Path) -> bool:
        return self.resolve(path).exists()

    def apply_patch(self, patch: FilePatch) -> None:
        target = self.resolve(patch.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch.content, encoding="utf-8")
