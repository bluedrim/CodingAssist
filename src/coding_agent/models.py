from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Task:
    goal: str
    workspace: Path
    max_reviews: int = 100
    apply_changes: bool = False
    exact_reviews: bool = False


@dataclass(frozen=True)
class PlanStep:
    title: str
    detail: str
    done: bool = False


@dataclass(frozen=True)
class PromptRevision:
    original: str
    improved: str
    changes: tuple[str, ...]
    required_sections: tuple[str, ...] = ("Objective", "Context", "Constraints", "Verification", "Done")

    @property
    def changed(self) -> bool:
        return self.original.strip() != self.improved.strip()

    @property
    def section_names(self) -> tuple[str, ...]:
        sections: list[str] = []
        for line in self.improved.splitlines():
            if ":" in line and not line.startswith(" "):
                name = line.split(":", 1)[0].strip()
                if name:
                    sections.append(name)
        return tuple(sections)

    @property
    def missing_sections(self) -> tuple[str, ...]:
        present = {section.lower() for section in self.section_names}
        return tuple(section for section in self.required_sections if section.lower() not in present)

    @property
    def quality_score(self) -> float:
        if not self.required_sections:
            return 1.0
        present = len(self.required_sections) - len(self.missing_sections)
        return present / len(self.required_sections)


@dataclass(frozen=True)
class FilePatch:
    path: Path
    content: str
    reason: str

    @property
    def is_python(self) -> bool:
        return self.path.suffix == ".py"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    message: str
    path: Path | None = None
    line: int | None = None

    @property
    def location(self) -> str:
        if self.path is None:
            return ""
        if self.line is None:
            return self.path.as_posix()
        return f"{self.path.as_posix()}:{self.line}"


@dataclass
class ReviewReport:
    iteration: int
    focus: str = "patch"
    findings: list[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == Severity.ERROR for f in self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == Severity.WARNING)

    @property
    def has_warnings(self) -> bool:
        return self.warning_count > 0


@dataclass(frozen=True)
class ReviewSummary:
    total_reviews: int
    error_count: int
    warning_count: int
    focus_counts: dict[str, int]

    @classmethod
    def from_reports(cls, reports: list[ReviewReport]) -> ReviewSummary:
        focus_counts: dict[str, int] = {}
        for report in reports:
            focus_counts[report.focus] = focus_counts.get(report.focus, 0) + 1
        return cls(
            total_reviews=len(reports),
            error_count=sum(report.error_count for report in reports),
            warning_count=sum(report.warning_count for report in reports),
            focus_counts=focus_counts,
        )

    @property
    def passed(self) -> bool:
        return self.error_count == 0


@dataclass(frozen=True)
class TestResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    @property
    def command_text(self) -> str:
        return " ".join(self.command)


@dataclass
class AgentRunResult:
    goal: str
    prompt_revision: PromptRevision
    plan: list[PlanStep]
    patches: list[FilePatch]
    reviews: list[ReviewReport]
    tests: list[TestResult]
    applied: bool
    prompt_review: ReviewReport | None = None

    @property
    def passed(self) -> bool:
        prompt_review_passed = self.prompt_review.passed if self.prompt_review else True
        final_review_passed = self.reviews[-1].passed if self.reviews else True
        if self.applied:
            return prompt_review_passed and final_review_passed and self.verified
        return prompt_review_passed and final_review_passed

    @property
    def verified(self) -> bool:
        return self.applied and bool(self.tests) and all(test.passed for test in self.tests)

    @property
    def verification_status(self) -> str:
        if not self.tests:
            return "not_run"
        if self.verified:
            return "verified"
        if self.failed_tests:
            return "failed"
        return "unverified"

    @property
    def failed_tests(self) -> list[TestResult]:
        return [test for test in self.tests if not test.passed]

    @property
    def review_pass_count(self) -> int:
        return len(self.reviews)
