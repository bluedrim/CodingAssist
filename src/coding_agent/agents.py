from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from coding_agent.backend import BackendError, ModelBackend
from coding_agent.models import FilePatch, Finding, PlanStep, PromptRevision, ReviewReport, Severity, Task, TestResult
from coding_agent.workspace import Workspace


class PromptImproverAgent:
    """Turns vague user goals into a stable implementation prompt."""

    max_goal_chars = 500
    required_sections = ("Objective", "Context", "Constraints", "Verification", "Done")

    def __init__(self, backend: ModelBackend | None = None) -> None:
        self.backend = backend

    def improve(self, goal: str, workspace: Workspace) -> PromptRevision:
        original = goal.strip()
        if self._is_structured(original):
            return PromptRevision(original=goal, improved=original, changes=())

        normalized = self._normalize(original)
        if self._is_structured(normalized):
            return PromptRevision(original=goal, improved=normalized, changes=())

        bounded, truncated = self._bound(normalized)
        objective = bounded or "Improve the coding agent project."

        fallback_prompt = self._build_prompt(objective, workspace)
        improved = self._complete_with_backend(fallback_prompt) if self.backend else fallback_prompt
        changes = [
            "Added objective/context/constraints/verification/done sections.",
            "Added workspace-aware project context.",
        ]
        if self.backend and improved != fallback_prompt:
            changes.append("Improved by model backend.")
        if self.backend and improved == fallback_prompt:
            changes.append("Used deterministic fallback prompt.")
        if normalized != original:
            changes.append("Normalized whitespace.")
        if truncated:
            changes.append(f"Trimmed goal to {self.max_goal_chars} characters.")
        return PromptRevision(
            original=goal,
            improved=improved,
            changes=tuple(changes),
        )

    def _is_structured(self, goal: str) -> bool:
        normalized = goal.lower()
        required = ("objective:", "verification:", "done:")
        return all(marker in normalized for marker in required)

    def _normalize(self, goal: str) -> str:
        return " ".join(goal.split())

    def _bound(self, goal: str) -> tuple[str, bool]:
        if len(goal) <= self.max_goal_chars:
            return goal, False
        return goal[: self.max_goal_chars].rstrip(), True

    def _build_prompt(self, objective: str, workspace: Workspace) -> str:
        files = workspace.list_files()
        package_hint = "existing Python package" if any(path.parts[:1] == ("src",) for path in files) else "new Python package"
        return "\n".join(
            [
                self._section("Objective", objective),
                self._section("Context", f"Work inside the {package_hint} at {workspace.root}."),
                self._section("Constraints", "Keep edits focused, preserve existing behavior, and avoid unsafe filesystem writes."),
                self._section("Verification", "Run syntax checks and available unit tests."),
                self._section("Done", "The requested behavior is implemented, reviewed, and verified."),
            ]
        )

    def _complete_with_backend(self, fallback_prompt: str) -> str:
        try:
            candidate = self.backend.complete(fallback_prompt) if self.backend else fallback_prompt
        except BackendError:
            return fallback_prompt
        revision = PromptRevision(original=fallback_prompt, improved=candidate, changes=())
        if revision.missing_sections:
            return fallback_prompt
        return candidate

    def _section(self, name: str, value: str) -> str:
        if name not in self.required_sections:
            raise ValueError(f"unknown prompt section: {name}")
        return f"{name}: {value}"


class PromptReviewAgent:
    """Reviews improved prompts before planning and coding."""

    risky_terms = ("rm -rf", "delete everything", "format disk", "sudo ")

    def review(self, revision: PromptRevision) -> ReviewReport:
        findings: list[Finding] = []
        for section in revision.missing_sections:
            findings.append(Finding(Severity.ERROR, f"Prompt is missing required section: {section}."))
        if revision.quality_score < 1.0:
            findings.append(Finding(Severity.ERROR, f"Prompt quality score is {revision.quality_score:.2f}; expected 1.00."))

        lowered = revision.improved.lower()
        for term in self.risky_terms:
            if term in lowered:
                findings.append(Finding(Severity.WARNING, f"Prompt contains risky operation term: {term.strip()}."))

        verification = self._section_value(revision, "Verification")
        if verification and "test" not in verification.lower() and "syntax" not in verification.lower():
            findings.append(Finding(Severity.WARNING, "Verification section should mention tests or syntax checks."))
        return ReviewReport(iteration=0, focus="prompt", findings=findings)

    def _section_value(self, revision: PromptRevision, section_name: str) -> str:
        prefix = f"{section_name}:"
        for line in revision.improved.splitlines():
            if line.lower().startswith(prefix.lower()):
                return line.split(":", 1)[1].strip()
        return ""


class PlannerAgent:
    def plan(self, task: Task, workspace: Workspace) -> list[PlanStep]:
        files = workspace.list_files()
        has_package = any(path.parts[:1] == ("src",) for path in files)
        return [
            PlanStep("Inspect workspace", f"Found {len(files)} files."),
            PlanStep("Choose implementation path", "Extend existing Python package." if has_package else "Create a Python package."),
            PlanStep("Generate focused changes", f"Implement the requested goal: {task.goal}"),
            PlanStep("Review and repair", f"Run up to {task.max_reviews} review passes."),
            PlanStep("Verify", "Run syntax checks and unit tests when present."),
        ]


class CoderAgent:
    def draft(self, task: Task, workspace: Workspace, findings: list[Finding] | None = None) -> list[FilePatch]:
        existing = set(workspace.list_files())
        patches: list[FilePatch] = []
        findings = findings or []

        if Path("AGENT_TASK.md") not in existing:
            patches.append(
                FilePatch(
                    Path("AGENT_TASK.md"),
                    self._task_doc(task.goal),
                    "Record the requested coding-agent goal.",
                )
            )

        if any("missing unittest" in finding.message.lower() for finding in findings):
            patches.append(
                FilePatch(
                    Path("tests/test_agent_contract.py"),
                    self._contract_test(),
                    "Add a contract test required by reviewer feedback.",
                )
            )

        return patches

    def repair(self, task: Task, workspace: Workspace, report: ReviewReport) -> list[FilePatch]:
        return self.draft(task, workspace, report.findings)

    def _task_doc(self, goal: str) -> str:
        return (
            "# Agent Task\n\n"
            f"Goal: {goal}\n\n"
            "The agent should inspect the workspace, plan focused edits, review changes, "
            "run verification, and iterate on actionable findings.\n"
        )

    def _contract_test(self) -> str:
        return (
            "import unittest\n"
            "from pathlib import Path\n\n"
            "from coding_agent import AgentOrchestrator\n\n\n"
            "class AgentContractTests(unittest.TestCase):\n"
            "    def test_orchestrator_can_be_constructed(self) -> None:\n"
            "        workspace = Path.cwd()\n"
            "        orchestrator = AgentOrchestrator(workspace)\n"
            "        self.assertEqual(workspace.resolve(), orchestrator.workspace.root)\n"
        )


class ReviewerAgent:
    def review(self, iteration: int, workspace: Workspace, patches: list[FilePatch]) -> ReviewReport:
        findings: list[Finding] = []
        seen_paths: set[Path] = set()

        if not patches and iteration == 1:
            findings.append(Finding(Severity.WARNING, "No changes were proposed."))

        has_test_patch = any(patch.path.parts[:1] == ("tests",) for patch in patches)
        for patch in patches:
            if patch.path in seen_paths:
                findings.append(Finding(Severity.ERROR, "Duplicate patch path in one review batch.", patch.path))
            seen_paths.add(patch.path)
            if patch.path == Path(".") or not patch.path.name:
                findings.append(Finding(Severity.ERROR, "Patch path must point to a file.", patch.path))
            if not patch.reason.strip():
                findings.append(Finding(Severity.ERROR, "Patch reason must explain why the change is needed.", patch.path))
            if patch.path.is_absolute():
                findings.append(Finding(Severity.ERROR, "Patch path must be relative.", patch.path))
            if ".." in patch.path.parts:
                findings.append(Finding(Severity.ERROR, "Patch path must not escape workspace.", patch.path))
            if not patch.content.strip():
                findings.append(Finding(Severity.ERROR, "Patch content must not be empty.", patch.path))
            if "\x00" in patch.content:
                findings.append(Finding(Severity.ERROR, "Patch content must be text, not binary data.", patch.path))
            if len(patch.content) > 20_000:
                findings.append(Finding(Severity.WARNING, "Patch is large; consider splitting it.", patch.path))
            if patch.content and not patch.content.endswith("\n"):
                findings.append(Finding(Severity.WARNING, "Text patches should end with a newline.", patch.path))
            if patch.is_python:
                findings.extend(self._review_python_patch(patch))
                if patch.path.parts[:1] == ("src",) and not has_test_patch:
                    findings.append(Finding(Severity.WARNING, "Source changes should include a related test patch.", patch.path))
            if patch.path.parts[:1] == ("tests",):
                findings.extend(self._review_test_patch(patch))

        if not any(path.parts[:1] == ("tests",) for path in workspace.list_files()) and not any(
            patch.path.parts[:1] == ("tests",) for patch in patches
        ):
            findings.append(Finding(Severity.ERROR, "Missing unittest coverage for agent behavior."))

        return ReviewReport(iteration=iteration, focus="patch", findings=findings)

    def _review_python_patch(self, patch: FilePatch) -> list[Finding]:
        findings: list[Finding] = []
        try:
            ast.parse(patch.content)
        except SyntaxError as exc:
            findings.append(Finding(Severity.ERROR, f"Python syntax error: {exc.msg}", patch.path, exc.lineno))
        if "TODO" in patch.content:
            findings.append(Finding(Severity.WARNING, "Patch contains TODO marker.", patch.path))
        return findings

    def _review_test_patch(self, patch: FilePatch) -> list[Finding]:
        if patch.path.suffix != ".py":
            return []
        if "unittest.TestCase" in patch.content or "def test_" in patch.content:
            return []
        return [Finding(Severity.ERROR, "Test patch must define a discoverable test.", patch.path)]


class ProjectReviewAgent:
    """Reviews the project itself across repeated, rotating focus areas."""

    def __init__(self) -> None:
        self._checks = (
            ("package structure", self._check_package_structure),
            ("test contract", self._check_test_contract),
            ("workspace hygiene", self._check_workspace_hygiene),
            ("cli contract", self._check_cli_contract),
            ("agent architecture", self._check_agent_architecture),
            ("documentation contract", self._check_documentation_contract),
        )

    def review(self, iteration: int, workspace: Workspace) -> ReviewReport:
        focus, check = self._checks[(iteration - 1) % len(self._checks)]
        return ReviewReport(iteration=iteration, focus=focus, findings=check(workspace))

    def review_many(self, workspace: Workspace, passes: int = 100) -> list[ReviewReport]:
        if passes < 1:
            raise ValueError("passes must be at least 1")
        return [self.review(iteration, workspace) for iteration in range(1, passes + 1)]

    def _check_package_structure(self, workspace: Workspace) -> list[Finding]:
        files = set(workspace.list_files())
        required = [
            Path("pyproject.toml"),
            Path("README.md"),
            Path("src/coding_agent/__init__.py"),
            Path("src/coding_agent/cli.py"),
            Path("src/coding_agent/agents.py"),
            Path("src/coding_agent/backend.py"),
            Path("src/coding_agent/models.py"),
            Path("src/coding_agent/orchestrator.py"),
            Path("src/coding_agent/workspace.py"),
        ]
        findings = [
            Finding(Severity.ERROR, f"Required project file is missing: {path}", path)
            for path in required
            if path not in files
        ]
        if Path("pyproject.toml") in files:
            pyproject = workspace.read_text(Path("pyproject.toml"))
            if "coding-agent" not in pyproject or "coding_agent.cli:main" not in pyproject:
                findings.append(Finding(Severity.ERROR, "pyproject must expose the coding-agent CLI script.", Path("pyproject.toml")))
            if 'requires-python = ">=3.11"' not in pyproject:
                findings.append(Finding(Severity.ERROR, "pyproject must require Python 3.11 or newer.", Path("pyproject.toml")))
        return findings

    def _check_test_contract(self, workspace: Workspace) -> list[Finding]:
        files = set(workspace.list_files())
        findings: list[Finding] = []
        if not any(path.parts[:1] == ("tests",) and path.suffix == ".py" for path in files):
            findings.append(Finding(Severity.ERROR, "Project has no Python tests.", Path("tests")))
        test_path = Path("tests/test_orchestrator.py")
        if test_path not in files:
            findings.append(Finding(Severity.ERROR, "Orchestrator test module is missing.", test_path))
            return findings

        content = workspace.read_text(test_path)
        if "test_can_run_exactly_100_reviews" not in content:
            findings.append(Finding(Severity.ERROR, "Tests must cover exactly 100 review passes.", test_path))
        for test_name in (
            "test_reviewer_rejects_path_escape",
            "test_reviewer_rejects_python_syntax_error",
            "test_apply_result_is_verified",
            "test_project_review_cli_does_not_require_goal",
            "test_project_review_cli_fails_if_any_review_fails",
            "test_cli_requires_goal_for_patch_mode",
            "test_workspace_rejects_path_escape_on_apply",
            "test_reviewer_rejects_empty_patch",
            "test_reviewer_rejects_duplicate_patch_paths",
            "test_tester_fails_when_unittest_discovers_zero_tests",
            "test_reviewer_rejects_missing_patch_reason",
            "test_test_result_exposes_command_text",
            "test_agent_run_result_lists_failed_tests",
            "test_reviewer_rejects_directory_patch_path",
            "test_reviewer_warns_when_source_patch_lacks_test_patch",
            "test_tester_compiles_tests_directory",
            "test_review_report_tracks_warnings",
            "test_agent_run_result_counts_review_passes",
            "test_review_summary_aggregates_reports",
            "test_orchestrator_summarizes_project_review",
            "test_project_review_cli_prints_summary",
            "test_prompt_improver_structures_vague_goal",
            "test_prompt_improver_preserves_structured_goal",
            "test_prompt_improver_normalizes_whitespace",
            "test_prompt_improver_trims_long_goal",
            "test_prompt_revision_reports_section_names",
            "test_prompt_revision_reports_missing_sections",
            "test_prompt_revision_scores_quality",
            "test_prompt_improver_emits_all_required_sections",
            "test_prompt_improver_rejects_unknown_section",
            "test_prompt_reviewer_rejects_missing_sections",
            "test_prompt_reviewer_warns_on_risky_terms",
            "test_prompt_reviewer_warns_when_verification_is_weak",
            "test_agent_run_result_fails_when_prompt_review_fails",
            "test_agent_run_cli_json_output",
            "test_project_review_cli_json_output",
            "test_prompt_improver_uses_model_backend",
            "test_prompt_improver_falls_back_when_backend_returns_bad_prompt",
            "test_openai_backend_extracts_output_text",
            "test_cli_rejects_openai_backend_without_api_key",
            "test_agent_run_result_reports_not_run_verification",
            "test_agent_run_result_reports_failed_verification",
            "test_orchestrator_uses_improved_prompt_for_patches",
            "test_cli_prints_prompt_revision",
        ):
            if test_name not in content:
                findings.append(Finding(Severity.ERROR, f"Missing boundary test: {test_name}.", test_path))
        return findings

    def _check_workspace_hygiene(self, workspace: Workspace) -> list[Finding]:
        files = set(workspace.list_files())
        findings: list[Finding] = []
        if Path(".gitignore") not in files:
            findings.append(Finding(Severity.WARNING, "Project should include .gitignore.", Path(".gitignore")))
            return findings

        gitignore = workspace.read_text(Path(".gitignore"))
        for pattern in ("__pycache__/", "*.py[cod]", ".pytest_cache/"):
            if pattern not in gitignore:
                findings.append(Finding(Severity.WARNING, f".gitignore should contain {pattern}.", Path(".gitignore")))
        return findings

    def _check_cli_contract(self, workspace: Workspace) -> list[Finding]:
        cli_path = Path("src/coding_agent/cli.py")
        if cli_path not in set(workspace.list_files()):
            return [Finding(Severity.ERROR, "CLI module is missing.", cli_path)]

        content = workspace.read_text(cli_path)
        findings: list[Finding] = []
        for flag in ("--max-reviews", "--exact-reviews", "--review-project", "--apply", "--json", "--backend", "--model"):
            if flag not in content:
                findings.append(Finding(Severity.ERROR, f"CLI must expose {flag}.", cli_path))
        return findings

    def _check_documentation_contract(self, workspace: Workspace) -> list[Finding]:
        readme_path = Path("README.md")
        if readme_path not in set(workspace.list_files()):
            return [Finding(Severity.ERROR, "README is missing.", readme_path)]

        content = workspace.read_text(readme_path)
        findings: list[Finding] = []
        for phrase in (
            "PromptImproverAgent",
            "PromptReviewAgent",
            "--backend openai",
            "--review-project",
            "--exact-reviews",
            "--json",
            "goal 인자를 요구하지 않습니다",
            "python3 -m unittest discover -s tests",
        ):
            if phrase not in content:
                findings.append(Finding(Severity.ERROR, f"README must document {phrase}.", readme_path))
        return findings

    def _check_agent_architecture(self, workspace: Workspace) -> list[Finding]:
        agents_path = Path("src/coding_agent/agents.py")
        if agents_path not in set(workspace.list_files()):
            return [Finding(Severity.ERROR, "Agent module is missing.", agents_path)]

        content = workspace.read_text(agents_path)
        required_agents = (
            "PromptImproverAgent",
            "PromptReviewAgent",
            "PlannerAgent",
            "CoderAgent",
            "ReviewerAgent",
            "TesterAgent",
            "ProjectReviewAgent",
        )
        return [
            Finding(Severity.ERROR, f"Missing architecture role: {agent_name}.", agents_path)
            for agent_name in required_agents
            if agent_name not in content
        ]


class TesterAgent:
    def verify(self, workspace: Workspace) -> list[TestResult]:
        results: list[TestResult] = []
        if (workspace.root / "src").exists():
            results.append(self._run((sys.executable, "-m", "compileall", "-q", "src"), workspace.root))
        else:
            python_files = [path.as_posix() for path in workspace.list_files() if path.suffix == ".py"]
            if python_files:
                results.append(self._run((sys.executable, "-m", "py_compile", *python_files), workspace.root))

        tests_dir = workspace.root / "tests"
        if tests_dir.exists():
            results.append(self._run((sys.executable, "-m", "compileall", "-q", "tests"), workspace.root))
            results.append(self._run_unittest((sys.executable, "-m", "unittest", "discover", "-s", "tests"), workspace.root))
        return results

    def _run(self, command: tuple[str, ...], cwd: Path) -> TestResult:
        env = os.environ.copy()
        package_src = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = f"{package_src}{os.pathsep}{env.get('PYTHONPATH', '')}"
        completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
        return TestResult(command, completed.returncode, completed.stdout, completed.stderr)

    def _run_unittest(self, command: tuple[str, ...], cwd: Path) -> TestResult:
        result = self._run(command, cwd)
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode == 0 and "Ran 0 tests" in output:
            return TestResult(command, 1, result.stdout, f"{result.stderr}\nNo tests were discovered.".lstrip())
        return result
