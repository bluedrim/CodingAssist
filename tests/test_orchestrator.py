from pathlib import Path
from io import StringIO
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coding_agent import AgentOrchestrator
from coding_agent.agents import PromptImproverAgent, PromptReviewAgent, ReviewerAgent, TesterAgent
from coding_agent.backend import BackendError, OpenAIResponsesBackend
from coding_agent.cli import main
from coding_agent.models import AgentRunResult, FilePatch, Finding, PromptRevision, ReviewReport, ReviewSummary, Severity, TestResult
from coding_agent.workspace import Workspace


class AgentOrchestratorTests(unittest.TestCase):
    def test_dry_run_proposes_task_doc_and_contract_test(self) -> None:
        with self.subTest("dry run"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                result = AgentOrchestrator(tmp_path).run("build an agent", max_reviews=100, apply_changes=False)

                proposed = {patch.path.as_posix() for patch in result.patches}
                self.assertIn("AGENT_TASK.md", proposed)
                self.assertIn("tests/test_agent_contract.py", proposed)
                self.assertTrue(result.reviews[-1].passed)
                self.assertFalse((tmp_path / "AGENT_TASK.md").exists())

    def test_apply_writes_files_and_runs_tests(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            result = AgentOrchestrator(tmp_path).run("build an agent", max_reviews=100, apply_changes=True)

            self.assertTrue((tmp_path / "AGENT_TASK.md").exists())
            self.assertTrue((tmp_path / "tests/test_agent_contract.py").exists())
            self.assertTrue(result.passed)
            self.assertTrue(result.verified)

    def test_rejects_zero_review_limit(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            orchestrator = AgentOrchestrator(Path(directory))

            with self.assertRaisesRegex(ValueError, "max_reviews"):
                orchestrator.run("build", max_reviews=0)

    def test_can_run_exactly_100_reviews(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            result = AgentOrchestrator(Path(directory)).run(
                "build an agent",
                max_reviews=100,
                apply_changes=False,
                exact_reviews=True,
            )

            self.assertEqual(100, len(result.reviews))
            self.assertTrue(result.reviews[-1].passed)

    def test_project_review_runs_100_passes_against_this_project(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = AgentOrchestrator(project_root).review_project(100)

        self.assertEqual(100, len(result))
        self.assertEqual(
            {
                "package structure",
                "test contract",
                "workspace hygiene",
                "cli contract",
                "agent architecture",
                "documentation contract",
            },
            {report.focus for report in result},
        )
        self.assertTrue(all(report.passed for report in result))

    def test_reviewer_rejects_path_escape(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("../outside.py"), "print('bad')\n", "escape")],
            )

            self.assertIn(Severity.ERROR, {finding.severity for finding in report.findings})

    def test_reviewer_rejects_directory_patch_path(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("."), "content\n", "directory")],
            )

            self.assertTrue(any("file" in finding.message.lower() for finding in report.findings))

    def test_reviewer_rejects_python_syntax_error(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("src/broken.py"), "def broken(:\n", "syntax")],
            )

            self.assertTrue(any("syntax error" in finding.message.lower() for finding in report.findings))

    def test_reviewer_warns_when_source_patch_lacks_test_patch(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("src/coding_agent/new_feature.py"), "VALUE = 1\n", "feature")],
            )

            self.assertTrue(any("test patch" in finding.message.lower() for finding in report.findings))

    def test_apply_result_is_verified(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            result = AgentOrchestrator(Path(directory)).run("build an agent", apply_changes=True)

            self.assertTrue(result.verified)

    def test_applied_result_without_tests_is_not_passed(self) -> None:
        result = AgentRunResult(
            goal="manual",
            prompt_revision=PromptRevision("manual", "manual", ()),
            plan=[],
            patches=[],
            reviews=[ReviewReport(iteration=1)],
            tests=[],
            applied=True,
        )

        self.assertFalse(result.verified)
        self.assertFalse(result.passed)

    def test_project_review_cli_fails_if_any_review_fails(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            with patch("sys.stdout"):
                exit_code = main(["review", "--workspace", directory, "--review-project", "--max-reviews", "2"])

            self.assertEqual(1, exit_code)

    def test_project_review_cli_does_not_require_goal(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        with patch("sys.stdout"):
            exit_code = main(["--workspace", str(project_root), "--review-project", "--max-reviews", "5"])

        self.assertEqual(0, exit_code)

    def test_cli_rejects_non_positive_review_count(self) -> None:
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                main(["--review-project", "--max-reviews", "0"])

        self.assertEqual(2, raised.exception.code)

    def test_cli_requires_goal_for_patch_mode(self) -> None:
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                main([])

        self.assertEqual(2, raised.exception.code)

    def test_workspace_rejects_path_escape_on_apply(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory))

            with self.assertRaisesRegex(ValueError, "escapes workspace"):
                workspace.apply_patch(FilePatch(Path("../outside.txt"), "bad\n", "escape"))

    def test_reviewer_rejects_empty_patch(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("empty.txt"), "", "empty")],
            )

            self.assertTrue(any("empty" in finding.message.lower() for finding in report.findings))

    def test_reviewer_warns_when_patch_lacks_final_newline(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("note.txt"), "missing newline", "newline")],
            )

            self.assertIn(Severity.WARNING, {finding.severity for finding in report.findings})

    def test_reviewer_rejects_duplicate_patch_paths(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [
                    FilePatch(Path("same.txt"), "one\n", "first"),
                    FilePatch(Path("same.txt"), "two\n", "second"),
                ],
            )

            self.assertTrue(any("duplicate" in finding.message.lower() for finding in report.findings))

    def test_reviewer_rejects_binary_patch_content(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("data.bin"), "bad\x00data\n", "binary")],
            )

            self.assertTrue(any("binary" in finding.message.lower() for finding in report.findings))

    def test_reviewer_rejects_missing_patch_reason(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("note.txt"), "content\n", "")],
            )

            self.assertTrue(any("reason" in finding.message.lower() for finding in report.findings))

    def test_reviewer_warns_for_large_patch(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            report = ReviewerAgent().review(
                1,
                Workspace(Path(directory)),
                [FilePatch(Path("large.txt"), f"{'x' * 20_001}\n", "large")],
            )

            self.assertIn(Severity.WARNING, {finding.severity for finding in report.findings})

    def test_tester_fails_when_unittest_discovers_zero_tests(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "empty.py").write_text("# no tests\n", encoding="utf-8")

            results = TesterAgent().verify(Workspace(root))

            self.assertFalse(results[-1].passed)
            self.assertIn("No tests were discovered", results[-1].stderr)

    def test_tester_compiles_tests_directory(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "test_broken.py").write_text("def broken(:\n", encoding="utf-8")

            results = TesterAgent().verify(Workspace(root))

            self.assertFalse(results[0].passed)

    def test_test_result_exposes_command_text(self) -> None:
        result = TestResult(("python3", "-m", "unittest"), 0, "", "")

        self.assertEqual("python3 -m unittest", result.command_text)

    def test_agent_run_result_lists_failed_tests(self) -> None:
        failed = TestResult(("bad",), 1, "", "failed")
        passed = TestResult(("good",), 0, "", "")
        result = AgentRunResult(
            goal="manual",
            prompt_revision=PromptRevision("manual", "manual", ()),
            plan=[],
            patches=[],
            reviews=[ReviewReport(iteration=1)],
            tests=[passed, failed],
            applied=True,
        )

        self.assertEqual([failed], result.failed_tests)

    def test_agent_run_result_counts_review_passes(self) -> None:
        result = AgentRunResult(
            goal="manual",
            prompt_revision=PromptRevision("manual", "manual", ()),
            plan=[],
            patches=[],
            reviews=[ReviewReport(iteration=1), ReviewReport(iteration=2)],
            tests=[],
            applied=False,
        )

        self.assertEqual(2, result.review_pass_count)

    def test_prompt_improver_structures_vague_goal(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            revision = PromptImproverAgent().improve("add cache", Workspace(Path(directory)))

            self.assertTrue(revision.changed)
            self.assertIn("Objective: add cache", revision.improved)
            self.assertIn("Verification:", revision.improved)
            self.assertIn("Done:", revision.improved)

    def test_prompt_improver_preserves_structured_goal(self) -> None:
        structured = "Objective: build\nVerification: test\nDone: shipped"
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            revision = PromptImproverAgent().improve(structured, Workspace(Path(directory)))

            self.assertFalse(revision.changed)
            self.assertEqual(structured, revision.improved)

    def test_prompt_improver_normalizes_whitespace(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            revision = PromptImproverAgent().improve(" add   cache\n now ", Workspace(Path(directory)))

            self.assertIn("Objective: add cache now", revision.improved)
            self.assertIn("Normalized whitespace.", revision.changes)

    def test_prompt_improver_trims_long_goal(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            revision = PromptImproverAgent().improve("x" * 700, Workspace(Path(directory)))

            objective = revision.improved.splitlines()[0]
            self.assertLessEqual(len(objective.removeprefix("Objective: ")), PromptImproverAgent.max_goal_chars)
            self.assertTrue(any("Trimmed goal" in change for change in revision.changes))

    def test_prompt_revision_reports_section_names(self) -> None:
        revision = PromptRevision(
            original="build",
            improved="Objective: build\nVerification: test\nDone: shipped",
            changes=(),
        )

        self.assertEqual(("Objective", "Verification", "Done"), revision.section_names)

    def test_prompt_revision_reports_missing_sections(self) -> None:
        revision = PromptRevision(
            original="build",
            improved="Objective: build\nVerification: test\nDone: shipped",
            changes=(),
        )

        self.assertEqual(("Context", "Constraints"), revision.missing_sections)

    def test_prompt_revision_scores_quality(self) -> None:
        revision = PromptRevision(
            original="build",
            improved="Objective: build\nContext: repo\nConstraints: safe\nVerification: test\nDone: shipped",
            changes=(),
        )

        self.assertEqual(1.0, revision.quality_score)

    def test_prompt_improver_emits_all_required_sections(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            revision = PromptImproverAgent().improve("add cache", Workspace(Path(directory)))

            self.assertEqual((), revision.missing_sections)
            self.assertEqual(PromptImproverAgent.required_sections, revision.section_names)

    def test_prompt_improver_rejects_unknown_section(self) -> None:
        improver = PromptImproverAgent()

        with self.assertRaisesRegex(ValueError, "unknown prompt section"):
            improver._section("Risk", "low")

    def test_prompt_reviewer_rejects_missing_sections(self) -> None:
        revision = PromptRevision(
            original="build",
            improved="Objective: build\nVerification: test\nDone: shipped",
            changes=(),
        )

        report = PromptReviewAgent().review(revision)

        self.assertFalse(report.passed)
        self.assertTrue(any("missing required section" in finding.message for finding in report.findings))

    def test_prompt_reviewer_warns_on_risky_terms(self) -> None:
        revision = PromptRevision(
            original="cleanup",
            improved=(
                "Objective: cleanup with rm -rf\n"
                "Context: repo\n"
                "Constraints: safe\n"
                "Verification: test\n"
                "Done: shipped"
            ),
            changes=(),
        )

        report = PromptReviewAgent().review(revision)

        self.assertIn(Severity.WARNING, {finding.severity for finding in report.findings})

    def test_prompt_reviewer_warns_when_verification_is_weak(self) -> None:
        revision = PromptRevision(
            original="build",
            improved=(
                "Objective: build\n"
                "Context: repo\n"
                "Constraints: safe\n"
                "Verification: look manually\n"
                "Done: shipped"
            ),
            changes=(),
        )

        report = PromptReviewAgent().review(revision)

        self.assertTrue(any("Verification section" in finding.message for finding in report.findings))

    def test_agent_run_result_fails_when_prompt_review_fails(self) -> None:
        result = AgentRunResult(
            goal="manual",
            prompt_revision=PromptRevision("manual", "Objective: manual", ()),
            plan=[],
            patches=[],
            reviews=[ReviewReport(iteration=1)],
            tests=[],
            applied=False,
            prompt_review=ReviewReport(
                iteration=0,
                focus="prompt",
                findings=[Finding(Severity.ERROR, "bad prompt")],
            ),
        )

        self.assertFalse(result.passed)

    def test_agent_run_result_reports_not_run_verification(self) -> None:
        result = AgentRunResult(
            goal="manual",
            prompt_revision=PromptRevision("manual", "manual", ()),
            plan=[],
            patches=[],
            reviews=[ReviewReport(iteration=1)],
            tests=[],
            applied=False,
        )

        self.assertEqual("not_run", result.verification_status)

    def test_agent_run_result_reports_failed_verification(self) -> None:
        result = AgentRunResult(
            goal="manual",
            prompt_revision=PromptRevision("manual", "manual", ()),
            plan=[],
            patches=[],
            reviews=[ReviewReport(iteration=1)],
            tests=[TestResult(("test",), 1, "", "failed")],
            applied=True,
        )

        self.assertEqual("failed", result.verification_status)

    def test_agent_run_cli_json_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            stdout = StringIO()
            with patch("sys.stdout", stdout):
                exit_code = main(["add cache", "--workspace", directory, "--json"])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, exit_code)
            self.assertEqual("agent_run", payload["kind"])
            self.assertTrue(payload["passed"])
            self.assertEqual("not_run", payload["verification_status"])
            self.assertIn("prompt_revision", payload)

    def test_project_review_cli_json_output(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        stdout = StringIO()

        with patch("sys.stdout", stdout):
            exit_code = main(["--workspace", str(project_root), "--review-project", "--max-reviews", "6", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("project_review", payload["kind"])
        self.assertEqual(6, payload["summary"]["total_reviews"])

    def test_prompt_improver_uses_model_backend(self) -> None:
        class FakeBackend:
            def complete(self, prompt: str) -> str:
                return (
                    "Objective: backend objective\n"
                    "Context: backend context\n"
                    "Constraints: backend constraints\n"
                    "Verification: backend test\n"
                    "Done: backend done"
                )

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            revision = PromptImproverAgent(FakeBackend()).improve("add cache", Workspace(Path(directory)))

        self.assertIn("backend objective", revision.improved)
        self.assertIn("Improved by model backend.", revision.changes)

    def test_prompt_improver_falls_back_when_backend_returns_bad_prompt(self) -> None:
        class BadBackend:
            def complete(self, prompt: str) -> str:
                return "Objective: incomplete"

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            revision = PromptImproverAgent(BadBackend()).improve("add cache", Workspace(Path(directory)))

        self.assertEqual((), revision.missing_sections)
        self.assertIn("Used deterministic fallback prompt.", revision.changes)

    def test_openai_backend_extracts_output_text(self) -> None:
        backend = object.__new__(OpenAIResponsesBackend)

        text = backend._extract_text({"output_text": "hello"})

        self.assertEqual("hello", text)

    def test_cli_rejects_openai_backend_without_api_key(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit) as raised:
                    main(["add cache", "--backend", "openai"])

        self.assertEqual(2, raised.exception.code)

    def test_orchestrator_uses_improved_prompt_for_patches(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            result = AgentOrchestrator(Path(directory)).run("add cache")

            self.assertTrue(result.prompt_revision.changed)
            self.assertIn("Objective: add cache", result.patches[0].content)

    def test_cli_prints_prompt_revision(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            stdout = StringIO()
            with patch("sys.stdout", stdout):
                exit_code = main(["add cache", "--workspace", directory])

            self.assertEqual(0, exit_code)
            self.assertIn("Prompt improved: True", stdout.getvalue())

    def test_review_report_tracks_warnings(self) -> None:
        report = ReviewReport(
            iteration=1,
            findings=[Finding(Severity.WARNING, "check this")],
        )

        self.assertEqual(1, report.warning_count)
        self.assertTrue(report.has_warnings)

    def test_review_summary_aggregates_reports(self) -> None:
        summary = ReviewSummary.from_reports(
            [
                ReviewReport(iteration=1, focus="a", findings=[Finding(Severity.ERROR, "broken")]),
                ReviewReport(iteration=2, focus="a", findings=[Finding(Severity.WARNING, "warn")]),
                ReviewReport(iteration=3, focus="b"),
            ]
        )

        self.assertEqual(3, summary.total_reviews)
        self.assertEqual(1, summary.error_count)
        self.assertEqual(1, summary.warning_count)
        self.assertEqual({"a": 2, "b": 1}, summary.focus_counts)
        self.assertFalse(summary.passed)

    def test_orchestrator_summarizes_project_review(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        summary = AgentOrchestrator(project_root).summarize_project_review(6)

        self.assertEqual(6, summary.total_reviews)
        self.assertTrue(summary.passed)
        self.assertIn("package structure", summary.focus_counts)

    def test_project_review_cli_prints_summary(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        stdout = StringIO()

        with patch("sys.stdout", stdout):
            exit_code = main(["--workspace", str(project_root), "--review-project", "--max-reviews", "6"])

        self.assertEqual(0, exit_code)
        self.assertIn("Project review summary", stdout.getvalue())

    def test_finding_location_formats_path_and_line(self) -> None:
        from coding_agent.models import Finding

        finding = Finding(Severity.ERROR, "broken", Path("src/app.py"), 12)

        self.assertEqual("src/app.py:12", finding.location)

    def test_workspace_exists_respects_workspace_boundary(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory))

            self.assertFalse(workspace.exists(Path("missing.txt")))
            with self.assertRaisesRegex(ValueError, "escapes workspace"):
                workspace.exists(Path("../outside.txt"))
