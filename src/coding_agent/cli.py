from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_agent.backend import BackendError, OpenAIResponsesBackend
from coding_agent.orchestrator import AgentOrchestrator
from coding_agent.models import AgentRunResult, Finding, ReviewReport, ReviewSummary, Severity, TestResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Codex-style local coding agent.")
    parser.add_argument("goal", nargs="?", help="Coding goal to execute.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace root.")
    parser.add_argument("--max-reviews", type=int, default=100, help="Maximum review and repair passes.")
    parser.add_argument("--exact-reviews", action="store_true", help="Run exactly --max-reviews review passes.")
    parser.add_argument("--review-project", action="store_true", help="Review the current project instead of drafting patches.")
    parser.add_argument("--apply", action="store_true", help="Write proposed changes to disk.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--backend", choices=("local", "openai"), default="local", help="Prompt improvement backend.")
    parser.add_argument("--model", help="OpenAI model name when --backend openai is used.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_reviews < 1:
        parser.error("--max-reviews must be at least 1")
    if not args.review_project and not args.goal:
        parser.error("goal is required unless --review-project is used")

    backend = None
    if args.backend == "openai":
        try:
            backend = OpenAIResponsesBackend(model=args.model)
        except BackendError as exc:
            parser.error(str(exc))

    orchestrator = AgentOrchestrator(args.workspace, backend=backend)
    if args.review_project:
        reviews = orchestrator.review_project(args.max_reviews)
        summary = ReviewSummary.from_reports(reviews)
        if args.json:
            print(json.dumps(_project_review_payload(summary, reviews), indent=2, sort_keys=True))
            return 0 if summary.passed else 1
        print(
            f"Project review summary: {summary.total_reviews} passes, "
            f"{summary.error_count} errors, {summary.warning_count} warnings"
        )
        print(f"Project review focuses: {summary.focus_counts}")
        for report in reviews:
            print(f"- pass {report.iteration} [{report.focus}]: {report.error_count} errors, {report.warning_count} warnings")
            for finding in report.findings:
                location = f" ({finding.location})" if finding.location else ""
                print(f"  {finding.severity.value}: {finding.message}{location}")
        return 0 if summary.passed else 1

    result = orchestrator.run(
        args.goal or "",
        max_reviews=args.max_reviews,
        apply_changes=args.apply,
        exact_reviews=args.exact_reviews,
    )

    if args.json:
        print(json.dumps(_agent_run_payload(result), indent=2, sort_keys=True))
        return 0 if result.passed else 1

    print(f"Goal: {result.goal}")
    print(f"Prompt improved: {result.prompt_revision.changed}")
    if result.prompt_revision.changed:
        print(result.prompt_revision.improved)
    if result.prompt_review:
        print(
            f"Prompt review: {result.prompt_review.error_count} errors, "
            f"{result.prompt_review.warning_count} warnings"
        )
        for finding in result.prompt_review.findings:
            print(f"  {finding.severity.value}: {finding.message}")
    print(f"Applied: {result.applied}")
    print(f"Verified: {result.verified}")
    print(f"Verification status: {result.verification_status}")
    print(f"Proposed patches: {len(result.patches)}")
    for patch in result.patches:
        print(f"- {patch.path}: {patch.reason}")

    print(f"Review passes: {len(result.reviews)}")
    for report in result.reviews:
        print(f"- pass {report.iteration}: {report.error_count} errors, {report.warning_count} warnings")
        for finding in report.findings:
            location = f" ({finding.location})" if finding.location else ""
            print(f"  {finding.severity.value}: {finding.message}{location}")

    for test in result.tests:
        status = "passed" if test.passed else "failed"
        print(f"Test {status}: {test.command_text}")
        if test.stdout:
            print(test.stdout.rstrip())
        if test.stderr:
            print(test.stderr.rstrip())

    return 0 if result.passed else 1


def _finding_payload(finding: Finding) -> dict[str, object]:
    return {
        "severity": finding.severity.value,
        "message": finding.message,
        "location": finding.location,
    }


def _review_payload(report: ReviewReport) -> dict[str, object]:
    return {
        "iteration": report.iteration,
        "focus": report.focus,
        "passed": report.passed,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "findings": [_finding_payload(finding) for finding in report.findings],
    }


def _test_payload(test: TestResult) -> dict[str, object]:
    return {
        "command": test.command_text,
        "returncode": test.returncode,
        "passed": test.passed,
        "stdout": test.stdout,
        "stderr": test.stderr,
    }


def _project_review_payload(summary: ReviewSummary, reviews: list[ReviewReport]) -> dict[str, object]:
    return {
        "kind": "project_review",
        "passed": summary.passed,
        "summary": {
            "total_reviews": summary.total_reviews,
            "error_count": summary.error_count,
            "warning_count": summary.warning_count,
            "focus_counts": summary.focus_counts,
        },
        "reviews": [_review_payload(report) for report in reviews],
    }


def _agent_run_payload(result: AgentRunResult) -> dict[str, object]:
    return {
        "kind": "agent_run",
        "goal": result.goal,
        "passed": result.passed,
        "applied": result.applied,
        "verified": result.verified,
        "verification_status": result.verification_status,
        "prompt_revision": {
            "changed": result.prompt_revision.changed,
            "original": result.prompt_revision.original,
            "improved": result.prompt_revision.improved,
            "changes": list(result.prompt_revision.changes),
            "missing_sections": list(result.prompt_revision.missing_sections),
            "quality_score": result.prompt_revision.quality_score,
        },
        "prompt_review": _review_payload(result.prompt_review) if result.prompt_review else None,
        "patches": [
            {"path": patch.path.as_posix(), "reason": patch.reason}
            for patch in result.patches
        ],
        "reviews": [_review_payload(report) for report in result.reviews],
        "tests": [_test_payload(test) for test in result.tests],
    }
