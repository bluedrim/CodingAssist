from __future__ import annotations

from pathlib import Path

from coding_agent.agents import (
    CoderAgent,
    PlannerAgent,
    ProjectReviewAgent,
    PromptImproverAgent,
    PromptReviewAgent,
    ReviewerAgent,
    TesterAgent,
)
from coding_agent.backend import ModelBackend
from coding_agent.models import AgentRunResult, FilePatch, ReviewReport, ReviewSummary, Severity, Task
from coding_agent.workspace import Workspace


class AgentOrchestrator:
    """Coordinates planning, coding, review, repair, and verification."""

    def __init__(self, workspace: Path, backend: ModelBackend | None = None) -> None:
        self.workspace = Workspace(workspace)
        self.prompt_improver = PromptImproverAgent(backend)
        self.prompt_reviewer = PromptReviewAgent()
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.reviewer = ReviewerAgent()
        self.project_reviewer = ProjectReviewAgent()
        self.tester = TesterAgent()

    def run(
        self,
        goal: str,
        max_reviews: int = 100,
        apply_changes: bool = False,
        exact_reviews: bool = False,
    ) -> AgentRunResult:
        if max_reviews < 1:
            raise ValueError("max_reviews must be at least 1")

        prompt_revision = self.prompt_improver.improve(goal, self.workspace)
        prompt_review = self.prompt_reviewer.review(prompt_revision)
        task = Task(
            goal=prompt_revision.improved,
            workspace=self.workspace.root,
            max_reviews=max_reviews,
            apply_changes=apply_changes,
            exact_reviews=exact_reviews,
        )
        plan = self.planner.plan(task, self.workspace)
        patches = self.coder.draft(task, self.workspace)
        reviews: list[ReviewReport] = []

        for iteration in range(1, max_reviews + 1):
            report = self.reviewer.review(iteration, self.workspace, patches)
            reviews.append(report)
            if report.passed and not exact_reviews:
                break
            if not report.passed:
                patches = self._merge_patches(patches, self.coder.repair(task, self.workspace, report))

        if apply_changes:
            final_report = reviews[-1] if reviews else None
            if not prompt_review.passed:
                raise RuntimeError("Refusing to apply changes while prompt review errors remain.")
            if final_report and any(finding.severity == Severity.ERROR for finding in final_report.findings):
                raise RuntimeError("Refusing to apply changes while review errors remain.")
            for patch in patches:
                self.workspace.apply_patch(patch)

        tests = self.tester.verify(self.workspace) if apply_changes else []
        return AgentRunResult(goal, prompt_revision, plan, patches, reviews, tests, apply_changes, prompt_review)

    def _merge_patches(self, current: list[FilePatch], repairs: list[FilePatch]) -> list[FilePatch]:
        merged = {patch.path: patch for patch in current}
        for patch in repairs:
            merged[patch.path] = patch
        return list(merged.values())

    def review_project(self, passes: int = 100) -> list[ReviewReport]:
        return self.project_reviewer.review_many(self.workspace, passes)

    def summarize_project_review(self, passes: int = 100) -> ReviewSummary:
        return ReviewSummary.from_reports(self.review_project(passes))
