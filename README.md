# Codex-Style Coding Agent

Python 기반 멀티 에이전트 coding agent 예제입니다. Codex에서 중요한 작업 방식을 로컬 알고리즘으로 옮겼습니다.

- 작업을 작게 나누는 `PlannerAgent`
- 모호한 요청을 구조화된 작업 프롬프트로 개선하는 `PromptImproverAgent`
- 개선된 프롬프트의 누락 섹션과 위험 신호를 점검하는 `PromptReviewAgent`
- 파일을 읽고 변경안을 만드는 `CoderAgent`
- 결함과 누락 테스트를 찾는 `ReviewerAgent`
- 검증 명령을 실행하는 `TesterAgent`
- 최대 100회까지 리뷰와 수정을 반복하는 `AgentOrchestrator`

이 프로젝트는 외부 API 없이 표준 라이브러리만으로 동작합니다. LLM을 붙이고 싶다면 `ModelBackend` 인터페이스를 구현하면 됩니다.
실행 시 입력 goal은 자동으로 Objective, Context, Constraints, Verification, Done 섹션을 가진 프롬프트로 개선됩니다.
프롬프트 개선 단계는 공백을 정규화하고 지나치게 긴 goal을 제한해 반복 리뷰가 안정적인 입력을 받게 합니다.
OpenAI API에 연결하려면 `OPENAI_API_KEY`를 설정한 뒤 `--backend openai`를 사용합니다.

## 실행

```bash
PYTHONPATH=src python3 -m coding_agent "Add a hello world CLI"
```

OpenAI Responses API로 프롬프트 개선을 수행하려면:

```bash
OPENAI_API_KEY=... PYTHONPATH=src python3 -m coding_agent "Add a hello world CLI" --backend openai --model gpt-5.4-mini
```

기본값은 실제 파일을 바꾸지 않는 dry-run입니다. 변경을 적용하려면:

```bash
PYTHONPATH=src python3 -m coding_agent "Add a hello world CLI" --apply
```

100회 리뷰 루프를 명시하려면:

```bash
PYTHONPATH=src python3 -m coding_agent "Improve this project" --max-reviews 100 --exact-reviews --apply
```

현재 프로젝트 자체를 100회 검토하려면:

```bash
PYTHONPATH=src python3 -m coding_agent --review-project --max-reviews 100
```

프로젝트 리뷰는 패키지 구조, 테스트 계약, 작업공간 위생, CLI 계약, 에이전트 아키텍처를 반복 점검합니다.
`--review-project` 모드는 goal 인자를 요구하지 않습니다.
자동화에서 쓰려면 `--json`으로 machine-readable output을 받을 수 있습니다.

## 테스트

```bash
python3 -m unittest discover -s tests
```
