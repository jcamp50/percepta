## Testing and CI/CD Plan

This document describes the comprehensive test strategy for both Python (pytest) and Node (Jest), how to structure tests, what to run locally (pre-commit, pre-push), and how CI should run on GitHub Actions. It also notes which `scripts/` items can be migrated or kept.


### Goals
- Provide fast, reliable feedback locally.
- Achieve high unit coverage across Python and Node.
- Run a selected integration set on PRs to gate merges.
- Run full suites on `main` and optionally nightly (including e2e via docker-compose).


## Test Suite Structure

### Python (pytest)
- Layout
  - `tests/unit/...`: pure unit tests with mocks only; no network/DB
  - `tests/integration/...`: DB and HTTP integration (Postgres + pgvector, HTTP endpoints)
  - `tests/e2e/...` (optional): cross-service tests (Node + Python), behind a marker

- Example `pytest.ini`

```ini
[pytest]
testpaths = tests
addopts = -ra -q --maxfail=1 --durations=10 --strict-markers --cov=py --cov-report=term-missing
markers =
  integration: tests that hit services or DB
  e2e: end-to-end multi-service tests
```

- Example `tests/conftest.py`

```python
import asyncio
import os
import pytest

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    # Use a test database or mocks during unit tests
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/percepta_test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    yield
```

- High-priority unit coverage
  - `py/utils/video_embeddings.py`
    - `project_clip_to_1536`: dimension/normalization/invalid inputs
    - `create_grounded_embedding`: fusion weights, normalization, invalid args; mock `embed_text`
  - `py/ingest/transcription.py` (`TranscriptionService`)
    - device/compute selection, lazy `load_model`
    - bytes/file transcription flow; segment to dict; async wrapper
    - error paths and temp file cleanup
  - `py/ingest/video.py` (`determine_capture_interval`)
    - thresholds, keyword list override, fallbacks when stores fail
  - `py/database/connection.py`
    - engine/session factory creation; env override
  - `py/database/models.py`
    - smoke tests for ORM mappings; minimal instance creation
  - Memory/RAG (when present): mock vector/LLM to validate control flow

- Integration coverage
  - Postgres + pgvector insertion/query (vector length, cosine searches)
  - Python HTTP endpoints (selected) using `httpx.AsyncClient` or `requests`
  - Mock Twitch/Streamlink/LLM via `respx/responses`

- E2E (optional, marked `e2e`)
  - Full flow via docker-compose: Node chat→Python ingest→DB. Slow; run on demand.


### Node (Jest)
- Layout
  - `node/__tests__/unit/...`
  - `node/__tests__/integration/...`

- `jest.config.js`

```js
module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.js'],
  collectCoverage: true,
  collectCoverageFrom: ['node/**/*.js', '!node/**/index.js'],
  coverageThreshold: { global: { lines: 80, statements: 80, functions: 80, branches: 70 } },
  setupFiles: ['<rootDir>/node/__tests__/setupEnv.js'],
  resetMocks: true,
  restoreMocks: true
};
```

- `node/__tests__/setupEnv.js`

```js
process.env.LOG_LEVEL = 'ERROR';
process.env.PYTHON_SERVICE_URL = 'http://localhost:18000';
```

- High-priority unit coverage
  - `node/utils/logger.js`: levels/category filtering, timestamp format, `chat()` decoration
  - `node/chat.js`: initialization, event handlers, forward-to-Python, `sendMessage` validation, polling loop
  - `node/stream.js`: broadcaster ID fetch, stream URL fetch, monitoring loop events, retries, errors
  - `node/video.js`: interval logic, ffmpeg presence check (mock), frame send request, interval adjustment, shutdown
  - `node/audio.js`: chunk detection loop, file readiness backoff, sending, delayed cleanup map, error categorization

- Integration coverage (Node)
  - With mocked/stubbed Python endpoints (`nock`/mocked axios)
  - Optionally point to a local stub server on a test port


## Migrating `scripts/`

Most checks in `scripts/` should be converted into pytest/Jest and deleted after coverage exists:
- Migrate (to `tests/integration/` or unit as appropriate):
  - `test_broadcaster_id*.py`, `test_endpoint_*.py`, `test_comprehensive_monitoring.py`,
    `test_event_*`, `test_embeddings.py`, `test_rag*.py`, `test_pgvector.py`,
    `verify_*`, `smoke_vector_store.py`, `test_rate_limiting_*`, `test_multi_user_parallel.py`,
    `test_simple_api.py`, `test_node_integration.py`, `test_metadata_poller_method.py`.

Keep as scripts (manual/dev-only):
- `init_twitch_oauth.js` (interactive OAuth)
- `setup_db.sql` (or move to migrations later)
- Dev launchers: `start_python_service.ps1`, `start_services_test.ps1` (if team uses them)


## Local Developer Workflow

### Pre-commit (fast)
Run format/lint and ultra-fast unit tests that complete <~30s.

Tools:
- Python: `black`, `ruff`, `pytest -k 'unit and not slow'`
- Node: `eslint --max-warnings 0`, `jest -c jest.config.js --passWithNoTests --selectProjects unit` (or subset)

Example `.pre-commit-config.yaml` (Python side):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
        language_version: python3
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.5
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest fast unit
        entry: bash -c "pytest -q -k 'unit and not slow'"
        language: system
        pass_filenames: false
```

Node pre-commit with Husky (`.husky/pre-commit`):

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[pre-commit] Lint & quick unit tests"
npm run lint
npm run test:unit -- --maxWorkers=50% --passWithNoTests
```


### Pre-push (blocker for broken unit tests)
Run full unit (Python + Node), skip integration/e2e.

Husky `/.husky/pre-push`:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[pre-push] Running full unit tests (Py + Node)"
pytest -q -k "unit and not e2e"
npm run test -- --maxWorkers=50% --passWithNoTests
```


## GitHub Actions (server)

### Pull Request (required checks)
- Name: “CI (unit + integration)”
- Runs:
  - Setup Python/Node
  - Python unit + selected integration (spin Postgres with pgvector; mock external APIs)
  - Node unit + selected integration (mock axios or hit local stub server)
  - Upload combined coverage reports (optional gating on thresholds)

Workflow sketch (single job matrix or two jobs):

```yaml
name: CI (unit + integration)
on:
  pull_request:

jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov responses respx
      - name: Start Postgres (pgvector)
        uses: ankane/setup-postgres@v1
        with:
          postgres-version: 15
      - name: Run pytest (unit + selected integration)
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
        run: |
          pytest -q -m "not e2e"

  node:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - name: Install Node deps
        run: npm ci
      - name: Run Jest (unit + selected integration)
        run: npm test -- --passWithNoTests
```


### Push to `main`
- Run the full suite:
  - All Python unit + integration (can include more integration cases)
  - All Node unit + integration
  - Upload coverage artifacts

Add a separate workflow or reuse the above with `on: push: branches: [ main ]` and a broader test selection (e.g., include DB-heavy tests).


### Nightly (optional) – Full docker-compose E2E
- Spin up Node, Python, Postgres, Redis, etc.
- Run `pytest -m e2e` and a Node E2E subset

```yaml
name: E2E (docker-compose)
on:
  schedule:
    - cron: "0 7 * * *"  # daily 07:00 UTC

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Compose up
        run: docker compose up -d --build
      - name: Wait for health
        run: ./scripts/verify_service.py  # or curl health endpoints with retries
      - name: Python E2E
        run: pytest -q -m e2e
      - name: Compose down
        if: always()
        run: docker compose down -v
```


## Branch Protection (GitHub)
- Protect `main`:
  - Require PR before merge
  - Require status checks to pass:
    - “CI (unit + integration)”
    - Optional: “E2E (docker-compose)” (not required)
  - Optionally enforce linear history / require approvals


## NPM and Pytest Commands / Scripts

`package.json` (examples):

```json
{
  "scripts": {
    "lint": "eslint \"node/**/*.js\"",
    "test": "jest",
    "test:unit": "jest __tests__/unit",
    "test:integration": "jest __tests__/integration --runInBand",
    "coverage": "jest --coverage"
  }
}
```

Python:
- Unit: `pytest -q -k "unit and not e2e"`
- Integration (local DB up): `pytest -q -m integration`
- All except e2e: `pytest -q -m "not e2e"`
- Full: `pytest`


## Notes and Recommendations
- Unit tests must avoid network and filesystem flakiness; mock external IO.
- Integration tests should be explicit and marked; keep them bounded in time.
- E2E tests are valuable but slow—keep them optional or nightly.
- Coverage thresholds:
  - Node global thresholds in `jest.config.js`
  - Python gates via `--cov` and optional `coverage xml` + a coverage action or fail threshold
- Once pytest/Jest equivalents exist, delete redundant `scripts/test_*.py/.js` and keep only dev/ops scripts:
  - `init_twitch_oauth.js`, `setup_db.sql`, `start_python_service.ps1`, `start_services_test.ps1`.


## Quick Start (local)
1) Install deps
- Python: `pip install -r requirements.txt`
- Node: `npm ci`

2) Fast loop
- `pre-commit install` (Python)
- `npx husky install` (Node) then add `.husky` hooks as above

3) Run tests
- Python unit: `pytest -q -k unit`
- Node unit: `npm run test:unit`

4) Before pushing
- `pytest -q -k "unit and not e2e"`
- `npm test -- --passWithNoTests`


