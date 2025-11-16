# Integration Tests Setup

Integration tests require a running Postgres database with pgvector extension.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. Start Postgres with pgvector:
   ```bash
   docker-compose up -d postgres
   ```

2. Create the test database:
   ```bash
   docker exec -it percepta-postgres psql -U postgres -c "CREATE DATABASE percepta_test;"
   docker exec -it percepta-postgres psql -U postgres -d percepta_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. Run integration tests:
   ```bash
   pytest -q -m integration
   ```

### Option 2: Using Environment Variable

Set `TEST_DATABASE_URL` to point to your test database:

```bash
export TEST_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/percepta_test"
pytest -q -m integration
```

On Windows PowerShell:
```powershell
$env:TEST_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/percepta_test"
pytest -q -m integration
```

## Default Configuration

- **Host**: localhost
- **Port**: 5432 (matches docker-compose.yml)
- **Database**: percepta_test
- **User**: postgres
- **Password**: postgres

## Skipping Tests

If the database is not available, tests will be automatically skipped with a clear message.

## CI/CD

In CI environments (GitHub Actions), the database is automatically started as a service. See `.github/workflows/ci.yml` for details.

