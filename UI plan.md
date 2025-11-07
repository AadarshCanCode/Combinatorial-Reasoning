# CRQUBO UI Redevelopment Plan

## Objectives
- Provide an interactive interface to run CRQUBO reasoning pipelines without relying on Gradio.
- Support both power users (researchers/engineers) and exploratory users (product managers, analysts).
- Integrate telemetry, cost tracking, and pipeline configuration editing.

## Guiding Principles
1. **Decouple UI from core pipeline** via REST/GraphQL API layer.
2. **Modular architecture** to align with CRQUBO pipeline stages.
3. **Observability-first**: instrument UI with analytics and logging.
4. **Security-aware**: no direct exposure of API keys or sensitive configuration.
5. **Extensible**: allow future integrations (Jupyter widgets, VS Code extension, CLI).

## High-Level Architecture
```
[Browser UI] ⇄ [UI Gateway API] ⇄ [CRQUBO Service Layer]
```
- **Browser UI**: React/Next.js SPA with modular components.
- **UI Gateway API**: FastAPI service exposing REST endpoints (query processing, history, config).
- **CRQUBO Service Layer**: Existing pipeline invoked via API service with dependency injection.

## Key Features
1. **Query Console**
   - Query input with domain selector and optional parameters (retrieval, verification, retries).
   - Real-time validation and contextual help.
   - Run button with execution status (queued, running, completed, failed).

2. **Reasoning Chain Viewer**
   - Timeline view for chain-of-thought steps.
   - Tree view for tree-of-thought output (collapsible nodes).
   - Confidence scoring visualization (heat map / gradient bars).
   - Step-level metadata (type, utility score, source).

3. **Optimization Diagnostics**
   - Visual summary of candidate vs. selected steps.
   - Diversity vs. utility scatter plot.
   - Solver performance (time, iterations, fallback used).

4. **Verification Insights**
   - Consistency score display.
   - Highlighted contradictions or circular reasoning.
   - Links to detailed Z3 traces/logs.

5. **Final Inference Summary**
   - Answer card with confidence indicator.
   - Strategy used (direct synthesis, step-by-step, etc.).
   - Evidence references (step numbers, retrieved knowledge snippets).

6. **History & Analytics**
   - Persisted session history with filters (date, domain, status).
   - Export to CSV/JSON.
   - API usage metrics (token counts, cost estimates, runtime).

7. **Configuration Management**
   - Editable config profiles with validation (Pydantic schemas).
   - Toggle modules (retrieval, verification, solver type).
   - Save/duplicate configs, compare diffs.

8. **User Management (Phase 2)**
   - Authentication (OIDC) and role-based access.
   - Personal vs. shared configuration.
   - Audit trail of runs and config changes.

## Technical Stack Recommendation
- **Frontend**: Next.js + TypeScript + Tailwind CSS + Recharts/D3 for visualizations.
- **Backend**: FastAPI + SQLModel/PostgreSQL for persistence + Redis for task queue/cache.
- **Task Execution**: Celery or Dramatiq workers to run CRQUBO pipeline asynchronously.
- **Messaging**: WebSockets or Server-Sent Events for run status updates.
- **Auth**: OAuth2 (Auth0/Okta) with JWT validation.

## API Endpoints (Draft)
- `POST /api/runs` – submit query run request.
- `GET /api/runs/{run_id}` – poll single run result.
- `GET /api/runs` – list runs with filters.
- `POST /api/configs` – save new config profile.
- `GET /api/configs` – list existing configs.
- `GET /api/metrics` – aggregated usage stats.
- `POST /api/replays` – rerun past query with modifications.

## Data Model (Simplified)
- **Run**: id, query, domain, flags, status, start/end time, cost estimate.
- **ReasoningStep**: run_id, order, content, type, confidence, metadata.
- **VerificationIssue**: run_id, step_reference, type, description.
- **ConfigurationProfile**: id, name, payload (JSON), owner.
- **AnalyticsSnapshot**: run_id, tokens_used, latency_breakdown.

## UX Milestones
1. **Phase 0 – API Layer**
   - Implement FastAPI service wrapping CRQUBO pipeline.
   - Add asynchronous job processing and result persistence.
2. **Phase 1 – Minimum Viable UI**
   - Query console, basic result rendering, history list.
   - Simple configuration editor.
3. **Phase 2 – Observability & Analytics**
   - Diagnostics dashboards and cost tracking.
   - Real-time status updates.
4. **Phase 3 – Collaboration Features**
   - User accounts, shared configs, audit trails.
   - Advanced visualization for reasoning graphs.

## Testing Strategy
- Component tests with React Testing Library.
- Integration tests using Playwright.
- API contract tests (pytest + httpx).
- Load testing with Locust targeting API layer.
- UX telemetry instrumentation and error reporting (Sentry/Datadog RUM).

## Migration Considerations
- Provide CLI fallback and Jupyter notebook examples during transition.
- Create data migration to import existing Gradio history if needed.
- Update documentation (README, DEMO_GUIDE) once new UI available.
- Communicate deprecation timeline for Gradio demo.

## Open Questions
- Do we need on-prem support (air-gapped environments)?
- Should pipeline execution be pluggable for batching/offline runs?
- Are there compliance requirements (PII, GDPR) affecting logging/analytics?
- Do we support custom LLM endpoints (Azure OpenAI, GPT-NeoX)?

## Next Steps
1. Finalize API contract and data models.
2. Set up FastAPI skeleton with dependency injection.
3. Prototype query submission flow with mocked pipeline.
4. Define design system and UI mockups.
5. Plan phased rollout and user feedback loops.
