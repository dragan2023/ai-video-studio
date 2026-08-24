# Resumable Batch H3 Planning and LLM Profiles

## Goal
Process imported thick scripts of arbitrary practical length without one request lifetime: persist six-shot H3 enrichment batches, expose progress/retry in Studio, and let creators select a locally configured planner profile without exposing credentials.

## Fixed Decisions
- Batch size is six shots; completed batches persist before the next begins.
- A project records progress and its selected profile snapshot; restart resumes incomplete work.
- The browser sends only a configured profile id and optional model override. Base URLs and keys stay only in local .env.
- Profile changes apply to a new run only; an active project stays pinned to its selected profile.
- Existing manual/import/legacy projects remain readable and use the current default profile when no snapshot exists.

## Tasks
1. Add persistent BatchPlanningStatus/BatchPlanningRun fields to FilmProject and migration-safe defaults; test legacy JSON.
2. Add named planner profiles to Settings from local environment variables and a non-secret public profile view.
3. Refactor PlannerService enrichment to enrich explicit six-shot batches and return each saved batch result.
4. Extend PlanningManager to start/resume/cancel imported-project batch runs, serialize updates, and persist after every batch.
5. Add API endpoints for profile listing, start/status/retry batch planning, and use project profile selection without returning secrets.
6. Add React profile selector and visible progress/errors/retry in the preproduction workbench.
7. Test mocked 70-shot resume and profile isolation; build frontend; run a real full-script job through the Studio background endpoint.

## Verification
- Focused pytest for persistence, manager resume, profiles, API status/retry, and legacy compatibility.
- Vite build.
- Real uploaded 70-shot run is launched through Studio background work, not a single external command timeout.

## Risks
- LLM output remains non-deterministic; preserve existing parser/patch/default fallbacks per batch and surface which batch used fallback.
- No API key can enter a project record, response, trace, or web state.
