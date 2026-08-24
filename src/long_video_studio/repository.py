from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from long_video_studio.domain import AssetRecord, FilmProject, LLMClient, RenderJob, RenderObservation, utc_now
from long_video_studio.h3_limits import H3_MAX_SHOT_SECONDS


class StudioRepository:
    """Small SQLite repository storing versionable domain objects as JSON."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id);

                CREATE TABLE IF NOT EXISTS render_observations (
                    id TEXT PRIMARY KEY,
                    source_key TEXT NOT NULL UNIQUE,
                    project_id TEXT NOT NULL,
                    shot_id TEXT NOT NULL,
                    render_profile TEXT NOT NULL,
                    task TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_render_observations_profile_task
                    ON render_observations(render_profile, task, created_at);

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_clients (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _dump(value: AssetRecord | FilmProject | RenderJob | RenderObservation) -> str:
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))

    def save_asset(self, asset: AssetRecord) -> AssetRecord:
        existing = self.get_asset_by_sha256(asset.sha256)
        # A different ID with the same digest is an ingest-time duplicate, so
        # merge its metadata. Saving the same ID is an explicit edit and must
        # replace roles/tags instead of keeping stale values.
        if existing and existing.id != asset.id:
            merged = existing.model_copy(
                update={
                    "caption": asset.caption or existing.caption,
                    "tags": sorted(set(existing.tags) | set(asset.tags)),
                    "roles": list(dict.fromkeys([*existing.roles, *asset.roles])),
                }
            )
            asset = merged
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assets(id, sha256, kind, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    sha256=excluded.sha256,
                    kind=excluded.kind,
                    payload=excluded.payload
                """,
                (
                    asset.id,
                    asset.sha256,
                    asset.kind.value,
                    asset.created_at.isoformat(),
                    self._dump(asset),
                ),
            )
        return asset

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return AssetRecord.model_validate_json(row["payload"]) if row else None

    def get_asset_by_sha256(self, sha256: str) -> AssetRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM assets WHERE sha256 = ?", (sha256,)).fetchone()
        return AssetRecord.model_validate_json(row["payload"]) if row else None

    def list_assets(self) -> list[AssetRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM assets ORDER BY created_at DESC").fetchall()
        return [AssetRecord.model_validate_json(row["payload"]) for row in rows]

    def save_assets(self, assets: Iterable[AssetRecord]) -> list[AssetRecord]:
        return [self.save_asset(asset) for asset in assets]

    def delete_asset(self, asset_id: str) -> AssetRecord | None:
        asset = self.get_asset(asset_id)
        if not asset:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
        return asset

    def save_project(self, project: FilmProject) -> FilmProject:
        project.updated_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects(id, created_at, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    project.id,
                    project.created_at.isoformat(),
                    project.updated_at.isoformat(),
                    self._dump(project),
                ),
            )
        return project

    def get_project(self, project_id: str) -> FilmProject | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._load_project(row["payload"]) if row else None

    def list_projects(self) -> list[FilmProject]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM projects ORDER BY updated_at DESC").fetchall()
        return [self._load_project(row["payload"]) for row in rows]

    def delete_project(self, project_id: str) -> FilmProject | None:
        """Delete a project and its jobs atomically while preserving shared assets."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if row is None:
                return None
            project = self._load_project(row["payload"])
            connection.execute("DELETE FROM jobs WHERE project_id = ?", (project_id,))
            connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return project

    @staticmethod
    def _load_project(payload: str) -> FilmProject:
        """Load legacy projects while keeping the H3 output-duration ceiling.

        Existing projects may contain a historical 15s shot. They remain
        readable for review; any edit/replan/render path validates the new
        output limit before sending a request to H3.
        """

        data = json.loads(payload)
        for shot in data.get("shots", []):
            if isinstance(shot, dict) and float(shot.get("duration_seconds", 0)) > H3_MAX_SHOT_SECONDS:
                shot["legacy_duration_seconds"] = shot["duration_seconds"]
                shot["duration_seconds"] = H3_MAX_SHOT_SECONDS
                for line in shot.get("dialogue", []):
                    if isinstance(line, dict):
                        if line.get("start_seconds") is not None:
                            line["start_seconds"] = min(
                                float(line["start_seconds"]),
                                H3_MAX_SHOT_SECONDS - 0.5,
                            )
                        if line.get("end_seconds") is not None:
                            line["end_seconds"] = min(
                                float(line["end_seconds"]),
                                H3_MAX_SHOT_SECONDS,
                            )
                            if line.get("start_seconds") is not None and line["end_seconds"] <= line["start_seconds"]:
                                line["end_seconds"] = min(
                                    H3_MAX_SHOT_SECONDS,
                                    line["start_seconds"] + 0.5,
                                )
                beats = shot.get("visual_beats") or []
                if beats:
                    beats[-1]["end_seconds"] = H3_MAX_SHOT_SECONDS
        return FilmProject.model_validate(data)

    def get_active_planner(self) -> dict[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = 'active_planner'"
            ).fetchone()
        if not row:
            return {"profile_id": "default", "model": ""}
        try:
            data = json.loads(row["value"])
        except (TypeError, ValueError):
            return {"profile_id": "default", "model": ""}
        return {
            "profile_id": str(data.get("profile_id", "default")),
            "model": str(data.get("model", "")),
        }

    def set_active_planner(self, profile_id: str, model: str = "") -> dict[str, str]:
        value = json.dumps(
            {"profile_id": profile_id, "model": model or ""},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value) VALUES ('active_planner', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (value,),
            )
        return {"profile_id": profile_id, "model": model or ""}

    def list_llm_clients(self) -> list[LLMClient]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM llm_clients ORDER BY created_at ASC"
            ).fetchall()
        return [LLMClient.model_validate_json(row["payload"]) for row in rows]

    def get_llm_client(self, client_id: str) -> LLMClient | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM llm_clients WHERE id = ?", (client_id,)
            ).fetchone()
        return LLMClient.model_validate_json(row["payload"]) if row else None

    def save_llm_client(self, client: LLMClient) -> LLMClient:
        client.updated_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_clients(id, created_at, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    client.id,
                    client.created_at.isoformat(),
                    client.updated_at.isoformat(),
                    json.dumps(client.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
                ),
            )
        return client

    def delete_llm_client(self, client_id: str) -> LLMClient | None:
        existing = self.get_llm_client(client_id)
        if not existing:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM llm_clients WHERE id = ?", (client_id,))
        return existing

    def save_job(self, job: RenderJob) -> RenderJob:
        job.updated_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(id, project_id, created_at, updated_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    job.id,
                    job.project_id,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    self._dump(job),
                ),
            )
        return job

    def get_job(self, job_id: str) -> RenderJob | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return RenderJob.model_validate_json(row["payload"]) if row else None

    def get_latest_job(self, project_id: str) -> RenderJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM jobs WHERE project_id = ? ORDER BY updated_at DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        return RenderJob.model_validate_json(row["payload"]) if row else None

    def list_active_jobs(self) -> list[RenderJob]:
        """Return the latest queued/running job for each project."""

        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM jobs ORDER BY updated_at DESC").fetchall()
        active: list[RenderJob] = []
        seen_projects: set[str] = set()
        for row in rows:
            job = RenderJob.model_validate_json(row["payload"])
            if job.project_id in seen_projects:
                continue
            seen_projects.add(job.project_id)
            if job.status in {"queued", "running"}:
                active.append(job)
        return active

    def save_render_observation(self, observation: RenderObservation) -> RenderObservation:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO render_observations(
                    id, source_key, project_id, shot_id, render_profile,
                    task, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO NOTHING
                """,
                (
                    observation.id,
                    observation.source_key,
                    observation.project_id,
                    observation.shot_id,
                    observation.render_profile,
                    observation.task.value,
                    observation.created_at.isoformat(),
                    self._dump(observation),
                ),
            )
        return observation

    def list_render_observations(self, render_profile: str | None = None) -> list[RenderObservation]:
        query = "SELECT payload FROM render_observations"
        parameters: tuple[str, ...] = ()
        if render_profile is not None:
            query += " WHERE render_profile = ?"
            parameters = (render_profile,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [RenderObservation.model_validate_json(row["payload"]) for row in rows]
