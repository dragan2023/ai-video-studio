from __future__ import annotations

from dataclasses import dataclass, replace

from long_video_studio.assets import AssetService
from long_video_studio.compiler import FilmCompiler
from long_video_studio.config import Settings
from long_video_studio.domain import LLMClient
from long_video_studio.estimator import RenderEstimator
from long_video_studio.planner import PlannerError, PlannerService
from long_video_studio.preproduction import PreproductionPlanner
from long_video_studio.preproduction_assets import PreproductionAssetGenerator
from long_video_studio.repository import StudioRepository
from long_video_studio.service_status import ServiceStatusCollector


@dataclass
class StudioServices:
    settings: Settings
    repository: StudioRepository
    assets: AssetService
    planner: PlannerService
    preproduction: PreproductionPlanner
    preproduction_assets: PreproductionAssetGenerator
    compiler: FilmCompiler
    estimator: RenderEstimator
    service_status: ServiceStatusCollector
    active_planner_profile_id: str = "default"
    active_planner_model: str = ""

    @classmethod
    def create(cls, settings: Settings) -> StudioServices:
        settings.ensure_directories()
        repository = StudioRepository(settings.database_path)
        assets = AssetService(settings, repository)
        estimator = RenderEstimator(settings, repository)
        estimator.backfill()
        active = repository.get_active_planner()
        services = cls(
            settings=settings,
            repository=repository,
            assets=assets,
            planner=PlannerService(settings, repository),
            preproduction=PreproductionPlanner(),
            preproduction_assets=PreproductionAssetGenerator(settings, repository, assets),
            compiler=FilmCompiler(settings, estimator=estimator, repository=repository),
            estimator=estimator,
            service_status=ServiceStatusCollector(settings),
            active_planner_profile_id=active["profile_id"],
            active_planner_model=active["model"],
        )
        # Runtime status must report the model-management default, not stale env
        # values.  active_planner_view is read live so switching the default in
        # the UI is reflected on the next status poll without a restart.
        services.service_status.planner_view = services.active_planner_view
        return services

    def _resolve_client_config(self, profile_id: str) -> dict[str, str | bool] | None:
        """Resolve an active planner client from env profiles or persisted clients."""
        try:
            profile = self.settings.planner_profile(profile_id)
        except ValueError:
            profile = None
        if profile is not None:
            return {
                "base_url": profile.base_url or "",
                "api_key": profile.api_key,
                "model": profile.model or "",
                "wire_api": profile.wire_api,
                "display_name": profile.display_name,
                "available": bool(profile.base_url and profile.api_key and profile.model),
            }
        client = self.repository.get_llm_client(profile_id)
        if client is not None:
            return {
                "base_url": client.base_url or "",
                "api_key": client.api_key,
                "model": client.model or "",
                "wire_api": client.wire_api,
                "display_name": client.display_name or client.id,
                "available": bool(client.base_url.strip() and client.model.strip()),
            }
        return None

    def set_active_planner(self, profile_id: str, model: str = "") -> dict[str, str | bool]:
        profile_id = profile_id or "default"
        model = (model or "").strip()
        config = self._resolve_client_config(profile_id)
        if config is None:
            raise ValueError(f"unknown planner client: {profile_id}")
        if not config["available"]:
            raise ValueError(f"Planner client is not configured or unavailable: {profile_id}")
        self.active_planner_profile_id = profile_id
        self.active_planner_model = model
        self.repository.set_active_planner(profile_id, model)
        return self.active_planner_view()

    def active_planner_view(self) -> dict[str, str | bool]:
        profile_id = self.active_planner_profile_id
        model = self.active_planner_model
        config = self._resolve_client_config(profile_id)
        if config is None:
            display_name = "Default" if profile_id == "default" else profile_id
            available = profile_id == "default" and bool(self.settings.planner_base_url)
            resolved_model = model or self.settings.planner_model or ""
        else:
            display_name = str(config["display_name"])
            available = bool(config["available"])
            resolved_model = model or str(config["model"] or "")
        return {
            "profile_id": profile_id,
            "model": model,
            "display_name": display_name,
            "resolved_model": resolved_model,
            "available": available,
        }

    def list_llm_clients(self) -> list[dict[str, object]]:
        """Return env + persisted clients, annotating the active default."""
        active_id = self.active_planner_profile_id
        env = [
            {
                **profile.public(),
                "source": "env",
                "is_default": profile.id == active_id,
            }
            for profile in self.settings.planner_profiles
        ]
        seen = {item["id"] for item in env}
        db = [
            {
                **client.public(),
                "source": "db",
                "is_default": client.id == active_id,
            }
            for client in self.repository.list_llm_clients()
            if client.id not in seen
        ]
        return [*env, *db]

    def create_llm_client(self, payload: LLMClient) -> dict[str, object]:
        client = payload.model_copy(update={"id": payload.id.strip().lower()})
        if self._resolve_client_config(client.id) is not None:
            # The env default profile shadows DB ids; do not allow silent dupes.
            raise ValueError(f"llm client id already exists: {client.id}")
        if not client.base_url.strip() or not client.model.strip():
            raise ValueError("base_url and model are required")
        saved = self.repository.save_llm_client(client)
        return {"client": saved.public()}

    def update_llm_client(self, client_id: str, payload: dict[str, object]) -> dict[str, object]:
        existing = self.repository.get_llm_client(client_id)
        if existing is None:
            raise KeyError(client_id)
        updates = dict(payload)
        api_key = updates.pop("api_key", None)
        # Keep the stored key when the caller does not supply a new one or
        # leaves the field blank; only a non-empty value replaces it.
        if api_key is None or (isinstance(api_key, str) and not api_key.strip()):
            updates["api_key"] = existing.api_key
        else:
            updates["api_key"] = str(api_key).strip()
        updates.setdefault("id", client_id)
        updates.setdefault("created_at", existing.created_at)
        updates.setdefault("updated_at", existing.updated_at)
        updated = LLMClient.model_validate({**existing.model_dump(mode="python"), **updates})
        if not updated.base_url.strip() or not updated.model.strip():
            raise ValueError("base_url and model are required")
        saved = self.repository.save_llm_client(updated)
        return {"client": saved.public()}

    def delete_llm_client(self, client_id: str) -> dict[str, object]:
        existing = self.repository.get_llm_client(client_id)
        if existing is None:
            raise KeyError(client_id)
        if client_id == self.active_planner_profile_id:
            raise ValueError("cannot delete the active default LLM client; set another default first")
        self.repository.delete_llm_client(client_id)
        return {"deleted": True, "id": client_id}

    def set_default_llm_client(self, client_id: str, model: str = "") -> dict[str, str | bool]:
        return self.set_active_planner(client_id, model)

    def resolve_planner(self, *, profile_id: str | None = None, model: str | None = None) -> PlannerService:
        """Return a PlannerService bound to the selected/active planner client.

        A default env profile without a per-call model override returns the
        shared default planner so existing callers (and tests that patch it)
        keep working.
        """
        selected_id = (profile_id or self.active_planner_profile_id) or "default"
        selected_model = model if model is not None else self.active_planner_model
        if selected_model:
            selected_model = selected_model.strip()
        if selected_id == "default" and not selected_model and not self.repository.get_llm_client("default"):
            return self.planner
        config = self._resolve_client_config(selected_id)
        if config is None:
            raise PlannerError(f"unknown planner profile: {selected_id}")
        if not config["available"]:
            raise PlannerError(f"Planner profile is not configured or unavailable: {selected_id}")
        return PlannerService(
            replace(
                self.settings,
                planner_base_url=str(config["base_url"]) or None,
                planner_api_key=config["api_key"],
                planner_model=selected_model or str(config["model"] or ""),
                planner_wire_api=str(config["wire_api"] or "chat_completions"),
            ),
            self.repository,
        )
