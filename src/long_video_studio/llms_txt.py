from __future__ import annotations

from textwrap import dedent, indent


def _agent_connection(base: str, *, mcp_enabled: bool, mcp_requires_token: bool) -> str:
    if not mcp_enabled:
        return dedent(
            f"""
            MCP is disabled for this deployment. Agents can inspect
            `{base}/openapi.json` and use the REST API, or ask the operator to enable
            `STUDIO_MCP_ENABLED` before attempting MCP operations.
            """
        ).strip()
    if mcp_requires_token:
        return dedent(
            f"""
            This deployment requires a bearer token. Obtain it through an approved
            secret channel and expose it to Codex without putting it in a config file
            or command argument:

            ```bash
            export NAUTILUS_STUDIO_MCP_TOKEN='<provided-by-operator>'

            # Codex
            codex mcp add nautilus-studio \\
              --url {base}/mcp/ \\
              --bearer-token-env-var NAUTILUS_STUDIO_MCP_TOKEN
            codex mcp get nautilus-studio

            # Claude Code (single quotes preserve runtime environment expansion)
            claude mcp add --transport http nautilus-studio {base}/mcp/ \\
              --header 'Authorization: Bearer ${{NAUTILUS_STUDIO_MCP_TOKEN}}'
            claude mcp get nautilus-studio
            ```
            """
        ).strip()
    return dedent(
        f"""
        This trusted-network deployment does not require an MCP bearer token:

        ```bash
        # Codex
        codex mcp add nautilus-studio --url {base}/mcp/
        codex mcp get nautilus-studio

        # Claude Code
        claude mcp add --transport http nautilus-studio {base}/mcp/
        claude mcp get nautilus-studio
        ```

        If the operator later enables `STUDIO_MCP_TOKEN`, remove and re-add this MCP
        server using `--bearer-token-env-var`; never put the token in a command
        argument or config file.
        """
    ).strip()


def render_llms_txt(
    base_url: str,
    *,
    mcp_enabled: bool = True,
    mcp_requires_token: bool = False,
) -> str:
    """Render the agent-facing discovery document for this Studio instance."""

    base = base_url.rstrip("/")
    mcp_connection = indent(
        _agent_connection(
            base,
            mcp_enabled=mcp_enabled,
            mcp_requires_token=mcp_requires_token,
        ),
        "        ",
    ).removeprefix("        ")
    mcp_discovery = f"{base}/mcp/" if mcp_enabled else "disabled in this deployment"
    return dedent(
        f"""
        # Nautilus Studio

        > Nautilus Studio is a creator-first, agentic AI film workshop for planning,
        > editing, and rendering long-form videos from stories and creator assets.

        This document is the machine-readable entry point for the Studio instance at
        {base}/. Codex, Claude Code, and other agents should prefer the in-process MCP
        server for project operations and use the REST API only for capabilities not
        exposed through MCP, such as uploading files, editing individual shots, or
        downloading rendered media.

        ## Discovery

        - Creator UI: {base}/
        - Streamable HTTP MCP: {mcp_discovery}
        - OpenAPI schema: {base}/openapi.json
        - API documentation: {base}/docs
        - Studio health: {base}/api/health
        - Model service and GPU status: {base}/api/services/status
        - This document: {base}/llms.txt

        Service availability is dynamic. Before planning or rendering, call the MCP
        tool `studio_status` or GET `/api/services/status`. Do not infer that FL2VA,
        Ref2VA, image generation, or image editing is ready merely because the Studio
        UI or MCP endpoint is reachable.

        ## Connect an agent

        The MCP transport is Streamable HTTP and the trailing slash in `/mcp/` is
        required.

        {mcp_connection}

        Other MCP clients should use the Streamable HTTP transport at `{base}/mcp/`.
        When bearer authentication is enabled, send
        `Authorization: Bearer <operator-provided-token>` on every MCP request.

        A bearer token on `/mcp/` does not protect the UI or REST API. Deploy the
        complete Studio service only on a network appropriate for its projects and
        creator assets.

        ## MCP tools

        - `studio_status()`: inspect planner/model health, GPU telemetry, and active
          planning or render jobs.
        - `studio_list_projects()`: list projects and their latest render state.
        - `studio_get_project(project_id)`: read a complete project and storyboard.
        - `studio_list_assets()`: list creator assets with roles and tags.
        - `studio_import_asset(path, tags?, roles?)`: import a server-local path from
          the operator-configured allowed roots. It cannot read arbitrary paths. Valid
          roles are `character`, `location`, `prop`, `style`, `start_frame`, `audio`,
          and `reference`.
        - `studio_plan_project(prompt, title?, duration_seconds?, continuation_mode?,
          reference_asset_ids?, ultra_fast_anchor_strategy?,
          ultra_fast_transition?)`: create a project and start asynchronous storyboard
          planning. It returns a project id immediately. Defaults are 30 seconds,
          `quality`, `independent`, and `fade_black`; project duration must be 15–900
          seconds. Valid transitions are `fade_black`, `dissolve`, `hard_cut`, and
          `random`.
        - `studio_render_project(project_id, force?)`: start a render after all required
          model endpoints are ready.
        - `studio_render_status(project_id)`: inspect the latest render job.
        - `studio_cancel_planning(project_id)`: cancel active storyboard planning.

        MCP resources:

        - `studio://projects`
        - `studio://project/{{project_id}}`

        MCP prompt:

        - `creator_workflow`: the recommended asset, planning, review, render, and
          polling sequence.

        ## Recommended agent workflow

        1. Call `studio_status` and stop with an actionable explanation if a required
           provider is unconfigured or unhealthy.
        2. Call `studio_list_assets`. Use only asset ids explicitly selected by the
           user; never infer selection from everything in the library.
        3. Call `studio_plan_project`. Valid continuation modes are:
           - `ultra_fast`: FL2VA for every shot, with generated/edited shot anchors.
           - `fast`: Ref2VA using the previous clip's final five seconds.
           - `quality`: Ref2VA using the complete previous clip.
        4. Poll `studio_get_project(project_id)` until planning completes, then present
           the storyboard for review. Do not start costly GPU rendering merely because
           planning completed.
        5. After the user approves the storyboard, call `studio_render_project` and
           poll `studio_render_status` at a reasonable interval. Avoid duplicate render
           submissions unless the user explicitly requests `force=true`.
        6. When the job is complete, use `GET /api/jobs/{{job_id}}/output` to stream or
           download the final video.

        ## Model service topology

        - Planner: generates the world bible, shot drafts, detailed H3 prompts, and
          continuity review. It may use an OpenAI-compatible or Responses-compatible
          provider, with a deterministic heuristic fallback for local development.
        - Qwen Image T2I: creates a fresh opening anchor when no image material is
          explicitly selected.
        - Qwen Image Edit: composes selected character/location/style assets and edits
          continuation anchors.
        - MiniMax-H3 FL2VA: generates a video from an opening frame and shot prompt.
        - MiniMax-H3 Ref2VA: extends prior video context in `fast` or `quality` mode.
        - FFmpeg: assembles shots, transitions, audio, and optional sidecar subtitles.

        The Studio orchestrates these providers, but model servers are separate
        deployments. Their current endpoints, readiness, and GPU utilization are
        intentionally reported by `studio_status` rather than embedded here.

        ## REST API

        Prefer the MCP tools for agent-driven project lifecycle operations. Use REST
        when MCP does not expose the needed media or fine-grained editing operation.
        The canonical request/response schemas are always available from
        `{base}/openapi.json`.

        Common REST operations:

        - `GET /api/assets`: list assets.
        - `POST /api/assets/upload`: multipart upload with `files`, `tags`, and `roles`.
        - `PATCH /api/assets/{{asset_id}}`: edit asset name, caption, tags, or roles.
        - `GET /api/assets/{{asset_id}}/content`: read asset media.
        - `POST /api/projects/plan-async`: start storyboard planning.
        - `GET /api/projects/{{project_id}}`: poll or read a project.
        - `GET /api/projects/{{project_id}}/planner-trace`: inspect planner stages and
          raw diagnostic events when planning fails.
        - `PATCH /api/projects/{{project_id}}`: edit the brief or world bible.
        - `PATCH /api/projects/{{project_id}}/shots/{{shot_id}}`: edit one shot.
        - `POST /api/projects/{{project_id}}/compile`: inspect the execution plan and
          required model capabilities before rendering.
        - `POST /api/projects/{{project_id}}/render`: submit a render.
        - `GET /api/projects/{{project_id}}/jobs/latest`: inspect its latest job.
        - `GET /api/jobs/{{job_id}}/output`: stream the finished video.

        ## Operational rules for agents

        - Treat planning, importing, editing, deleting, and rendering as state-changing
          operations. State the intended action before invoking them.
        - Rendering can reserve substantial GPU time. Require an approved storyboard
          and healthy required providers before submission.
        - Never fabricate project, shot, asset, or job ids. Discover them through MCP
          or REST.
        - Never put provider keys or MCP bearer tokens into project prompts, assets,
          logs, or repository files.
        - Do not retry a failed render blindly. Read `studio_render_status`, service
          status, and the planner trace or job error first.

        ## Source and support

        - Source: https://github.com/yeahdongcn/nautilus-studio
        - Issues: https://github.com/yeahdongcn/nautilus-studio/issues
        """
    ).lstrip()
