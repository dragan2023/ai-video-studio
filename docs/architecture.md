# Architecture

Nautilus Studio is a creator-facing control plane. Model topology remains
behind a small set of provider contracts; creators work with stories,
materials, shots, and finished videos rather than node graphs.

```text
brief + material library
          |
          v
 creative director -> shot directors -> continuity critic
          |
          v
 project bible + editable shots
          |
          +--> explicit start frame --------------------+
          |                                             |
          +--> ordered references --> image edit anchor |
                                                        v
                                             FL2VA / Ref2VA provider
                                                        |
                                                        v
                                          boundary extraction + assembly
                                                        |
                                                        v
                                             preview + sidecar metadata
```

## State

SQLite stores assets, projects, shots, and render jobs. Original material and
generated media live under the configured data directory. Provider jobs are
submitted through durable vLLM-Omni video endpoints and polled by the runner;
the local job records creator-visible progress and output paths.

The Python service only serves a pre-built React/Vite workspace configured by
`STUDIO_WEB_ROOT`. There is no second legacy UI implementation; a missing or
incomplete bundle fails startup so frontend and API behavior cannot silently
drift apart.

## Provider boundaries

- Planner: Responses-compatible structured generation with a deterministic
  fallback.
- Image edit: ordered multimodal chat request returning an image. Local
  vLLM-Omni and hosted OpenAI-compatible endpoints share this contract.
- Video: MiniMax-H3 FL2VA and Ref2VA adapters. The interface keeps room for
  other vLLM-Omni, SGLang, or hosted video backends.
- Media: local ffmpeg/ffprobe assembly and Pillow-based no-stretch image fit.

## Observability boundary

`GET /api/services/status` combines three independent signals without turning
one into another:

- provider health from each configured service's HTTP `/health` endpoint;
- optional vLLM-Omni running/waiting request counters from `/metrics`;
- optional GPU utilization from a versioned, atomically replaced local JSON
  snapshot.

The model-serving process and the GPU collector are separate operator-owned
services. Nautilus never invokes SSH, a vendor GPU utility, a lease broker, or
process-control commands from an API request. This keeps the public control
plane portable across MUSA, CUDA, other local runtimes, and hosted vendors.
Missing or stale GPU telemetry is reported explicitly and never converted to a
false `0%` reading. GPU samples are diagnostic only and cannot authorize a
lease, select devices, or terminate a process.

## Continuity policy

An explicit creator start frame always wins. Without one, the runtime may
compose an anchor from project context and ordered references. `ultra_fast`
defaults to storyboard anchors: shot 1 uses selected materials through Image
Edit or falls back to T2I, while shot 2 and later edit the previous final frame
into a new composition and append selected creator references after it. Every
shot then runs through FL2VA and the media layer adds a fixed or seeded-random
transition. Its legacy `boundary` strategy still sends the previous final
frame directly to FL2VA. `fast` sends the previous clip's final five seconds
to Ref2VA; `quality` sends the complete previous clip. The two Ref2VA modes add
a non-persistent anti-replay instruction only to those continuation requests.

The H3 adapter sends `quality=lossless` explicitly by default. This is separate
from Nautilus's continuation choice: the adapter's quality field selects the
model's native versus Cache-DiT conditioning path, while `ultra_fast`/`fast`/
`quality` select how the previous shot is supplied. `flow_shift=12`,
`audio_flow_shift=3`, and 50 diffusion steps are the validated accuracy
baseline. The model's internal reference-noise timestep remains fixed until a
dedicated artifact/continuity sweep justifies exposing it.

Every planned H3 shot is limited to 14 seconds. This intentionally leaves
container-timestamp headroom below H3's hard 15-second reference-video limit:
a nominal 15-second encode can be reported by ffprobe as slightly longer than
15 seconds and would otherwise make the following Ref2VA request fail.

## Hierarchical storyboard planning

With an OpenAI-compatible planner configured, the default pipeline makes
separate calls with separate responsibilities:

1. The creative director creates the World Bible and a compact shot spine
   (active subjects, landmarks, opening/ending states, audio phase, and an
   explicit transition kind).
2. One shot director expands each spine row into an editable `ShotSpec` with
   a dense English H3 timeline. The first shot and continuation shots receive
   different system instructions; continuation shots hold the inherited
   boundary state before introducing a new action.
3. A continuity critic reviews adjacent pairs and returns the corrected shot
   set. It checks identity, wardrobe, eyelines, landmarks, props, lighting,
   motion direction, action replay, and dialogue/audio separation.

The compiler preserves structured detail instead of slicing the main prompt to
a small word count. H3's 350--500 words is a useful generation range, not a
reason to pad prose; `audit_context_ir` reports unusually short or verbose
descriptions. Set `STUDIO_PLANNER_PIPELINE=single_pass` only for a provider
that cannot support the multi-call path. The optional `STUDIO_H3_SKILLS_DIR`
points at downloaded MiniMax-H3 skill packs; Nautilus selects the matching
style pack and sends only its shot/continuity/audio/QC excerpt, never all
workflow instructions at once.

Planner diagnostics are persisted as bounded events on the project and exposed
through `/api/projects/{project_id}/planner-trace`. The Studio Debug Console
combines these server events with browser-side click/request events, so a
missing request, provider error, malformed JSON response, or continuity-critic
failure can be distinguished without retrying blindly. Normal project/list
responses omit the trace payload; the dedicated endpoint returns it on demand.

## Security boundary

The built-in API is a single-user development service. Authentication,
authorization, tenant isolation, rate limiting, TLS, and content policy belong
in a trusted reverse proxy or hosting layer. Provider credentials are read from
environment variables or local operator configuration and must never be stored
with project media.
