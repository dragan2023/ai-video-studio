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

## Continuity policy

An explicit creator start frame always wins. Without one, the runtime may
compose an anchor from project context and ordered references. Continuous
shots use Ref2VA when configured: `fast` passes the previous clip's final five
seconds, while `quality` passes the complete previous clip. The runtime adds a
non-persistent anti-replay instruction only to these continuation requests, so
storyboard prompts remain editable and retries cannot accumulate duplicate
constraints. When Ref2VA is unavailable, the previous boundary frame and
FL2VA remain an internal compatibility fallback. Real scene cuts can request
a new anchor.

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

## Security boundary

The built-in API is a single-user development service. Authentication,
authorization, tenant isolation, rate limiting, TLS, and content policy belong
in a trusted reverse proxy or hosting layer. Provider credentials are read from
environment variables or local operator configuration and must never be stored
with project media.
