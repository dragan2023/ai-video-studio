# Image-edit providers

Nautilus Studio treats image editing as an optional anchor-frame stage between
story planning and video generation. The planner chooses ordered reference
assets; a provider turns them into one scene-complete frame; FL2VA then uses
that frame as its visual anchor.

```text
scene + characters + props + shot prompt
                  |
          reference manifest
                  |
        ImageEditProvider.edit()
                  |
          generated anchor.png
                  |
              FL2VA clip
```

## Why this is provider-neutral

Many creators will use a hosted image-edit API because they do not have enough
GPU memory to deploy a diffusion model. Self-hosted vLLM-Omni remains a first-
class feature for teams that need local data, predictable cost, and backend
control. Both paths implement the same `ImageEditProvider` protocol.

Supported provider keys:

- `disabled`: preserve the current direct-to-video fallback;
- `vllm-omni`: a local or remote vLLM-Omni multimodal chat endpoint;
- `openai-compatible`: a hosted endpoint accepting ordered `image_url` content.

Vendor-native APIs can add a small adapter without modifying the planner or
runner.

## Reference manifest

Image APIs do not understand Nautilus asset roles by themselves. The studio
therefore compiles metadata into a deterministic, numbered prompt:

```text
REFERENCE MANIFEST (use the images in this exact order):
[1] label=old town; role=location; tags=street, night; caption=wet market
[2] label=hero; role=character; tags=red coat; caption=lead character
[3] label=companion; role=character; tags=blue scarf; caption=second lead
```

The images follow this text in exactly the same order. Provider adapters must
preserve that order.

## Opening-frame precedence

Nautilus treats an explicit creator start frame as immutable input:

1. if a shot has `start_frame_asset_id`, FL2VA receives that image directly;
2. with one or more explicitly selected image references, Image Edit composes
   an anchor from the project bible, shot prompt, and ordered images;
3. with zero selected images, the separate T2I provider creates the anchor from
   the planner-authored prompt;
4. when the provider required by the selected route is unavailable, rendering
   fails before a job is created with the exact missing endpoint named.

The planner-authored `anchor_prompt` is retained in the storyboard even when
Image Edit is disabled. This makes the intended opening composition inspectable
and editable before services are brought online; a reference image with only a
character, location, prop, or style role is not treated as an explicit start
frame.

The default `scene-cuts` mode preserves the previous shot's boundary for
continuous clips and composes a new anchor only for the opening shot or a real
scene cut. This avoids overwriting creator intent and duplicating motion across
adjacent clips.

## Visible preproduction approval

Before a project renders, the Preproduction Planner presents every shot's source
script evidence, transition decision, opening-frame source and asset gap in the
creator UI. User material and explicit start frames remain immutable. Black or
subtitle shots use a deterministic system black frame; only an approved,
independent shot with no usable image reference may use the configured T2I
provider. A matching cut, sound bridge, brightness match or occlusion transition
is a visual hard cut and therefore never reuses the previous boundary frame.

A GPT Image 2-compatible provider uses the existing `openai-compatible` T2I
adapter. Configure its base URL, model and API key only in local `.env`; never
store a key in a project, plan response or client state. The UI must show the
planned image count and receive one explicit approval before image generation.

## Qwen model boundary

The original Qwen-Image-Edit checkpoint uses the single-image
`QwenImageEditPipeline`. Multi-image compositing requires a Plus pipeline such
as Qwen-Image-Edit-2509 or Qwen-Image-Edit-2511
(`QwenImageEditPlusPipeline`). A capability check must reject multi-image
requests when the configured checkpoint is single-image.

## Configuration

```bash
export STUDIO_IMAGE_EDIT_PROVIDER=vllm-omni
export STUDIO_IMAGE_EDIT_BASE_URL=http://127.0.0.1:8093
export STUDIO_IMAGE_EDIT_MODEL=Qwen/Qwen-Image-Edit-2511
export STUDIO_IMAGE_EDIT_MAX_REFERENCES=4
export STUDIO_IMAGE_EDIT_TRUE_CFG_SCALE=4.0
export STUDIO_IMAGE_EDIT_GUIDANCE_SCALE=1.0
export STUDIO_IMAGE_EDIT_ANCHOR_MODE=scene-cuts
```

The vLLM-Omni server must also allow the same number of image inputs. The
included launcher defaults `MAX_REFERENCE_IMAGES=4` and passes it through
`--limit-mm-per-prompt`; without that flag vLLM accepts the request but keeps
only its default per-prompt image allowance.

The planner writes the complete direct-to-model anchor prompt, including
ordered reference names, roles, and visual contributions. The shot editor keeps
it editable with a 1000-character cap. When the tokenizer path is configured,
the adapter also rejects prompts above 1000 exact Qwen text tokens before any
generation request; prompts are never silently truncated. The tokenizer path is
optional: for a remote vLLM-Omni server, the tokenizer is already mounted in
the serving container and should not be pointed at that container-only path on
the Studio host. If the configured host path does not contain `tokenizer.json`,
the exact local check is skipped while the 1000-character cap remains active;
the provider then performs its own token validation.

For hosted APIs, use `openai-compatible` and provide
`STUDIO_IMAGE_EDIT_API_KEY` through the environment. Never place provider keys
inside project files or persisted story data.

## Acceptance matrix

Before enabling a provider in production, validate at least:

1. one location plus one character;
2. one location plus two distinct characters;
3. one location, multiple characters, and a prop;
4. identity and wardrobe retention;
5. target aspect ratio without stretching;
6. no unintended text, logo, or watermark;
7. deterministic error handling when the provider exceeds its image limit.

The probe helper writes a key-free receipt with input hashes, generation
settings, output dimensions, and output hash:

```bash
python scripts/probe-image-edit.py \
  --base-url http://127.0.0.1:8093 \
  --model Qwen/Qwen-Image-Edit-2511 \
  --reference location=/path/to/scene.png \
  --reference character=/path/to/hero.png \
  --reference character=/path/to/companion.png \
  --reference prop=/path/to/prop.png \
  --prompt "Compose both characters and the prop naturally in the scene" \
  --negative-prompt "text, logo, watermark" \
  --output /tmp/nautilus-image-edit/scene-two-characters-prop.png \
  --receipt /tmp/nautilus-image-edit/scene-two-characters-prop.json
```

The `--model` value must match the server's `--served-model-name`. The included
launcher defaults both to `Qwen/Qwen-Image-Edit-2511`; set both explicitly when
serving a different checkpoint or alias.

The receipt intentionally excludes the endpoint and API key. Visual identity
and composition still require human review; hashes and dimensions are not a
correctness verdict.
