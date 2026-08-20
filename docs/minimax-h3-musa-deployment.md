# MiniMax-H3 on MUSA

This guide records the vLLM-Omni deployment used by Nautilus Studio. Image
provenance and performance evidence are listed separately: changing an image
does not transfer timing claims from an older benchmark. Do not copy CUDA/H100
parallelism flags into a MUSA deployment without re-running the matrix below.

## Current Studio test image

- image: operator-supplied `VLLM_OMNI_MUSA_IMAGE`, current internal tag suffix
  `minimax-h3-20260819`
- digest: `sha256:ccf07399174fceac3f0e1c56e26c42046a4244c4bae05f9f61875120d3e47a46`
- vLLM-Omni revision: `6ae03f5253f74bfe1a040b9d8c35ecb78c32b2ea`
- vLLM-Omni version: `0.1.dev28+g6ae03f525.musa`
- PyTorch: `2.11.0.post1+musa5.2.0`
- torchada: `0.1.82`

The pushed digest passed compiled TP8 FL2VA and Ref2VA request-level smokes on
MTT S5000. It is the preferred common runtime for subsequent MiniMax-H3, Qwen
Image, and Qwen Image Edit Studio tests. Model-specific requests must still be
smoked after deployment; the final-tag acceptance run did not re-run the Qwen
Image or Qwen Image Edit matrices.

The registry image may require Moore Threads network credentials. External
deployments can substitute an equivalent image built from the recorded
vLLM-Omni revision and dependency stack. Set the reference through the
environment instead of committing private registry topology:

```bash
export VLLM_OMNI_MUSA_IMAGE='<registry>/<namespace>/vllm-omni:minimax-h3-20260819@sha256:ccf07399174fceac3f0e1c56e26c42046a4244c4bae05f9f61875120d3e47a46'
```

The included container wrapper fails closed unless the digest-qualified image,
leased devices, owner, ticket, lease handle, and task-owned container name are
explicit:

```bash
VLLM_OMNI_MUSA_IMAGE="$VLLM_OMNI_MUSA_IMAGE" \
LEASED_GPU_IDS=0,1,2,3 \
CONTAINER_NAME='hlease-<owner>-<task>-<node>-gpu0_1_2_3' \
CONTAINER_OWNER='<lease-owner>' \
CONTAINER_TICKET='<task-id>' \
LEASE_HANDLE='<lease-handle>' \
scripts/run-musa-vllm-omni-service.sh \
  vllm serve /home/dist/models/MiniMax/MiniMax-H3/Ref2VA \
    --omni --host 0.0.0.0 --port 8092 --trust-remote-code \
    --num-gpus 4 --tensor-parallel-size 4 \
    --text-encoder-tp-size 4 \
    --vae-patch-parallel-size 4 --vae-parallel-mode tile \
    --vae-use-tiling --diffusion-attention-backend FLASH_ATTN
```

The wrapper deliberately uses a non-privileged `mthreads` container and passes
only `MTHREADS_VISIBLE_DEVICES=<physical leased ids>`. The runtime remaps those
devices to `musa:0..N-1` inside the container. Do not add `--privileged`,
`MUSA_VISIBLE_DEVICES`, or `CUDA_VISIBLE_DEVICES` to a shared-node functional
run; privileged containers can see the whole host and are reserved for an
isolated whole-node profiling lease.

The wrapper also rejects mutable tag-only references. Keeping the tag next to
`@sha256:<digest>` is useful for humans, while the digest prevents a local cache
or later tag update from being mistaken for the image documented here.

## Historical TP4 profile environment

The TP4 parameter and performance matrix below was collected with the
`minimax-h3-20260815` image. Preserve this provenance when comparing a newer
image:

- digest: `sha256:23ae27867cd19ce848a27688dba262d0569c67ff3c3b599cc2a429f8ab184a8b`
- vLLM-Omni revision: `45c33a4a776450a7ba7875992d417757364fef47`
- accelerator: 4 x MTT S5000, 80 GiB each
- driver: `3.3.5-server`
- PyTorch: `2.11.0.post1+musa5.2.0`
- torchada: `0.1.79`

The primary Ref2VA benchmark uses the same first clip, prompt, references, and
seed for every profile: `1280x704`, 10 seconds, 24 FPS, 50 inference steps, and
the original BF16 checkpoint. It does not use quantization, Cache-DiT, fewer
steps, or a lower resolution.

### 20260815 recommended Ref2VA profile

For Nautilus Studio clips up to the H3 nominal 15-second output limit at `1280x704`, the best validated
four-GPU profile is TP4/TE4/VAE-PP4 with MUSA FlashAttention and no CPU
offload:

```bash
vllm serve /home/dist/models/MiniMax/MiniMax-H3/Ref2VA \
  --omni \
  --host 0.0.0.0 \
  --port 8092 \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

The full 10-second, 50-step request completed in `1709.6s` (about 28m30s).
A 14-second, two-step memory gate also completed successfully in `131.8s`.
That 20260815 serving-parameter sweep did not remove the roughly 30-minute DiT
bottleneck, but this profile was the fastest stable four-GPU configuration
found in that sweep.

### Compilation and graph capture

This image enables regional `torch.compile` by default; no additional serving
flag is required. Confirm the active path from the startup log:

```text
Regional compilation applied to 52 module(s) for repeated blocks ['MiniMaxH3DiTBlock'].
Model runner: transformer compiled with torch.compile.
```

On a warmed two-step request the default compiled path took `83.564s`, while
`--enforce-eager` took `85.464s` (`2.27%` slower). Keep the compiled path for
serving and use eager only for diagnosis. The first request includes lazy
compilation, so compare steady-state requests with the same resolution, frame
count, and inference-step count.

The exact `20260815` image does not expose the newer
`--diffusion-compile-*` CLI options, and its MiniMax-H3 diffusion runner has no
usable CUDA/MUSA Graph capture path. vLLM language-model CUDA Graph flags do
not apply to the H3 denoise loop.

### Memory-safe fallback

Add `--enable-cpu-offload` when validating a larger canvas, longer future model
limit, or a lower-memory S5000. On the benchmark above, the steady-state
offload profile took about `1710-1714s`, effectively tied with no-offload.
Offload is therefore a memory policy, not a speed optimization.

### Torch SDPA diagnostic fallback

Plain `TORCH_SDPA` OOMed on the two-step gate while requesting another
`26.39GiB`; adding CPU offload alone still OOMed. It can run when CPU offload
is combined with the MUSA allocator's expandable segments:

```bash
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True

vllm serve /home/dist/models/MiniMax/MiniMax-H3/Ref2VA \
  --omni --host 0.0.0.0 --port 8092 --trust-remote-code \
  --num-gpus 4 --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile \
  --enable-cpu-offload --vae-use-tiling \
  --diffusion-attention-backend TORCH_SDPA
```

The warmed two-step request then took `100.625s`, `20.42%` slower than the
`83.564s` FlashAttention baseline. Treat this as a compatibility or diagnostic
fallback, not the production profile.

## Request validation and teardown

The following example exposes the recommended Ref2VA service on port `8092`.
Adjust the model mount and leased physical devices for the target host.

```bash
LEASED_GPU_IDS=0,1,2,3 \
CONTAINER_NAME='hlease-<owner>-<task>-<node>-gpu0_1_2_3' \
CONTAINER_OWNER='<lease-owner>' \
CONTAINER_TICKET='<task-id>' \
LEASE_HANDLE='<lease-handle>' \
scripts/run-musa-vllm-omni-service.sh \
  vllm serve /home/dist/models/MiniMax/MiniMax-H3/Ref2VA \
    --omni --host 0.0.0.0 --port 8092 --trust-remote-code \
    --num-gpus 4 --tensor-parallel-size 4 \
    --text-encoder-tp-size 4 \
    --vae-patch-parallel-size 4 --vae-parallel-mode tile \
    --vae-use-tiling --diffusion-attention-backend FLASH_ATTN
```

Verify both health and a real request. Server startup alone is not a pass:

```bash
curl --fail http://127.0.0.1:8092/health
```

Use `/v1/videos` plus polling for 50-step requests. The synchronous endpoint
has a server-side timeout and returned HTTP 504 for a request longer than ten
minutes even though the diffusion worker was otherwise healthy.

Before releasing the lease, stop the exact owner-labelled container. Because
the launch wrapper uses `--rm`, a successful stop also removes the container:

```bash
CONTAINER_NAME='hlease-<owner>-<task>-<node>-gpu0_1_2_3' \
CONTAINER_OWNER='<lease-owner>' \
CONTAINER_TICKET='<task-id>' \
LEASE_HANDLE='<lease-handle>' \
scripts/stop-musa-vllm-omni-service.sh
```

The stop helper checks all three ownership labels and refuses to stop a
different lease's container. Verify the task's containers are gone before
releasing the lease through the operator's lease controller.

## Other model services

FL2VA is a separate model partition. The memory-safe four-GPU Studio profile
keeps the same TP/TE/VAE topology and adds CPU offload:

```bash
vllm serve /home/dist/models/MiniMax/MiniMax-H3/FL2VA \
  --omni --host 0.0.0.0 --port 8091 --trust-remote-code \
  --num-gpus 4 --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile \
  --enable-cpu-offload --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

Qwen Image Edit runs independently. This one-GPU command is the previously
validated S5000 topology; re-run its request smoke on a new image digest. Keep
the served name identical to Studio's configured model:

```bash
LEASED_GPU_IDS=4 \
CONTAINER_NAME='hlease-<owner>-<task>-<node>-gpu4' \
CONTAINER_OWNER='<lease-owner>' \
CONTAINER_TICKET='<task-id>' \
LEASE_HANDLE='<lease-handle>' \
scripts/run-musa-vllm-omni-service.sh \
  vllm serve /home/dist/models/Qwen/Qwen-Image-Edit-2511 \
    --omni --host 0.0.0.0 --port 8093 --trust-remote-code \
    --served-model-name Qwen/Qwen-Image-Edit-2511 \
    --num-gpus 1 --tensor-parallel-size 1 \
    --limit-mm-per-prompt '{"image":4}' \
    --vae-patch-parallel-size 1
```

Qwen Image provides the zero-material T2I route. Its served name must likewise
match `STUDIO_T2I_MODEL` when that field is set:

```bash
LEASED_GPU_IDS=5 \
CONTAINER_NAME='hlease-<owner>-<task>-<node>-gpu5' \
CONTAINER_OWNER='<lease-owner>' \
CONTAINER_TICKET='<task-id>' \
LEASE_HANDLE='<lease-handle>' \
scripts/run-musa-vllm-omni-service.sh \
  vllm serve /home/dist/models/Qwen/Qwen-Image-2512 \
    --omni --host 0.0.0.0 --port 8094 --trust-remote-code \
    --served-model-name Qwen/Qwen-Image-2512 \
    --num-gpus 1 --tensor-parallel-size 1 \
    --vae-patch-parallel-size 1
```

Connect the four service types to Studio with:

```bash
export STUDIO_H3_FL2VA_URL=http://127.0.0.1:8091
export STUDIO_H3_REF2VA_URL=http://127.0.0.1:8092
export STUDIO_IMAGE_EDIT_PROVIDER=vllm-omni
export STUDIO_IMAGE_EDIT_BASE_URL=http://127.0.0.1:8093
export STUDIO_IMAGE_EDIT_MODEL=Qwen/Qwen-Image-Edit-2511
export STUDIO_T2I_PROVIDER=vllm-omni
export STUDIO_T2I_BASE_URL=http://127.0.0.1:8094
export STUDIO_T2I_MODEL=Qwen/Qwen-Image-2512
```

## Historical 20260815 parameter matrix

| Candidate | Result | Decision |
| --- | --- | --- |
| TP4 / TE4 / VAE-PP4 / FlashAttention / no offload | 50-step: `1709.6s`; warm two-step: `83.564s` | Recommended |
| Same profile with CPU offload | steady 50-step: about `1710-1714s` | Memory-safe fallback |
| Same profile with `--enforce-eager` | warm two-step: `85.464s` (`2.27%` slower) | Keep compiled default |
| VAE-PP1 | warm two-step: `154.0s` | Reject: much slower |
| VAE-PP2 | service initialization fails | Unsupported by H3 native VAE |
| TE1 | real Ref2VA request OOMs on rank 0 | Reject |
| TE2 | text-encoder process-group initialization fails | Unsupported in this TP4 layout |
| TORCH_SDPA | two-step request tries to allocate another `26.39GiB` and OOMs | Reject |
| TORCH_SDPA / CPU offload / expandable segments | warm two-step: `100.625s` (`20.42%` slower) | Diagnostic fallback only |
| USP4 / HSDP4 / no offload | two-step: `146.7s`; 50-step exceeded 35 minutes | Reject on current MUSA/MCCL stack |
| USP4 / HSDP4 / CPU offload | first forward fails on cross-device parameter storage | Unsupported combination |
| TP8 / TE8 | no eligible contiguous eight-GPU development host | Not run |

## Historical NVIDIA CI cross-hardware reference

The Ref2VA V2V job in
[vLLM-Omni Buildkite build 13601](https://buildkite.com/vllm/vllm-omni/builds/13601/canvas?sid=019fff05-1b91-4a96-a5c4-25a96536c9ea&tab=output)
uses 4 x H100 with Ulysses sequence parallelism and HSDP rather than tensor
parallelism. Its relevant serving topology is:

```text
--task-type ref2va
--num-gpus 4 --usp 4 --ring 1
--use-hsdp --hsdp-shard-size 4 --hsdp-replicate-size 1
--text-encoder-tp-size 4
--vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling
--diffusion-attention-backend FLASH_ATTN
--enable-diffusion-pipeline-profiler
```

The CI request is `1344x768`, 209 frames at 24 FPS (`8.7s`), 8 inference
steps, seed 42, one two-step warmup, and three sequential measured requests.
The S5000 replay used the same benchmark command and request configuration with
the recommended TP4/TE4/VAE-PP4 MUSA profile; only the profiler flag was added
to that profile.

| Metric | 4 x H100 CI | 4 x S5000 | S5000 / H100 |
| --- | ---: | ---: | ---: |
| Client latency mean | `112.7249s` | `194.0041s` | `1.721x` |
| Stage-0 generation | `106.5792s` | `189.9449s` | `1.782x` |
| Prepare reference video | `0.4621s` | `0.5435s` | `1.176x` |
| Text encoding | `1.1934s` | `1.8719s` | `1.569x` |
| Video-condition encoding | `9.6146s` | `17.0400s` | `1.772x` |
| DiT diffusion | `88.6343s` | `162.4423s` | `1.833x` |
| VAE decoding | `3.3939s` | `6.9195s` | `2.039x` |
| Peak memory | `61312MiB` | `59930MiB` | `0.977x` |

The current S5000 stack therefore delivers `58.1%` of the H100 request
throughput for this exact workload. DiT diffusion contributes `73.81s`, or
`88.5%`, of the `83.37s` stage-0 gap. VAE decoding has the largest relative
ratio but only adds `3.53s`; optimization work should prioritize the denoise
path and its kernels/collectives before the text encoder or VAE.

This is a comparison of two validated serving stacks, not an isolated hardware
microbenchmark: the H100 job uses USP4/HSDP4 on a different CI revision, while the
MUSA image uses TP4 on revision `45c33a4a776450a7ba7875992d417757364fef47`.
Copying the H100 topology is not an optimization on the current MUSA stack:
USP4/HSDP4 took `146.7s` on the warmed two-step gate versus `83.564s` for TP4.

## Notes and non-options

- H3's native VAE accepts patch parallel size `1` or the complete DiT group;
  partial groups such as VAE-PP2 with TP4 fail validation.
- H3's wrapper does not consume `--vae-use-slicing`; do not treat that flag as
  a performance lever. Tiling comes from the checkpoint VAE configuration and
  the full VAE patch-parallel group.
- In the NVIDIA recipe, `--ring 1` makes the Ring factor one; the sequence
  parallel work comes from `--usp 4` (Ulysses) while HSDP shards the model.
- Cache-DiT is not part of this profile. In this image its dependency detects
  MUSA as `CpuPlatform`, and cached quality modes intentionally skip model
  evaluations, so it is neither a proven MUSA optimization nor quality-neutral.
- Do not reduce inference steps, change flow shift, or lower the resolution when
  comparing serving profiles. Those alter the workload rather than optimize it.
