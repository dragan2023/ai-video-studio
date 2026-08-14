# MiniMax-H3 on MUSA

This guide records the validated vLLM-Omni deployment used by Nautilus Studio.
It is intentionally tied to an exact image and workload; do not copy CUDA/H100
parallelism flags into a MUSA deployment without re-running the matrix below.

## Validated environment

- image: `registry.mthreads.com/mcconline/inference/vllm-omni:minimax-h3-20260815`
- digest: `sha256:23ae27867cd19ce848a27688dba262d0569c67ff3c3b599cc2a429f8ab184a8b`
- vLLM-Omni revision: `45c33a4a776450a7ba7875992d417757364fef47`
- accelerator: 4 x MTT S5000, 80 GiB each
- driver: `3.3.5-server`
- PyTorch: `2.11.0.post1+musa5.2.0`
- torchada: `0.1.79`

The registry image may require Moore Threads network credentials. External
deployments can substitute an equivalent image built from the recorded
vLLM-Omni revision and dependency stack; re-run the smoke and performance gates
before reusing the timing claims.

The primary Ref2VA benchmark uses the same first clip, prompt, references, and
seed for every profile: `1280x704`, 10 seconds, 24 FPS, 50 inference steps, and
the original BF16 checkpoint. It does not use quantization, Cache-DiT, fewer
steps, or a lower resolution.

## Recommended Ref2VA profile

For Nautilus Studio clips up to 14 seconds at `1280x704`, the best validated
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
The current serving-parameter space therefore does not remove the roughly
30-minute DiT bottleneck, but this profile is the fastest stable four-GPU
configuration found in the sweep.

Do not add `--enforce-eager`. On a warmed two-step request the default compiled
path took `83.6s`, while eager took `85.5s`.

### Memory-safe fallback

Add `--enable-cpu-offload` when validating a larger canvas, longer future model
limit, or a lower-memory S5000. On the benchmark above, the steady-state
offload profile took about `1710-1714s`, effectively tied with no-offload.
Offload is therefore a memory policy, not a speed optimization.

## Container example

The following example exposes the recommended Ref2VA service on port `8092`.
Adjust the model mount and visible devices for the target host.

```bash
docker run -d \
  --name nautilus-h3-ref2va \
  --runtime=mthreads \
  --privileged \
  --network host \
  --ipc host \
  --shm-size 1g \
  -e MTHREADS_VISIBLE_DEVICES=0,1,2,3 \
  -e MUSA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e PYTHONUNBUFFERED=1 \
  -v /mnt/nfs/models:/home/dist/models:ro \
  registry.mthreads.com/mcconline/inference/vllm-omni:minimax-h3-20260815 \
  bash -lc 'exec vllm serve \
    /home/dist/models/MiniMax/MiniMax-H3/Ref2VA \
    --omni --host 0.0.0.0 --port 8092 --trust-remote-code \
    --num-gpus 4 --tensor-parallel-size 4 \
    --text-encoder-tp-size 4 \
    --vae-patch-parallel-size 4 --vae-parallel-mode tile \
    --vae-use-tiling --diffusion-attention-backend FLASH_ATTN'
```

Verify both health and a real request. Server startup alone is not a pass:

```bash
curl --fail http://127.0.0.1:8092/health
```

Use `/v1/videos` plus polling for 50-step requests. The synchronous endpoint
has a server-side timeout and returned HTTP 504 for a request longer than ten
minutes even though the diffusion worker was otherwise healthy.

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

Qwen Image Edit runs independently on one GPU:

```bash
vllm serve /home/dist/models/Qwen/Qwen-Image-Edit-2511 \
  --omni --host 0.0.0.0 --port 8093 --trust-remote-code \
  --num-gpus 1 --tensor-parallel-size 1 \
  --vae-patch-parallel-size 1
```

Connect the three services to Studio with:

```bash
export STUDIO_H3_FL2VA_URL=http://127.0.0.1:8091
export STUDIO_H3_REF2VA_URL=http://127.0.0.1:8092
export STUDIO_IMAGE_EDIT_PROVIDER=vllm-omni
export STUDIO_IMAGE_EDIT_BASE_URL=http://127.0.0.1:8093
export STUDIO_IMAGE_EDIT_MODEL=/home/dist/models/Qwen/Qwen-Image-Edit-2511
```

## Parameter matrix

| Candidate | Result | Decision |
| --- | --- | --- |
| TP4 / TE4 / VAE-PP4 / FlashAttention / no offload | 50-step: `1709.6s`; warm two-step: `83.6s` | Recommended |
| Same profile with CPU offload | steady 50-step: about `1710-1714s` | Memory-safe fallback |
| Same profile with `--enforce-eager` | warm two-step: `85.5s` | Keep compiled default |
| VAE-PP1 | warm two-step: `154.0s` | Reject: much slower |
| VAE-PP2 | service initialization fails | Unsupported by H3 native VAE |
| TE1 | real Ref2VA request OOMs on rank 0 | Reject |
| TE2 | text-encoder process-group initialization fails | Unsupported in this TP4 layout |
| TORCH_SDPA | two-step request tries to allocate another `26.39GiB` and OOMs | FlashAttention required |
| USP4 / HSDP4 / no offload | two-step: `146.7s`; 50-step exceeded 35 minutes | Reject on current MUSA/MCCL stack |
| USP4 / HSDP4 / CPU offload | first forward fails on cross-device parameter storage | Unsupported combination |
| TP8 / TE8 | no eligible contiguous eight-GPU development host | Not run |

## Notes and non-options

- H3's native VAE accepts patch parallel size `1` or the complete DiT group;
  partial groups such as VAE-PP2 with TP4 fail validation.
- H3's wrapper does not consume `--vae-use-slicing`; do not treat that flag as
  a performance lever. Tiling comes from the checkpoint VAE configuration and
  the full VAE patch-parallel group.
- `--ring 1` does not enable useful parallel work; it explicitly keeps Ring
  parallelism disabled.
- Cache-DiT is not part of this profile. In this image its dependency detects
  MUSA as `CpuPlatform`, and cached quality modes intentionally skip model
  evaluations, so it is neither a proven MUSA optimization nor quality-neutral.
- Do not reduce inference steps, change flow shift, or lower the resolution when
  comparing serving profiles. Those alter the workload rather than optimize it.
