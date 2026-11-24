# GLM-4.5+ / GLM-4.6 in Maya (`maya-glm46`)

`maya-glm46` is the engine for Zhipu's **GLM-4.5 / GLM-4.6** (357 B, 92 trunk layers + one NextN block),
the `glm4moe` architecture of llama.cpp.  It is forked from `maya-mm3`'s engine (the GQA transformer family
MiniMax-M3.1 comes from) and runs on maya-ds4's GGUF block-type kernels (`src/ds4/ds4_iq`), so a llama.cpp
GGUF serves in place - no weight copies.

## The architecture it serves

* **GQA attention**: 96 query heads / 8 KV heads (12 a group), 128 a head; q, k and v have a bias; a per-head
  RMSNorm on q and k (128 values) and a **partial rope** on the first 64 dims (llama.cpp's "Neox" pairing -
  channel *i* with *i* + 32, frequency *i*); a causal pass over a BF16 K / V cache.
* **Plain residual**: `h += attention`, then `h = rms(h) * w` (post-attention norm), the FFN, `h += ffn`.
* **MoE**: a 3-layer dense lead (FF width 12288), then 160 routed experts (FF 1536): sigmoid gating with a
  per-expert *selection* bias, the top 8 taken on the biased scores but weighted by the *unbiased* sigmoid,
  renormalised and scaled by 2.5; one shared expert (always on) beside them.
* The shipped quant - ReadyArt's `Q8_0-IQ3_S-IQ3_S-IQ4_XS` (3.94 bpw) - is Q8_0 for attention, the dense and
  shared FFNs, the head and the embedding; IQ3_S for the routed gate and up, IQ4_XS for the down; F32 for the
  norms, router, biases and the per-expert selection bias.  Other type mixes are accepted as long as
  `src/ds4/ds4_iq` covers them; anything else is refused at load, named.

## The pack

    python tools/glm46/pack_gguf.py --model ~/models/GLM-4.6-Q8_0-IQ3_S-IQ3_S-IQ4_XS

writes `<model>/maya-glm46/` (`glm46.txt` - the geometry from the GGUF's `glm4moe.*` keys; `tensors.txt` - every
trunk tensor's type, shard, offset and shape; `tokenizer/`), so the engine maps the model's own files.  The
NextN (MTP) block stays in the GGUF and is not served yet.

## Running

    ./build/maya-glm46 --pack ~/models/GLM-4.6-Q8_0-IQ3_S-IQ3_S-IQ4_XS/maya-glm46 \
        --ids 1,2,3,4 [--gen N] [--dump-logits FILE --last N] [--max-context N] [--gpus 0,1]

    ./build/maya-glm46 --pack ... --serve            # GEN <max_new> [keys] <ids>, STOP, QUIT

Everything the other engines have:

* **batched prefill** (`--chunk N` tokens through a layer at once; the dense projections as GEMMs, the experts
  as sorted rows) and a token path for decode;
* **expert caching**: each GPU keeps as many of its layers' routed experts in VRAM as fit (`--resident-cap`,
  `--staging`, `--profile`, a pinned pool, promotions); the rest are staged over PCIe or computed on the CPU
  lane (`--cpu-threads`).  The logits are bit-identical whichever tier an expert runs on;
* the **layer split** over several GPUs (`--gpus`, `--layer-split auto|L,L,...`);
* **`--ablate DIRECTION.gguf`** (a direction projected out of every residual write; GEN `ablate=0|1`);
* **control vectors** - llama.cpp's flags, the same as the GLM engine:

      ./build/maya-glm46 --pack ... \
          --control-vector-scaled glm-4.6_honesty_vs_machiavellianism__debias.gguf:1.0,\
      glm-4.6_honesty_vs_machiavellianism__machiavellianism.gguf:1.0

  (`--control-vector FILE` is scale 1; `--control-vector-layer-range A B`; `--cvec-mode add|project`;
  `--cvec-dir per-layer|single:L`).  The loader reads llama-cvector-generator GGUFs (`direction.<l>`, 0-based,
  layer 0 never steered) and sums the files' directions times their scales; the vector is applied after the
  steered layer's FFN write, where llama.cpp's `build_cvec` sits.

## What is not served yet

Tensor parallel (`--tp`), the NextN / MTP drafter, DSpark drafts and the vision tower are refused or off for
this architecture (`--tp` stops with a message); the K / V caches are BF16 only (`--kv-cache bf16`, KV
streaming off).  They are the next steps, not silent fallbacks.

## Verification

* `ctest -R glm46_kernels_test` - the new attention kernel (GQA, causal, BF16 rows) across sequence lengths
  that cross the 64-key tile boundary, against a double reference; the q/k/v bias add, the plain SwiGLU and the
  sigmoid router likewise.  Exact to ~1e-7.
* `python tools/glm46/parity.py --model <dir with the synthetic model>` - `maya-glm46 --dump-logits` against
  `ref/glm46.py` (the architecture's reference forward) on the synthetic model: a relative RMS of ~2% on the
  logits and every top-1 agreeing, the same difference llama.cpp's own CPU and CUDA builds show against the
  same reference (q8_1 activations, FP16 weight bands, BF16 caches).  A row where the reference's routing sits on
  a #2/#3 near tie (< 0.02) may pick the other expert - as llama.cpp's backends do - and the tool says so.
* The control vectors: `python tools/glm46/parity.py --cvec FILE:SCALE [--cvec-mode add|project]`, and the same
  flags on the engine's logits.

The synthetic model (`tools/glm46/glm46_synth_gguf.py`) is the tensor set and type map above at 1/10 the width,
9 layers, 8 experts a token's 2 - small enough for one GPU, structurally the shipped quant.
