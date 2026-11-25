# GLM-4.5+ / GLM-4.6 in Maya (`maya-glm46`)

`maya-glm46` is the engine for Zhipu's **GLM-4.5 / GLM-4.6** (357 B: 92 trunk layers + one NextN block),
the `glm4moe` architecture of llama.cpp.  It is forked from `maya-mm3`'s engine (the GQA transformer family
MiniMax-M3.1 comes from) and runs on maya-ds4's GGUF block-type kernels (`src/ds4/ds4_iq`), so a llama.cpp
GGUF serves in place - no weight copies.  The NextN block is served as the model's **native MTP drafter**
(`--mtp`, on by default).

## The architecture it serves

* **GQA attention**: 96 query heads / 8 KV heads (12 a group), 128 a head; q, k and v have a bias; a per-head
  RMSNorm on q and k (128 values) and a **partial rope** on the first 64 dims (llama.cpp's "Neox" pairing -
  channel *i* with *i* + 32, frequency *i*); a causal pass over a K / V cache that is BF16, or (`--kv-cache`)
  int8 / fp8 / fp4 with the rows stored rotated.
* **Plain residual**: `h += attention`, then `h = rms(h) * w` (post-attention norm), the FFN, `h += ffn`.
* **MoE**: a 3-layer dense lead (FF width 12288), then 160 routed experts (FF 1536): sigmoid gating with a
  per-expert *selection* bias, the top 8 taken on the biased scores but weighted by the *unbiased* sigmoid,
  renormalised and scaled by 2.5; one shared expert (always on) beside them.
* **the shipped quants** (ReadyArt's `GLM-4.6-Derestricted-GGUF`), both with the NextN block:
  * `Q8_0-IQ3_S-IQ3_S-IQ4_XS` (3.94 bpw, ~176 GB, 4 shards): Q8_0 for attention, the dense and shared FFNs, the
    head and the embedding; IQ3_S for the routed gate and up, IQ4_XS for the down; F32 for the norms, router,
    biases and the per-expert selection bias.  ~10.9 MB an expert;
  * the smaller `Q6_K-IQ2_XS-IQ2_XS-IQ3_S` (2.9 bpw, ~129 GB, 3 shards): Q6_K for the dense weights (attention,
    the dense / shared FFNs, head, embedding, the NextN einsum), IQ2_XS for the routed gate and up, IQ3_S for the
    down.  ~7.6 MB an expert - the whole model fits better in RAM and a larger share of the experts in VRAM;
  * the smallest `Q5_K-IQ2_XXS-IQ2_XXS-IQ3_XXS` (2.6 bpw, ~114 GB, 3 shards): Q5_K dense, IQ2_XXS gate and up,
    IQ3_XXS down; ~7.1 MB an expert.  IQ2_XXS is the roughest of the sets - the answer quality it costs shows most
    outside the languages and domains it was measured on.
  Other type mixes are accepted as long as `src/ds4/ds4_iq` covers them (Q4_K / Q5_K / Q6_K / Q8_0 dense, the
  IQ1 / IQ2 / IQ3 / IQ4 experts); anything else is refused at load, named.

## The pack

    python tools/glm46/pack_gguf.py --model ~/models/GLM-4.6-Q8_0-IQ3_S-IQ3_S-IQ4_XS

writes `<model>/maya-glm46/` (`glm46.txt` - the geometry from the GGUF's `glm4moe.*` keys; `tensors.txt` - every
tensor's type, shard, offset and shape; `tokenizer/`), so the engine maps the model's own files.  When the GGUF
carries the NextN (MTP) block - `block_count` above `nextn_predict_layers`, as the shipped quant does - its full
layer at `blk.<trunk>` and `nextn.eh_proj` / `enorm` / `hnorm` / `shared_head_norm` are packed too and
`glm46.txt` says `nextn 1`; the engine then drafts with it.  A GGUF without one packs as before (`nextn 0`, the
engine says `NOTE: --mtp: this pack has no NextN (MTP) block`).

## Running

    ./build/maya-glm46 --pack ~/models/GLM-4.6-Q8_0-IQ3_S-IQ3_S-IQ4_XS/maya-glm46 \
        --ids 1,2,3,4 [--gen N] [--dump-logits FILE --last N] [--max-context N] [--gpus 0,1] \
        [--mtp on|off] [--mtp-drafts N]

    ./build/maya-glm46 --pack ... --serve            # GEN <max_new> [keys] <ids>, STOP, QUIT

**Native MTP drafting** (`--mtp`, on when the pack has the NextN block; `--mtp-drafts N`, default 3, at most 8).
The block is a full layer on the head's GPU: a KV cache of its own, the trunk's MoE, and its input the pair
`[rms(emb(token at p+1), enorm) | rms(h_p, hnorm)]` through `eh_proj`, `h_p` the trunk's post-output-norm hidden
(llama.cpp's `t_h_nextn`).  Its cache's entry p thus pairs `h_p` with the token at p+1 and its head - the trunk's
`output.weight`, after `shared_head_norm` - reads the token at p+2.  Every trunk forward fills the entries its rows
know (a prompt's chunk in slices of 64, a decode step's predecessor, a verify window's rows); a round then chains
up to N greedy drafts (each later step's carrier the block's own `shared_head_norm` output), the trunk verifies them
all in one window (`kWin` rows), the accepted prefix is kept with `commit_rows`, and the block's cache rows the
rejected drafts wrote are rewritten - in order, before any query past them - by the next rounds.  Every emitted
token is the trunk's own, whatever the drafts are (`--mtp` changes the speed, never the answer).  Off for
`--parallel` and with `--mtp off`.

## Maya's setup, dashboard and API (the TUI)

`maya-glm46` is a first-class engine of the Maya launcher, so GLM-4.6 gets the same setup screen, dashboard and API
as GLM-5.3-Flash and DeepSeek (the Monitor's Throughput, Where the experts live, the GPUs / VRAM map of a layer
split, the chat, the OpenAI and Anthropic endpoints).  Point the setup at the GGUF set you have:

    ./maya.sh --setup --model glm-4.6 --model-dir ~/models/GLM-4.6-Derestricted-Q8_0-IQ3_S-IQ3_S-IQ4_XS

(plain text with `--plain`, unattended with `--yes`; the folder must hold every shard of the set - a lone `.gguf`
works, a set of `-0000N-of-0000N.gguf` shards is found by its first).  The setup

1. checks the GPUs, asks which cards (the layer split over several) and the context - GLM-4.6's own reach comes from
   the GGUF (`glm4moe.context_length`, 200K), the default is 32K.  The K / V rows are in VRAM, 2 KB a position a
   layer in the setup's default `--kv-cache int8` (4 KB with `bf16`, 1.1 KB with `fp4`): 6 GB at 32K for all 93
   layers with int8, 1.5 GB at 8K.  `--context` and `--kv-cache` choose;
2. builds `maya-glm46` (CUDA: V100 and newer; AMD: the HIP build);
3. builds the pack (`tools/glm46/pack_gguf.py`, no copy of the weights) and exports the tokenizer and chat template;
4. writes `maya-glm-4.6.json` and `run-maya-glm-4.6.sh`.

Then `./maya.sh` (or `run-maya-glm-4.6.sh`) starts the dashboard on `http://127.0.0.1:8080/`, the API on `/v1`
(OpenAI) and `/v1/messages` (Anthropic), with the Monitor's live numbers from the engine's `INFO` / `STAT` / `VRAM`
lines: decode speed, the expert tiers (VRAM / pinned RAM / SSD only, the CPU lane's share), the GPU map of each
card, and `Speculation: MTP` (the NextN block drafting; the answers are the model's own).  A setup again
(`--repack`, `--context`, `--port`, `--gpus 0,1`, `--env KEY=VALUE`, `--mtp off`, `--mtp-drafts N`) keeps an earlier
config's engine settings it does not ask for; the Models page's switches and the chat write into the same config.

    ./maya.sh --config maya-glm-4.6.json             # start exactly this config
    ./maya.sh --check                                # what this PC can run, before the download or setup
    ./maya.sh --bench                                # the standard speed test (its own engine, same config)
    ./maya.sh --report                               # maya-report.txt: the PC, the setup, the speed lines

GLM-4.6 without the launcher (a service, a script, another front end) is the engine's own protocol above; the
launcher's config is just `exe`, `args`, `tokenizer`, `gpu` and the model's sampling.

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
  steered layer's FFN write, where llama.cpp's `build_cvec` sits.  A vector's `controlvector.model_hint` of
  `glm4_moe` (an older llama.cpp's spelling of this architecture, which the gghfez files carry) is accepted as
  this model, as `glm4moe` and `glm4` are; only another model's hint gets a NOTE (it is applied too, as
  llama.cpp would).

## Speed on a PC whose experts stream from RAM

A token routes 8 experts in each of the 92 layers - 736 expert computations, ~10.9 MB of weights each in the
Q8_0-IQ3_S set (~7.6 MB in the Q6_K-IQ2_XS one).  What decode costs is decided by where those experts live: the
resident region (VRAM), the staging slots (the hot ones copied into VRAM), the pinned RAM pool (the CPU lane
computes them) and the files on the SSD.  With two cards, 32 + 24 GB, and the 32K context, about 1,600 of the
14,720 experts are resident; the other ~89% run on the CPU lane, and that is the 4 tok/s - llama.cpp in the same
regime is in the same range.  The smaller quant both shrinks the model (~129 GB: every expert's read from RAM is
30% shorter) and its dense part (Q6_K ~13.5 GB vs Q8_0 ~17 GB), which leaves more for resident experts.

What moves it, in the order it pays:

* **the learned expert profile** (`--expert-profile FILE`, which a setup writes as
  `data/expert-profile-glm46.bin`): the engine counts every layer's routings and saves them after a request; the
  next engine start puts the most-routed experts in VRAM instead of a round-robin.  MoE routing is heavy-tailed, so
  the resident 11% can cover much more than 11% of the 736 slots a token (this is the same mechanism the other
  engines' shipped profiles use).  The file is per PC and per model geometry (92 x 160), a wrong one is ignored;
  run, talk to it, restart (`./maya.sh`, or the Models page) and compare with `--bench`.
* **the K / V cache's format** (`--kv-cache int8`, a setup's default; `fp8`, `fp4`, `bf16`): the rows are stored
  rotated by the 128-point Hadamard transform and rounded to 8 bits (an F32 scale a row for int8, a power-of-two
  scale a 32 values for fp8) - half the VRAM of bf16, at a rounding the rotation makes small (the kernel test below
  checks the decoding of every format to ~1e-6 against a double reference).  At 32K that is 6 GB instead of 12 GB,
  ~550 more resident experts; `fp4` is 3.5 times smaller still.
* **a shorter context** - the K / V rows are in VRAM (2 KB a position a layer with int8: 6 GB at 32K for all 93
  layers, 1.5 GB at 8K) - so `--max-context 8192` hands ~5 GB to resident experts.  The context is a setup setting
  (`--context`), and the setup prints the cache's size.
* **`--mtp off`** on this kind of PC: a verify window of 4 rows reads the *union* of their experts (up to ~26 a layer
  instead of 8) while it yields up to 4 tokens only if they are accepted.  Where almost every expert streams, that
  is a loss (the GLM engine measured x0.62 on one V100); where most are resident it wins.  Set it in the config
  (`--drafting off` in a setup writes `--mtp off`; `--drafting on` writes `--mtp on`), then compare with `--bench`.
* **`--chunk`** (a setup setting): the prompt chunk's scratch is ~1.1 GB a card at 1024 tokens - `--chunk 512`
  gives ~0.5 GB of it back to the experts.
* **`--cpu-threads N`**: the lane's experts are ggml's IQ3_S / IQ4_XS dots on this CPU; on a 13th/14th-gen Intel
  the E-cores can drag (try 8, the P-cores, against the default).  The engine also measures at load whether an
  expert is faster copied into a staging slot or computed on the lane ("CPU lane vs staging (measured)").

**KV streaming** (`--kv-streaming on`, `--kv-resident N`; `on` alone keeps 32K positions a layer in VRAM) is
implemented now: every position's K / V rows live in pinned host memory as well as in a VRAM ring of N positions
(whole 4096-position blocks, never below a prompt chunk), and attention reads a key from the ring while it holds the
position, else from host memory over PCIe.  It is the way to run a context that does not fit VRAM at all (GLM-4.6's
200K needs ~76 GB of K / V at BF16: impossible on a 32 + 24 GB pair) - but this model's attention is **dense**: every
decode token reads the whole context's keys, so the positions past the ring are read over PCIe each token (about
112 MB a layer a token at 24K positions: hundreds of ms a token).  That is why it is off by default here: at 32K the
cache already fits VRAM, where a decode token reads it in ~15 ms; a shorter context (above) is the same VRAM saving
without the host reads.  With `--parallel` it is off (each conversation has its own cache).

Diagnostics: `GLM46_TIMING=1` prints a decoded token's routing waits, CPU-lane and promotion time; the serve `STAT`
line has `vram_hit` (the routed experts found in VRAM) and `cpu` (the share the lane computed); `--verbose` prints a
layer's GPU time; `GLM46_PLAN_TRACE=1` prints the start's memory plan a card at a time.

## What is not served yet

Tensor parallel (`--tp`) and DSpark drafts are refused with a message (the model's own NextN block is the
drafter here); the vision tower is off for this text pack; `--parallel` decodes without drafts and turns KV
streaming off (each conversation has its own cache).  They are the next steps, not silent fallbacks.

## Verification

* `ctest -R glm46_kernels_test` - the new attention kernel (GQA, causal, BF16 rows) across sequence lengths
  that cross the 64-key tile boundary, against a double reference; the q/k/v bias add, the plain SwiGLU and the
  sigmoid router likewise.  Exact to ~1e-7.  The same test writes int8 / fp8 / fp4 K / V caches through the
  engine's own `qk_norm_rope`, decodes them back on the host (scales, the Hadamard rotation) and checks the
  attention against the double reference over exactly those rows: ~1e-6.
* `python tools/glm46/glm46_synth_gguf.py --quant xs|xxs` / `parity.py` - the smaller quants' type maps (Q6_K or
  Q5_K dense, IQ2_XS / IQ2_XXS experts, IQ3_S / IQ3_XXS down) run the whole engine, parity and MTP included (2%,
  top-1 agreeing as the Q8_0 set); `tools/glm46/plan_check.py --model ...` checks their memory plan (dense bytes,
  the per-expert blob), and `--resident-cap 0 --staging 0` puts the experts on the CPU lane (ggml's IQ2_XXS /
  IQ3_XXS dots) to the same tokens.
* `python tools/glm46/parity.py --model <dir with the synthetic model>` - `maya-glm46 --dump-logits` against
  `ref/glm46.py` (the architecture's reference forward) on the synthetic model: a relative RMS of ~2% on the
  logits and every top-1 agreeing, the same difference llama.cpp's own CPU and CUDA builds show against the
  same reference (q8_1 activations, FP16 weight bands, BF16 caches - the parity tool runs `--kv-cache bf16`).  A row where the reference's routing sits on
  a #2/#3 near tie (< 0.02) may pick the other expert - as llama.cpp's backends do - and the tool says so.
* The control vectors: `python tools/glm46/parity.py --cvec FILE:SCALE [--cvec-mode add|project]`, and the same
  flags on the engine's logits.
* `python tools/glm46/mtp_parity.py --model <dir with the synthetic model>` - the NextN block against
  `ref/glm46.py`'s own block (`mtp_prefill` / `mtp_step`): the chain's three steps all agree on top-1, the carries
  and logits within ~2% (the same quantized-kernel difference as the trunk), so the eh_proj pairing, the block's
  cache rows and the chain's carrier are right.
* `python tools/glm46/mtp_accept.py` - a synthetic model whose NextN block drafts the trunk's next token read one
  step early (`glm46_synth_gguf.py --mtp-identity`): 10-20 of its drafts are accepted per request, the tokens
  stay exactly `--mtp off`'s, and `--mtp-drafts 1..8` all pass - the verify / commit / chain-after-commit path.
* `python tools/glm46/plan_check.py --model <dir>` - the memory plan's two per-layer estimates against the pack's
  tensors: a layer's dense bytes must leave the three stacked `ffn_*_exps` tensors out (they are its experts, not
  weights) and one expert's blob must be their slice.  `GLM46_PLAN_TRACE=1 maya-glm46 ...` prints what each card
  needs: its layers' dense weights, the chunk's scratch, the staging slots (one expert's blob each - about 10.9 MB
  for the shipped quant), its K / V at the context, and the reserve.  A start that does not fit says the same and
  suggests a smaller `--chunk`, fewer `--staging` slots, a shorter `--max-context`, or more GPUs.

The synthetic model (`tools/glm46/glm46_synth_gguf.py`) is the tensor set and type map above at 1/10 the width,
9 layers, 8 experts a token's 2 - small enough for one GPU, structurally the shipped quant.
