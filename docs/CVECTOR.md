# Control vectors for creative writing

Steer a model's creative writing toward a trait - "more compassion, less sadism", "warmer
language", "sharper character focus" - with a **control vector**: a small per-layer direction on
the residual stream, in the file format llama.cpp uses for `--cvec`.  Both engines on this branch
load the same files:

- **`maya-ds4`** (DeepSeek-V4.1-Flash): `--control-vector` / `--control-vector-scaled` and friends,
  applied after each steered layer's MoE write - and the serve command `CVEXTRACT` that extracts
  the per-layer streams the vectors are made from.
- **`maya`** (GLM-5.3-Flash): the same flags; in `--serve` a loaded vector switches per request
  (`cvec=0|1`).  This branch has no extraction path for it, only the loader.

This is a port of [jukofyork/control-vectors](https://github.com/jukofyork/control-vectors)
(Apache-2.0; the datasets and algorithm are his, credit preserved in
[`tools/cvec/datasets/ATTRIBUTION.md`](../tools/cvec/datasets/ATTRIBUTION.md)).  The generator is
[`tools/cvec_create.py`](../tools/cvec_create.py); the engine side is the `strata::kernels::cvec`
kernel (shared by both engines) with the host loaders `src/program/generate.cpp` (GLM) and
`src/ds4/ds4_cvec.hpp` (DeepSeek).

---

## 1. Quick start

Make the vectors for one trait (e.g. compassion vs sadism):

```sh
python tools/cvec_create.py run \
    --trait dark_tetrad_continuations/compassion_vs_sadism.json \
    --pack <model>/maya-ds4 \
    --num-samples 32 \
    --out /tmp/cvec-compassion
```

`run` = `extract` (the engine runs every sampled prompt over its serve protocol, GPU) + `create`
(analyze and export, CPU-only).  It writes:

| File | Meaning |
| --- | --- |
| `/tmp/cvec-compassion/debias.gguf` | the midpoint correction - **always load it with a concept** |
| `/tmp/cvec-compassion/compassion.gguf` | steer toward compassion |
| `/tmp/cvec-compassion/sadism.gguf` | steer toward sadism (~the negative of the above) |
| `/tmp/cvec-compassion/samples/` | `A.npy`, `B.npy`, `meta.json` (re-run `create` without re-extracting) |

Note: `A.npy` / `B.npy` are **raw float32 memmaps, not `.npy` files** (no header) - the extraction
writes them one triplet at a time so a run can be resumed.  Read them with
`np.memmap(..., shape=<from meta.json>)`, not `np.load` (which fails with `UnpicklingError`).
`create` does this for you; it also refuses a samples dir where the file size disagrees with
`meta.json` (a stale file from a different `--num-samples`) or the extraction is incomplete
(`done < T`).

Apply them to a generation (on top of `--ablate`, if one is loaded - the two compose):

```sh
./build/maya-ds4 --pack <model>/maya-ds4 --serve \
    --control-vector-scaled /tmp/cvec-compassion/debias.gguf,/tmp/cvec-compassion/compassion.gguf
```

The start log says where it landed:

```
ds4: control vectors: add:1-39 (2 file(s))
```

The available traits (`--trait`, under [`tools/cvec/datasets/`](../tools/cvec/datasets/)):
`dark_tetrad_continuations/` (compassion vs sadism, empathy vs sociopathy, honesty vs
machiavellianism, humility vs narcissism), `other_continuations/` (optimism vs nihilism),
`writing_style_continuations/` (character focus, language, storytelling).

---

## 2. The pair rule

A concept vector is **relative to its debias vector**: the generator removes the midpoint of the
two class means from both poles, so the concept encodes *the difference between the poles*, not an
absolute style.  The engine **sums** every loaded file's per-layer directions (each file times its
scale), so a full effect is always the pair:

```
--control-vector-scaled debias.gguf,compassion.gguf          # toward compassion
--control-vector-scaled debias.gguf,sadism.gguf              # toward sadism
--control-vector-scaled debias.gguf,compassion.gguf:0.5      # a gentler dose (per-file scale)
```

Loading a concept without its debias is not an error but a wrong vector - the steering points at
"the pole, minus nothing" instead of "the pole, relative to the middle".

Each file's `direction.<L>` is the direction of the **0-based** layer `L` (as both engines on this
branch read it, and as `--ablate-layer` does - a vector made for another engine's 1-based
convention lands one layer off).  The direction is applied right after that layer's MoE write.
Layer 0 has no direction (the embedding sits before any layer), so `direction.0` is impossible and
the first layer's analysis is not exported; a direction past the model is ignored, as in llama.cpp.

**Layer range.** By default the engines steer layers `1 .. n_layers - 1` (0-based): the last
steerable layer is `n_layers - 1` and layer 0 is out.  `--control-vector-layer-range A B`
overrides the range.

**Mode.** `--cvec-mode` is `add` (the default - `h += d`, llama.cpp's stock control vector) or
`project` (`h -= s (h.v) v` with `v` the unit direction and `s = |d|` - the
experimental-speed-projection mode, which subtracts the vector's component from the stream).
`--cvec-dir single:L` (project mode only) uses layer `L`'s direction for every steered layer
instead of each layer's own.

**Per request.** On `maya` (`--serve`) a loaded vector is on by default and a request switches it
with the `cvec=0|1` key, so one server can answer steered and unsteered requests.  On `maya-ds4`
the vectors are CLI-level only, like `--ablate`: the engine's requests share one residual layout
and a per-request toggle would race with the captured graphs.

---

## 3. How the vectors are made

For each sampled prompt triplet - one stem + one writing prompt, three continuations
(`baseline` / `class0` / `class1`) - the generator runs the engine's `CVEXTRACT` (the decode path,
the stream means a vector would steer) and records the last token's mean-over-streams residual
after the embedding and after every layer:

```
delta[layer] = residual_after_layer - residual_after_previous_layer
A = class0 - baseline        (per layer)
B = class1 - baseline        (per layer)
```

Per layer it then does exactly what jukofyork's `direction_analyzer.py` does:

1. Eigendecompose the **symmetrized cross-covariance** `C = (AᵀB + BᵀA) / 2` (eigenvalues
   ascending, eigenvectors as rows).
2. Keep the eigenvectors whose **discriminant ratio** (between-class vs within-class variance on
   the direction) clears the running best by `--tol` (default 0.5), and **compound the strongest
   greedily** into one unit direction per layer.
3. **De-bias at the midpoint** of the two class means on that direction:

   ```
   debias   = midpoint * u
   class0   = (mean_A - midpoint) * u      # ~= -(class1)
   class1   = (mean_B - midpoint) * u
   ```

The three GGUFs are that, one `direction.<L>` F32[`hidden_size`] tensor per steered layer
(`general.architecture="controlvector"`, `controlvector.model_hint="deepseek-v41"`,
`controlvector.layer_count` = the trunk's layer count).

One documented deviation: the analysis runs in **float64** (jukofyork's script is float32) - the
eigensystem is more stable, and the GGUFs are stored F32 either way.

---

## 4. Extraction: requirements and cost

`extract` drives `maya-ds4 --serve` over its stdin protocol: `CVEXTRACT <file> <id,id,...>` runs
the ids through the decode path one token at a time (the model is reset first, and any loaded
control vectors are switched off for the run and back on after, so the capture is the stock
model's) and answers `CVOK` / `ERR`.  It needs:

- **The model loaded, any placement** - single device or a layer split across several GPUs both
  work; each part of a split captures its own rows and the result is assembled in order.  On
  V4.1-Flash (~100 GB) that is a multi-GPU machine or one with the routed experts off the GPU, not
  a 24 GB card.
- **The pack directory** - `--pack <model>/maya-ds4`: the tokenizer comes from `<pack>/tokenizer`
  (override with `--tokenizer`), the geometry (`hidden_size`, `num_hidden_layers`) from
  `<pack>/ds4.txt`.  No GGUF shard argument needed.
- **The prompt regime** - `--thinking chat|thinking` (default `chat`; the server's default regime
  is thinking).  The vectors are extracted from the streams the prompt assembles, so match the
  regime you will steer: `--thinking thinking` to steer a default-serve conversation.

Cost: **3 × `--num-samples` forward passes** over the sampled prompts (baseline + both classes per
triplet), each prompt fed token by token.  `--num-samples 32` is a verification-sized run;
jukofyork used ~10,000 prompts for his released vectors, and a full-quality run is meant to happen
on a spare machine.  The engine prints an `INFO ds4 cvec extract: token N of M` line per 64 tokens,
so a long run shows progress on the server's log.  `extract` is **resumable** (`--resume`):
re-run it on the same `--out` and it continues where it stopped (it verifies the samples dir is
from the same run first).  `create` is CPU-only and takes seconds.

Useful subcommands:

```sh
# what a triplet tokenizes to (debugging prompt assembly)
python tools/cvec_create.py tokenize --trait ... --pack <model>/maya-ds4 --show 2

# the two halves separately (e.g. extract here, create elsewhere)
python tools/cvec_create.py extract --trait ... --pack <model>/maya-ds4 --out /tmp/samples
python tools/cvec_create.py create  --samples /tmp/samples --out /tmp/cvec
```

Tuning knobs (all jukofyork defaults): `--num-samples`, `--seed`, `--tol` (0.5), `--start-layer`
(0), `--skip-end` (1 - the trailing layer he skips), `--model-hint` (`deepseek-v41`),
`--thinking` (`chat`), `--engine-arg` (repeatable, extra `maya-ds4` arguments).

---

## 5. The GLM engine

`maya` loads the same files with the same flag names (`--control-vector FILE`,
`--control-vector-scaled FILE:SCALE[,...]`, `--control-vector-layer-range A B`,
`--cvec-mode add|project`, `--cvec-dir per-layer|single:L`) and applies them after each steered
layer's FFN write, with the per-request `cvec=0|1` switch in `--serve`.  Its loader warns (does
not refuse) when `controlvector.model_hint` is not `qwen4exp`; the DeepSeek loader warns when it
is not `deepseek-v41` - a hint mismatch is a nudge, the geometry check (tensor length = the
model's hidden size) is the refusal.  This branch carries no extraction path for the GLM engine,
so a GLM vector here would be hand-made or carried over; the generator targets `maya-ds4`.

---

## 6. What was verified, where

| Check | Where | Result |
| --- | --- | --- |
| The math (planted direction recovered, noise filtered, midpoint/debias, `class1 ≈ -class0`) | `python tools/test_cvec_create.py` | pass |
| GGUF round-trip (0-based labels, layer 0 left out, `None` layers skipped, F32[`hidden_size`], `layer_count`) | `python tools/test_cvec_create.py` | pass |
| The raw A/B memmaps + `meta.json` round-trip, resume keeping the extracted rows, stale/incomplete refusal | `python tools/test_cvec_create.py` | pass |
| The kernel: `project` within 1e-4 of double, `add` and `off` bitwise, at `n = 2560` (GLM) and `n = 7168` (past the old 4096 bound; V4.1's 5120 included) | `ctest cvec_parity` | pass |
| The DeepSeek loader (`ds4_cvec.hpp::build`): summing, range, `single:L`, label and refusal handling | `ctest ds4_cvec_test` | pass |
| `--ablate`'s loader untouched | `ctest ds4_ablate_test` | pass |
| End-to-end on a synthetic 40-layer pack (dim 2048, MXFP4 experts, built for the smoke tests): load, `CVEXTRACT` shape (`(n_layers + 1) x dim` F32) and byte-identical re-runs, the load line, steered generation in both modes, `--ablate` + cvec together, `--serve` with cvec loaded | `build/maya-ds4` one-shot and `--serve` against the dummy pack (`--prefill-chunk 0`) | pass - extraction is deterministic; `add`/`project` each change the generated tokens; the ablation direction is active alone and under cvec; a `CVEXTRACT` run with cvec loaded captures the stock model byte for byte (the vectors are off for the run) |
| The steered run under `compute-sanitizer --tool memcheck` (the non-chunk prompt path) | same dummy pack, `--control-vector ... --cvec-mode project` | 0 errors |
| Load into the real V4.1-Flash pack, steered generation, `--ablate` coexistence | not on this box - the model lives on the remote | run it there |

So: the math, the kernel (both widths), the loader, the tool, and the whole engine path
(extract, load, apply, coexist, serve) are checked on this machine - the engine path against a
synthetic checkpoint, since the real one does not fit here.  On the generation machine the
remaining things to eyeball are the real model's load line and the text itself (is `:1.0` too
strong, is `:0.5` the sweet spot).  Two caveats the synthetic run surfaced, neither a cvec fault:

- The dummy's random weights leave its residual stream diverging (norms in the 1e9-1e10 by the
  last layers; every consumer reads through an RMS norm, so generation stays finite).  A control
  vector whose norm matches that scale makes `project` mode overflow (its scale is the vector's
  norm); real vectors are O(1-10), and unit-norm synthetic ones run clean.  The smoke tests
  re-scaled their directions to unit norm for that reason.
- The chunked-prompt path (`--prefill-chunk` left at its default) faults on this synthetic
  geometry even without any control vector loaded (the router kernel indexes its shared memory
  out of bounds once the stream's NaN reaches it) - a pre-existing engine condition the dummy
  exposes, not something the cvec path changes (with no vector file loaded the cvec tables are
  inert).  The smoke tests therefore ran with `--prefill-chunk 0`; the two `layer_chunk` apply
  sites are the same `cvec_apply` calls as the decode path's.

---

## 7. Notes

- **Building the engine here** uses the system g++ 13 as the CUDA host compiler and conda's nvcc
  13 for the device code (conda's g++ 15 trips an EDG assertion in nvcc 13's frontend), and
  `CMAKE_CUDA_ARCHITECTURES=86` to cut compile time to the local 3090.  A build for the generation
  machine wants the full architecture list and, for the test suite, `-DSTRATA_BUILD_TESTS=ON`.
- **`--ablate` coexistence** (DeepSeek engine): a rank-1 LoRA or direction projected out of each
  layer's *write* (before the mHC combine) and a control vector added to the *residual* after it
  are independent stages; both load in `Model::load` and neither needs the other.  A
  `--ablate-layer N` on one of these files reads `direction.N` directly (0-based, same labels).
- **The datasets** are jukofyork's, Apache-2.0 - keep the credit and license in
  [`tools/cvec/datasets/`](../tools/cvec/datasets/) (see the attribution file there).
- A vector is tied to the **model and the prompt format** it was extracted from (the renderer,
  the `--thinking` regime).  A V4.1 vector will not steer another checkpoint meaningfully, and a
  `chat`-regime vector steers a `thinking`-regime conversation only approximately.
