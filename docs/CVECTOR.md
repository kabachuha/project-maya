# Control vectors for creative writing

Steer the GLM-5.3-Flash runner's creative writing toward a trait - "more compassion, less
sadism", "warmer language", "sharper character focus" - with a **control vector**: a small per-layer
direction added to the residual stream, in the format llama.cpp uses for `--cvec`.

This is a faithful port of [jukofyork/control-vectors](https://github.com/jukofyork/control-vectors)
(Apache-2.0; the datasets and algorithm are his, credit preserved in
[`tools/cvec/datasets/ATTRIBUTION.md`](../tools/cvec/datasets/ATTRIBUTION.md)) to this engine. The
generator is [`tools/cvec_create.py`](../tools/cvec_create.py); the engine side is the `cvec_glm_apply`
kernel (add mode, one direction per trunk layer) loaded with the model like a LoRA.

---

## 1. Quick start

Make the vectors for one trait (e.g. compassion vs sadism):

```sh
python tools/cvec_create.py run \
    --trait dark_tetrad_continuations/compassion_vs_sadism.json \
    --tokenizer <pack>/tokenizer \
    --pack <pack> \
    --gguf <any pack shard .gguf> \
    --num-samples 32 \
    --out /tmp/cvec-compassion
```

`run` = `extract` (the engine runs every sampled prompt on the GPU, reference path) + `create`
(analyze and export, CPU-only). It writes:

| File | Meaning |
| --- | --- |
| `/tmp/cvec-compassion/debias.gguf` | the midpoint correction - **always load it with a concept** |
| `/tmp/cvec-compassion/compassion.gguf` | steer toward compassion |
| `/tmp/cvec-compassion/sadism.gguf` | steer toward sadism (~the negative of the above) |
| `/tmp/cvec-compassion/samples/` | `A.npy`, `B.npy`, `meta.json` (re-run `create` without re-extracting) |

Apply them to a generation:

```sh
STRATA_GLM_CVEC="/tmp/cvec-compassion/debias.gguf,/tmp/cvec-compassion/compassion.gguf" \
    ./build/strata --glm-pack <pack> ...
```

The start log says where it landed:

```
glm cvec: 2 file(s), add mode, layers 1..44 (43 steered)
```

The available traits (`--trait`, under [`tools/cvec/datasets/`](../tools/cvec/datasets/)):
`dark_tetrad_continuations/` (compassion vs sadism, empathy vs sociopathy, honesty vs machiavellianism,
humility vs narcissism), `other_continuations/` (optimism vs nihilism), `writing_style_continuations/`
(character focus, language, storytelling).

---

## 2. The pair rule

A concept vector is **relative to its debias vector**: the generator removes the midpoint of the two
class means from both poles, so the concept encodes *the difference between the poles*, not an absolute
style. The engine **sums** every loaded file's per-layer directions, so a full effect is always the pair:

```
STRATA_GLM_CVEC="debias.gguf,compassion.gguf"     # toward compassion
STRATA_GLM_CVEC="debias.gguf,sadism.gguf"         # toward sadism
STRATA_GLM_CVEC="debias.gguf,compassion.gguf:0.5" # a gentler dose (per-file scale)
```

Loading a concept without its debias is not an error but a wrong vector - the steering points at
"the pole, minus nothing" instead of "the pole, relative to the middle".

The list is comma-separated `path:scale` items (scale optional, default 1; several files are summed per
layer). Each file's `direction.<L>` (1-based `L`) is the direction of engine layer `L - 1`, applied right
after that layer's FFN write - the 1-based label is jukofyork's own (`layer_index + 1`), so a vector made
for this engine lands on the right layer without remapping.

**Layer range.** By default the engine steers trunk layers `1 .. n_layers - 1` (0-based): layer 0 has no
direction (the embedding sits before any layer), and the last layer is left alone. jukofyork's vectors
cover layers `0 .. n_layers - 2`, so on Maya-S24 (a 45-layer trunk) **43 layers are steered** - the
`layers 1..44 (43 steered)` line above. `STRATA_GLM_CVEC_FIRST` / `STRATA_GLM_CVEC_LAST` (or
`--control-vector-layer-range A B`) override the range; `STRATA_GLM_CVEC` can also come from the CLI
(`--control-vector FILE`, `--control-vector-scaled FILE:SCALE[,...]`).

**Per request.** In `--serve`, a loaded vector is on by default; a request switches it with the
`cvec=0|1` key, so one server can answer steered and unsteered requests.

---

## 3. How the vectors are made

For each sampled prompt triplet - one stem + one writing prompt, three continuations
(`baseline` / `class0` / `class1`) - the generator runs the model's **reference path** and records the
last token's mean-over-streams residual after the embedding and after every layer (`CVEXTRACT`):

```
delta[layer] = residual_after_layer - residual_after_previous_layer
A = class0 - baseline        (per layer)
B = class1 - baseline        (per layer)
```

Per layer it then does exactly what jukofyork's `direction_analyzer.py` does:

1. Eigendecompose the **symmetrized cross-covariance** `C = (AᵀB + BᵀA) / 2` (eigenvalues ascending,
   eigenvectors as rows).
2. Keep the eigenvectors whose **discriminant ratio** (between-class vs within-class variance on the
   direction) clears the running best by `--tol` (default 0.5), and **compound the strongest greedily**
   into one unit direction per layer.
3. **De-bias at the midpoint** of the two class means on that direction:

   ```
   debias   = midpoint * u
   class0   = (mean_A - midpoint) * u      # ~= -(class1)
   class1   = (mean_B - midpoint) * u
   ```

The three GGUFs are that, one `direction.<L>` F32[`n_embd`] tensor per steered layer
(`general.architecture="controlvector"`, `controlvector.model_hint="glm5-next"`,
`controlvector.layer_count` = the trunk's layer count).

One documented deviation: the analysis runs in **float64** (jukofyork's script is float32) - the
eigensystem is more stable, and the GGUFs are stored F32 either way.

---

## 4. Extraction: requirements and cost

`extract` drives the engine over its serve protocol (`CVEXTRACT <file> <ids...>`), so it needs:

- **The whole trunk on one part** - no layer split (`STRATA_GLM_SPLIT=0`-style loading on a single
  device that holds the model). On Maya-S24 (~93 GB) that is a big-GPU machine, not a 24 GB card.
- **The reference path** - the tool sets `STRATA_GLM_SLOW=1` itself.
- **The tokenizer** - `--tokenizer <pack>/tokenizer` (or a shard GGUF with the tokenizer inside).

Cost: **3 × `--num-samples` forward passes** over the sampled prompts (baseline + both classes per
triplet), each a full prompt on the *slow* path. `--num-samples 32` is a verification-sized run;
jukofyork used ~10,000 prompts for his released vectors, and a full-quality run is meant to happen on a
spare machine. `extract` is **resumable** (`--resume`): it re-opens the same `--out` and continues where
it stopped. `create` is CPU-only and takes seconds.

Useful subcommands:

```sh
# what a triplet tokenizes to (debugging prompt assembly)
python tools/cvec_create.py tokenize --trait ... --tokenizer <pack>/tokenizer --show 2

# the two halves separately (e.g. extract here, create elsewhere)
python tools/cvec_create.py extract --trait ... --pack <pack> --gguf <shard> --out /tmp/samples
python tools/cvec_create.py create  --samples /tmp/samples --out /tmp/cvec
```

Tuning knobs (all jukofyork defaults): `--num-samples`, `--seed`, `--tol` (0.5), `--start-layer` (0),
`--skip-end` (1 - the trailing layer he skips), `--model-hint` (`glm5-next`).

---

## 5. What was verified, where

| Check | Where | Result |
| --- | --- | --- |
| The math (planted direction recovered, noise filtered, midpoint/debias, `class1 ≈ -class0`) | `tools/test_cvec_create.py` | pass |
| GGUF round-trip (1-based labels, `None` layers skipped, F32[`n_embd`], `layer_count`) | `tools/test_cvec_create.py` | pass |
| The add kernel: `h += d` in every stream of every token, bitwise; non-steered layer and `cvec=0` no-ops | `ctest cvec_parity` | pass |
| Extraction on a real GLM model (45-layer fixture): shape (45+1)×`n_embd`, bitwise-deterministic, last-token-sensitive | `ctest glm_cvec_extract_test` | pass |
| Load into the real Maya-S24 pack (shape/tensor match) | the 3090 | `glm cvec: 2 file(s), add mode, layers 1..44 (43 steered)` |
| The rest of the suite | `ctest` (59 tests) | 56 pass; 3 pre-existing environmental failures (a missing Q2_0 fixture, a CPU without AVX512-VNNI, an nvcc-13 frontend crash in the untouched `glm_lora_kernels_test.cu`) |
| Real steered generation | not on this box - the 93 GB model does not fit the 24 GB card | run it on the generation machine |

So: the math, the kernel, the extraction, and the load are all checked end-to-end on this machine; the
one thing to eyeball remotely is the text itself (is `:1.0` too strong, is `:0.5` the sweet spot).

---

## 6. Notes

- **Building the engine here** uses the system g++ 13 as the CUDA host compiler and conda's nvcc 13 for
  the device code (conda's g++ 15 trips an EDG assertion in nvcc 13's frontend), and
  `CMAKE_CUDA_ARCHITECTURES=86` to cut compile time to the local 3090. A build for the generation
  machine wants the full architecture list and, for the test suite, `-DSTRATA_BUILD_TESTS=ON`.
- **jinja2 quirk:** this engine's GLM chat template uses `break`/`continue` inside tool-calling macros
  that never execute for plain writing prompts; some jinja2 builds (3.1.6 on Python 3.14) reject the
  tags at parse time. `cvec_create.py` neutralizes those specific tags before rendering, so the prompt
  text is byte-identical to the engine's own.
- **The datasets** are jukofyork's, Apache-2.0 - keep the credit and license in
  [`tools/cvec/datasets/`](../tools/cvec/datasets/) (see the attribution file there).
- A vector is tied to the **model and the prompt format** it was extracted from (template, reasoning
  effort line). Vectors for Maya-S24 will not steer another GLM checkpoint meaningfully.
