"""tools/cvec_create.py - control-vector generator for Maya's DeepSeek-V4.1 engine (maya-ds4).

WHAT THIS IS
------------
A port of the GLM engine's control-vector generator (itself a faithful, self-contained port of
jukofyork's `control-vectors`, https://github.com/jukofyork/control-vectors, Apache-2.0) to the
DeepSeek-V4.1 engine.  It turns a contrastive writing-style dataset (e.g. "compassion vs
sadism") into a pair of GGUF control vectors that steer the model's creative writing toward one
pole or the other.

The algorithm (unchanged from jukofyork, `direction_analyzer.py`):
  1. For each sampled prompt triplet (baseline / class0 / class1, same stem + writing prompt), run
     the model and take the per-layer residual DELTA at the LAST token:
         delta[layer] = residual_after_layer - residual_after_prev_layer
     then difference against the baseline:  A = class0 - baseline,  B = class1 - baseline.
  2. Per layer, eigendecompose the symmetrised cross-covariance  C = (A^T B + B^T A)/2  (rows are the
     eigenvectors, ascending eigenvalue), keep the eigenvectors whose between/within discriminant
     ratio clears --tol, and greedily compound the strongest ones into one unit direction.
  3. De-bias by the midpoint of the two class means on that direction:
         debias = midpoint * u,  class0 = (mean_A - midpoint) * u,  class1 = (mean_B - midpoint) * u
     so class0 ~= -class1 and the concept vectors are RELATIVE to the debias vector.

The prompts go through the engine's own renderer (serve/ds41_encoding.py's encode_messages - the
server's pinned format), in chat or thinking mode (--thinking, default chat; the server's default
regime is thinking).  Extraction runs the engine's CVEXTRACT (the decode path, the stream means a
vector would steer): row 0 the embedding, row l + 1 the residual after the 0-based layer l's MoE
write.

EXPORT (three GGUFs per trait, this repo's control-vector format)
-----------------------------------------------------------------
  <out>/debias.gguf            the midpoint correction (always loaded with a concept)
  <out>/<class0>.gguf          steer toward class0   (e.g. compassion)
  <out>/<class1>.gguf          steer toward class1   (e.g. sadism)  ~= negative of the above
Each is `general.architecture="controlvector"`, `controlvector.model_hint="deepseek-v41"`,
`controlvector.layer_count`, and one `direction.<L>` F32[hidden_size] per steered layer - L the
0-BASED layer index (this repo's engines' convention, as --ablate's: layer 0 has no direction, so
the first layer's analysis is not exported; a direction past the model is ignored by the loader).
The concept GGUFs are RELATIVE to the debias GGUF - the engine SUMS them, so you always load the
debias together with the concept you want (see docs/CVECTOR.md).

SUBCOMMANDS
-----------
  tokenize   print what a sample tokenizes to (debugging the prompt assembly)
  extract    run the engine's CVEXTRACT over the sampled triplets -> A.npy, B.npy, meta.json (GPU)
  create     analyze A.npy/B.npy and export the three GGUFs (CPU, fast)
  run        extract + create

The datasets under tools/cvec/datasets/ are Apache-2.0 from jukofyork/control-vectors; credit and
license are preserved there (see docs/CVECTOR.md).

EXTRACTION COST
---------------
One forward pass over each sampled prompt per class.  A small --num-samples is for verification on
one GPU; a full run (jukofyork used ~10000 prompts) is meant to run on a spare machine.  `extract`
is resumable (re-run it and it continues where it left off).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

TOOLS = pathlib.Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO / "serve"))
from strata_tokenizer import Tokenizer  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from ds41_encoding import encode_messages  # noqa: E402

DATASETS = TOOLS / "cvec" / "datasets"
MODEL_HINT = "deepseek-v41"
THINKING_MODES = ("chat", "thinking")


# -------------------------------------------------------------------------- dataset (jukofyork DatasetManager)
def load_stems(path: pathlib.Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "pre" not in d or "post" not in d:
        raise ValueError(f"{path} must have 'pre' and 'post' lists")
    return d


def load_continuations(path: pathlib.Path) -> tuple[list[str], list[list[str]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not d or "classes" not in d or "data" not in d:
        raise ValueError(f"{path} must have 'classes' and 'data'")
    return list(d["classes"]), [list(row) for row in d["data"]]


def load_prompts(path: pathlib.Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"{path} is empty")
    return lines


def resolve_trait(trait: str) -> pathlib.Path:
    """A trait path: absolute/relative as given, or a name under tools/cvec/datasets/ (a bare filename is
    searched recursively, so 'compassion_vs_sadism.json' works too)."""
    p = pathlib.Path(trait)
    if p.exists():
        return p
    cand = DATASETS / trait
    if cand.exists():
        return cand
    if "/" not in trait:
        hits = sorted(DATASETS.rglob(trait))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(f"trait '{trait}' is ambiguous: {hits}")
    raise FileNotFoundError(f"trait not found: {trait} (looked in CWD and {DATASETS})")


def generate_triplets(stems, cont_rows, prompts, num_triplets: int, seed: int
                      ) -> list[tuple[str, str, list[str], str]]:
    """One triplet = (pre, post, [class0_cont, class1_cont], writing_prompt).  Same stem + prompt for the
    whole triplet, matched continuations from one data row - exactly jukofyork's `_generate_datasets`."""
    rng = random.Random(seed)
    out = []
    for _ in range(num_triplets):
        pre = rng.choice(stems["pre"])
        post = rng.choice(stems["post"])
        row = rng.choice(cont_rows)
        wp = rng.choice(prompts)
        out.append((pre, post, row, wp))
    return out


def class_text(triplet, class_index: int) -> str:
    """The user-message content for class_index (0=baseline, 1=class0, 2=class1) = system_message + ' ' +
    writing_prompt, matching jukofyork's `use_separate_system_message=False`."""
    pre, post, row, wp = triplet
    stem = f"{pre} {post}"
    if class_index == 0:
        system_message = stem + "."
    else:
        system_message = f"{stem} {row[class_index - 1]}."
    return f"{system_message} {wp}"


# -------------------------------------------------------------------------- renderer + tokenizer + geometry
def make_renderer(thinking_mode: str):
    """The server's own prompt format (serve/ds41_encoding.py): one user message, no tools."""
    if thinking_mode not in THINKING_MODES:
        raise ValueError(f"--thinking must be one of {THINKING_MODES}")

    def render(content: str) -> str:
        return encode_messages([{"role": "user", "content": content}], thinking_mode=thinking_mode)
    return render


def load_tokenizer(tokenizer_path: pathlib.Path) -> Tokenizer:
    p = pathlib.Path(tokenizer_path)
    if p.is_dir():
        return Tokenizer.from_dir(p)
    if (p / "tokenizer").is_dir():
        return Tokenizer.from_dir(p / "tokenizer")
    raise SystemExit(f"tokenizer not found: {p} (a pack's tokenizer dir, or a directory containing one)")


def read_geometry(pack: pathlib.Path) -> tuple[int, int]:
    """(hidden_size, trunk num_hidden_layers) from the pack's ds4.txt (the engine's geometry, under
    config.json's names)."""
    path = pathlib.Path(pack) / "ds4.txt"
    if not path.is_file():
        raise SystemExit(f"{path} is missing - is --pack the engine's pack directory?")
    geo: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " " in line:
            k, v = line.split(" ", 1)
            geo[k] = v
    if "hidden_size" not in geo or "num_hidden_layers" not in geo:
        raise SystemExit(f"{path} has no hidden_size / num_hidden_layers")
    return int(geo["hidden_size"]), int(geo["num_hidden_layers"])


# -------------------------------------------------------------------------- extraction (GPU)
def _run_cvextract(proc, tmpfile: str, ids: list[int]) -> None:
    proc.stdin.write(f"CVEXTRACT {tmpfile} {','.join(map(str, ids))}\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if line.startswith("CVOK"):
            return
        if line.startswith("ERR"):
            raise SystemExit(f"CVEXTRACT failed: {line.strip()}")
        if not line.strip():
            continue
    raise SystemExit("engine closed the pipe during CVEXTRACT")


def _open_samples(out: pathlib.Path, T: int, n_layers: int, n_embd: int, resume: bool):
    # Open (or create) the raw A/B memmaps for the samples dir.  A fresh run maps them "w+" - which
    # TRUNCATES an existing file - so a resume must map them "r+" instead, keeping the extracted rows
    # (otherwise the already-done triplets are wiped to zero and then skipped).  A resume first checks
    # that meta.json and the files belong to THIS run (same T / n_layers / n_embd, files the right
    # size), so it can never silently mix rows from two different runs.
    meta_path = out / "meta.json"
    start, mode = 0, "w+"
    if resume and meta_path.exists():
        old = json.loads(meta_path.read_text())
        start = int(old.get("done", 0))
        bad = [(k, old.get(k), v) for k, v in (("T", T), ("n_layers", n_layers), ("n_embd", n_embd))
               if old.get(k) != v]
        if bad:
            raise SystemExit(f"cannot resume: {meta_path.name} records "
                             + ", ".join(f"{k}={o}" for k, o, _ in bad) + " but this run is "
                             + ", ".join(f"{k}={v}" for k, _, v in bad)
                             + " - a different run; delete the samples dir and re-run without --resume")
        want = T * n_layers * n_embd * np.dtype(np.float32).itemsize
        exist = [nm for nm in ("A", "B") if (out / f"{nm}.npy").exists()]
        for nm in exist:
            if (out / f"{nm}.npy").stat().st_size != want:
                raise SystemExit(f"cannot resume: {nm}.npy in {out} is "
                                 f"{(out / f'{nm}.npy').stat().st_size} bytes, not {want} (the size for shape "
                                 f"({T}, {n_layers}, {n_embd})) - delete the samples dir and re-run "
                                 f"without --resume")
        if len(exist) == 2:
            mode = "r+"
            print(f"resuming: {start}/{T} already extracted")
        else:
            start = 0
            print(f"resume requested but A.npy / B.npy are incomplete ({', '.join(exist) or 'none'}) "
                  f"- starting fresh")
    A = np.memmap(out / "A.npy", dtype=np.float32, mode=mode, shape=(T, n_layers, n_embd))
    B = np.memmap(out / "B.npy", dtype=np.float32, mode=mode, shape=(T, n_layers, n_embd))
    return A, B, start, meta_path


def extract(a: argparse.Namespace, triplets, class_names, renderer, tokenizer,
            n_embd: int, n_layers: int) -> None:
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = len(triplets)
    A, B, start, meta_path = _open_samples(out, T, n_layers, n_embd, a.resume)
    engine_args = [a.exe, "--serve", "--pack", str(a.pack), *a.engine_arg]
    p = subprocess.Popen(engine_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         bufsize=1)
    ready = False
    for line in p.stdout:
        if line.startswith("READY"):
            ready = True
            break
        if line.startswith("ERR"):
            raise SystemExit(f"engine start failed: {line.strip()}")
    if not ready:
        p.kill()
        raise SystemExit("the engine did not report READY")
    t0 = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin") as tf:
            for i in range(start, T):
                deltas = []
                for ci in range(3):
                    content = class_text(triplets[i], ci)
                    ids = tokenizer.encode(renderer(content), parse_special=True)
                    _run_cvextract(p, tf.name, ids)
                    tf.flush()
                    rows = np.fromfile(tf.name, dtype=np.float32)
                    if rows.size != (n_layers + 1) * n_embd:
                        raise SystemExit(f"CVEXTRACT wrote {rows.size} floats, expected {(n_layers + 1)*n_embd} "
                                         f"(n_layers={n_layers}, n_embd={n_embd})")
                    rows = rows.reshape(n_layers + 1, n_embd)
                    deltas.append((rows[1:] - rows[:-1]).astype(np.float32))
                A[i] = deltas[1] - deltas[0]      # class0 - baseline
                B[i] = deltas[2] - deltas[0]      # class1 - baseline
                A.flush()
                B.flush()
                if (i - start + 1) % max(1, a.progress) == 0 or i + 1 == T:
                    meta_path.write_text(json.dumps({"done": i + 1, "T": T, "n_layers": n_layers,
                                                     "n_embd": n_embd,
                                                     "class_names": class_names,
                                                     "seed": a.seed, "trait": a.trait}))
                    el = time.time() - t0
                    print(f"  {i+1}/{T} triplets in {el:.0f} s ({el/(i+1-start):.2f} s each)", flush=True)
    finally:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=120)
    A.flush()
    B.flush()
    del A, B
    print(f"extraction done -> {out}/A.npy, {out}/B.npy")


# -------------------------------------------------------------------------- analysis (jukofyork DirectionAnalyzer)
def sym_cross_cov_eigenvectors(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Eigenvectors of (A^T B + B^T A)/2 as ROWS, ascending eigenvalue (numpy eigh, like torch.linalg.eigh)."""
    ATB = A.T @ B
    C = (ATB + ATB.T) / 2.0
    _, V = np.linalg.eigh(C)      # V: columns are eigenvectors, ascending eigenvalue
    return V.T                    # rows are eigenvectors


def _project(data: np.ndarray, direction: np.ndarray) -> np.ndarray:
    d = direction / (np.linalg.norm(direction) or 1.0)
    return data @ d


def _discriminant_ratio(pa: np.ndarray, pb: np.ndarray) -> float:
    m1, m2 = pa.mean(), pb.mean()
    overall = np.concatenate([pa, pb]).mean()
    n1, n2 = pa.size, pb.size
    between = n1 * (m1 - overall) ** 2 + n2 * (m2 - overall) ** 2
    within = ((pa - m1) ** 2).sum() + ((pb - m2) ** 2).sum()
    return float(between / within) if within != 0 else 0.0


def analyze_layer(A: np.ndarray, B: np.ndarray, tol: float):
    """One layer's (S,E) class0-minus-baseline and class1-minus-baseline matrices -> (debias, c0, c1) vectors
    (each (E,)) or None if no direction cleared the greedy compound."""
    directions = sym_cross_cov_eigenvectors(A, B)          # (E,E), rows
    results = []
    for i in range(directions.shape[0]):
        direction = directions[i]
        proj = [_project(A, direction), _project(B, direction)]
        ratio = _discriminant_ratio(proj[0], proj[1])
        if ratio >= tol:
            scaled = float(proj[1].mean()) * direction     # the RAW (un-normalised) eigenvector, as jukofyork
            results.append((ratio, scaled))
    results.sort(key=lambda x: x[0], reverse=True)
    best_ratio = 0.0
    best_sum = np.zeros(directions.shape[1])
    best_means = (0.0, 0.0)
    selected = 0
    for _ratio, scaled in results:
        direction_sum = best_sum + scaled
        norm = np.linalg.norm(direction_sum)
        if norm == 0:
            continue
        direction = direction_sum / norm
        proj = [_project(A, direction), _project(B, direction)]
        ratio = _discriminant_ratio(proj[0], proj[1])
        if ratio > best_ratio + tol:
            best_ratio = ratio
            best_sum = direction_sum
            best_means = (float(proj[0].mean()), float(proj[1].mean()))
            selected += 1
    if selected == 0:
        return None
    midpoint = (best_means[0] + best_means[1]) / 2.0
    u = best_sum / np.linalg.norm(best_sum)
    return (midpoint * u, (best_means[0] - midpoint) * u, (best_means[1] - midpoint) * u)


def analyze(A: np.ndarray, B: np.ndarray, n_layers: int, start_layer: int,
            skip_end: int, tol: float) -> list[list[np.ndarray | None]]:
    """directions[type][layer] for type in {0:debias, 1:class0, 2:class1}; None where a layer was skipped."""
    directions = [[None] * n_layers for _ in range(3)]
    for l in range(start_layer, n_layers - skip_end):
        A_l = A[:, l, :].astype(np.float64)
        B_l = B[:, l, :].astype(np.float64)
        got = analyze_layer(A_l, B_l, tol)
        if got is not None:
            for t in range(3):
                directions[t][l] = got[t].astype(np.float32)
    return directions


# -------------------------------------------------------------------------- export
def export_gguf(directions: list[np.ndarray | None], n_layers: int, n_embd: int,
                model_hint: str, path: pathlib.Path) -> bool:
    """directions[layer] (0-BASED) -> one direction.<layer> F32[n_embd] per non-None layer, layer 0's left
    out (this repo's engines do not steer the first layer: no direction.0, and the loader's default range
    starts at layer 1)."""
    w = GGUFWriter()
    w.add("general.architecture", "controlvector")
    w.add("controlvector.model_hint", model_hint)
    w.add("controlvector.layer_count", int(n_layers), "u32")
    written = 0
    for layer, vec in enumerate(directions):
        if vec is None or layer < 1:
            continue
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size != n_embd:
            raise ValueError(f"direction for layer {layer} is {v.size}, expected {n_embd}")
        w.add_f32(f"direction.{layer}", v)              # 0-based layer index, as the engine's loaders read it
        written += 1
    if written == 0:
        return False
    w.write(path)
    return True


def _load_samples(samples: pathlib.Path, name: str, meta: dict) -> np.memmap:
    # A.npy / B.npy are RAW float32 memmaps (no .npy header): the extraction writes them one triplet at a
    # time (resumable), so np.load cannot read them - they must be mapped back with the shape meta.json
    # recorded.  A size check catches a stale file from a different run (different --num-samples).
    T, n_layers, n_embd = meta["T"], meta["n_layers"], meta["n_embd"]
    path = samples / f"{name}.npy"
    expected = (T * n_layers * n_embd) * np.dtype(np.float32).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(f"{path.name} is {actual} bytes but meta.json describes shape "
                         f"({T}, {n_layers}, {n_embd}) = {expected} bytes - a stale file from a different "
                         f"run; re-run `extract` (or `run`) so A/B and meta.json agree")
    return np.memmap(path, dtype=np.float32, mode="r", shape=(T, n_layers, n_embd))


def create(a: argparse.Namespace, class_names: list[str]) -> None:
    meta_path = a.samples / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} is missing - run `extract` (or `run`) first")
    meta = json.loads(meta_path.read_text())
    if meta.get("done", 0) < meta["T"]:
        raise SystemExit(f"extraction is incomplete ({meta['done']}/{meta['T']} triplets) - resume it "
                         f"with `extract --resume` before `create`")
    A = _load_samples(a.samples, "A", meta)
    B = _load_samples(a.samples, "B", meta)
    n_layers, n_embd = A.shape[1], A.shape[2]
    print(f"A/B: {A.shape} (num_triplets, n_layers, n_embd)")
    directions = analyze(A, B, n_layers, a.start_layer, a.skip_end, a.tol)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = ["debias", class_names[0], class_names[1]]
    for t, nm in enumerate(names):
        ok = export_gguf(directions[t], n_layers, n_embd, a.model_hint, out / f"{nm}.gguf")
        nvec = sum(1 for l, v in enumerate(directions[t]) if v is not None and l >= 1)
        print(f"  {nm}.gguf: {'wrote' if ok else 'SKIPPED (no layers)'} {nvec} layer direction(s)")
    print(f"done -> {out}")


# -------------------------------------------------------------------------- CLI
def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--num-samples", type=int, default=64,
                    help="number of triplets (== samples per class); jukofyork's num_prompt_samples ~ 3x this")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tol", type=float, default=0.5, help="discriminant-ratio tolerance (jukofyork default 0.5)")
    ap.add_argument("--start-layer", type=int, default=0, help="first layer index to analyse (default 0)")
    ap.add_argument("--skip-end", type=int, default=1, help="trailing layers to skip (default 1, jukofyork)")
    ap.add_argument("--model-hint", default=MODEL_HINT)
    ap.add_argument("--thinking", choices=THINKING_MODES, default="chat",
                    help="the prompt's regime (default chat; the server's default regime is thinking - match it "
                         "with --thinking thinking to steer a default-serve conversation)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tokenize", help="print the tokenized prompt for a sample triplet (debug)")
    p.add_argument("--trait", required=True, help="a continuations JSON under tools/cvec/datasets/")
    p.add_argument("--stems", default=str(DATASETS / "prompt_stems.json"))
    p.add_argument("--prompts", default=str(DATASETS / "writing_prompts.txt"))
    p.add_argument("--pack", required=True, help="the engine's pack directory (tokenizer + geometry)")
    p.add_argument("--tokenizer", default=None, help="override the pack's tokenizer dir")
    p.add_argument("--thinking", choices=THINKING_MODES, default="chat")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--show", type=int, default=1, help="how many triplets to show")

    p = sub.add_parser("extract", help="run the engine over the sampled triplets (GPU)")
    _add_common(p)
    p.add_argument("--trait", required=True)
    p.add_argument("--stems", default=str(DATASETS / "prompt_stems.json"))
    p.add_argument("--prompts", default=str(DATASETS / "writing_prompts.txt"))
    p.add_argument("--pack", required=True, help="the engine's pack directory (tokenizer + geometry)")
    p.add_argument("--tokenizer", default=None, help="override the pack's tokenizer dir")
    p.add_argument("--exe", default=str(REPO / "build" / "maya-ds4"))
    p.add_argument("--out", required=True, help="directory for A.npy / B.npy / meta.json")
    p.add_argument("--progress", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--engine-arg", action="append", default=[], help="extra maya-ds4 args, repeatable")

    p = sub.add_parser("create", help="analyze the extracted A/B and export the three GGUFs (CPU)")
    _add_common(p)
    p.add_argument("--samples", required=True, help="directory holding A.npy / B.npy / meta.json")
    p.add_argument("--out", required=True, help="directory for the .gguf outputs")

    p = sub.add_parser("run", help="extract then create")
    _add_common(p)
    p.add_argument("--trait", required=True)
    p.add_argument("--stems", default=str(DATASETS / "prompt_stems.json"))
    p.add_argument("--prompts", default=str(DATASETS / "writing_prompts.txt"))
    p.add_argument("--pack", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--exe", default=str(REPO / "build" / "maya-ds4"))
    p.add_argument("--out", required=True, help="output dir (A/B go to <out>/samples, GGUFs to <out>)")
    p.add_argument("--progress", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--engine-arg", action="append", default=[])

    a = ap.parse_args()

    if a.cmd == "tokenize":
        stems = load_stems(pathlib.Path(a.stems))
        classes, rows = load_continuations(resolve_trait(a.trait))
        prompts = load_prompts(pathlib.Path(a.prompts))
        triplets = generate_triplets(stems, rows, prompts, a.show, a.seed)
        renderer = make_renderer(a.thinking)
        tok = load_tokenizer(pathlib.Path(a.tokenizer or str(pathlib.Path(a.pack) / "tokenizer")))
        for i, t in enumerate(triplets):
            for ci, nm in enumerate(["baseline", classes[0], classes[1]]):
                content = class_text(t, ci)
                ids = tok.encode(renderer(content), parse_special=True)
                print(f"[triplet {i}] {nm}: {len(ids)} tokens")
                print(f"    text: {content[:160]!r}")
        return 0

    # extract / create / run
    trait_path = resolve_trait(a.trait)
    classes, rows = load_continuations(trait_path)
    stems = load_stems(pathlib.Path(a.stems))
    prompts = load_prompts(pathlib.Path(a.prompts))
    triplets = generate_triplets(stems, rows, prompts, a.num_samples, a.seed)
    print(f"trait '{trait_path.name}': classes {classes}, {len(triplets)} triplets")

    # where the A/B samples live and where the GGUFs go, per subcommand
    if a.cmd == "extract":
        samples_dir, gguf_out = pathlib.Path(a.out), pathlib.Path(a.out)
    elif a.cmd == "create":
        samples_dir, gguf_out = pathlib.Path(a.samples), pathlib.Path(a.out)
    else:  # run
        samples_dir, gguf_out = pathlib.Path(a.out) / "samples", pathlib.Path(a.out)

    if a.cmd in ("extract", "run"):
        renderer = make_renderer(a.thinking)
        tok = load_tokenizer(pathlib.Path(a.tokenizer or str(pathlib.Path(a.pack) / "tokenizer")))
        n_embd, n_layers = read_geometry(pathlib.Path(a.pack))
        print(f"geometry: n_embd={n_embd} trunk n_layers={n_layers}")
        extract(argparse.Namespace(**{**vars(a), "out": str(samples_dir)}),
                triplets, classes, renderer, tok, n_embd, n_layers)

    if a.cmd in ("create", "run"):
        meta = samples_dir / "meta.json"
        cn = json.loads(meta.read_text())["class_names"] if meta.exists() else classes
        create(argparse.Namespace(**{**vars(a), "samples": samples_dir, "out": str(gguf_out)}), cn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
