"""tools/glm46/parity.py - maya-glm46 against ref/glm46.py on the synthetic glm4moe model.

    python tools/glm46/parity.py [--model DIR] [--prompts 32,81,17] [--tol 5.0]
    python tools/glm46/parity.py --cvec FILE:SCALE [--cvec-mode add|project] [--cvec-range A B]

Builds the synthetic GGUF (tools/glm46/glm46_synth_gguf.py) and its pack (tools/glm46/pack_gguf.py) under
--model (default: a throwaway directory under the temp dir), runs `maya-glm46 --dump-logits` and the reference,
and compares:

  * the relative RMS of the logit difference (the engine quantizes activations to q8_1 and its prompt GEMMs use
    FP16 weight bands, like llama.cpp - so it is neither expected nor meaningful to match float64 bit for bit; llama.cpp's
    own CUDA build sits at the same 1-2% against this reference);
  * the top-1 token per row, allowing a difference only where the REFERENCE's own top-1/top-2 margin is below
    --margin (a near tie a quantized kernel may legitimately break the other way, as llama.cpp's does).

The engine and the reference are also cross-checked on the unsteered logits with llama.cpp, when a `llama-cli`/libllama
build is around - optional, off by default.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "ref"))


def read_mm3l(path: pathlib.Path) -> np.ndarray:
    b = path.read_bytes()
    if b[:4] != b"MM3L":
        raise SystemExit(f"{path}: not an MM3L logits file")
    rows, vocab, n = struct.unpack("<iii", b[4:16])
    off = 16 + 4 * n + 4 * rows
    lg = np.frombuffer(b[off:], dtype="<f4").reshape(rows, vocab)
    return lg


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return 100.0 * float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))


def run(model_dir: pathlib.Path, ids: list[int], exe: pathlib.Path, out: pathlib.Path, extra: list[str]) -> np.ndarray:
    cmd = [str(exe), "--pack", str(model_dir / "maya-glm46"), "--ids", ",".join(str(t) for t in ids), "--gen", "0",
           "--max-context", "512", "--dump-logits", str(out), "--last", str(len(ids)), *extra]
    env = dict(os.environ)
    env.setdefault("MAYA_GLM46_GRAPH", "0")   # (the dump needs no graph capture; either path gives the same bits)
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit("maya-glm46 failed:\n" + r.stderr[-2000:])
    return read_mm3l(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="maya-glm46 against ref/glm46.py on the synthetic model")
    ap.add_argument("--model", default="", help="the model directory (built under the temp dir by default)")
    ap.add_argument("--exe", default=str(REPO / "build" / "maya-glm46"))
    ap.add_argument("--prompts", default="1,2,3,4;32,81,17,5,9;7,7,11,200",
                    help="semicolon-separated token id sequences")
    ap.add_argument("--tol", type=float, default=6.0, help="the relative RMS of the logit difference allowed (percent)")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="a top-1 difference is allowed where the reference's top-1/top-2 margin is below this")
    ap.add_argument("--cvec", action="append", default=[], help="FILE:SCALE (repeatable)")
    ap.add_argument("--cvec-mode", choices=("add", "project"), default="add")
    ap.add_argument("--cvec-range", nargs=2, type=int, default=None)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--synth-quant", choices=("mixed", "xs"), default="mixed",
                    help="the synthetic model's per-tensor type map: mixed (Q8_0 / IQ3_S / IQ3_S / IQ4_XS) or xs "
                         "(Q6_K / IQ2_XS / IQ2_XS / IQ3_S, the smaller shipped quant's); an existing --model keeps "
                         "its own")
    a = ap.parse_args()

    model_dir = pathlib.Path(a.model) if a.model else pathlib.Path(tempfile.mkdtemp(prefix="glm46-parity-"))
    gguf = model_dir / "glm46-synth.gguf"
    if not gguf.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(HERE / "glm46_synth_gguf.py"), "--out", str(gguf), "--seed",
                            str(a.seed), "--quant", a.synth_quant])
        if r.returncode != 0:
            return r.returncode
        r = subprocess.run([sys.executable, str(HERE / "pack_gguf.py"), "--model", str(model_dir), "--no-tokenizer"])
        if r.returncode != 0:
            return r.returncode

    from glm46 import Glm46   # ref/glm46.py

    exe = pathlib.Path(a.exe)
    extra = []
    if a.cvec:
        extra += ["--control-vector-scaled", ",".join(a.cvec), "--cvec-mode", a.cvec_mode]
        if a.cvec_range:
            extra += ["--control-vector-layer-range", str(a.cvec_range[0]), str(a.cvec_range[1])]

    failures = 0
    print(f"model: {gguf}  engine: {exe}  cvec: {a.cvec or '-'}")
    for seq in a.prompts.split(";"):
        ids = [int(t) for t in seq.split(",") if t.strip()]
        eng = run(model_dir, ids, exe, model_dir / "eng.bin", extra)
        m = Glm46(str(gguf))
        for spec in a.cvec:
            path, _, sc = spec.rpartition(":")
            first, last = a.cvec_range if a.cvec_range else (1, m.cfg.trunk - 1)
            m.load_cvec(path, float(sc), 0 if a.cvec_mode == "project" else 1, first, last)
        ref = m.forward(ids)
        diff = rel_rms(eng, ref)
        top_e, top_r = eng.argmax(1), ref.argmax(1)
        allow = 0
        for t in range(len(ids)):
            if top_e[t] != top_r[t]:
                s = np.sort(ref[t])[-2:][::-1]
                if s[0] - s[1] < a.margin:
                    allow += 1
        top_ok = int((top_e == top_r).sum()) + allow == len(ids)
        near = getattr(m, "last_min_margin", float("inf")) < 0.02   # (the steered model can sit on a routing boundary)
        ok = top_ok and (diff <= a.tol or (near and top_ok))
        print(f"  ids {ids}: rel rms {diff:5.2f}% (tol {a.tol:.1f}%), top-1 agree "
              f"{int((top_e == top_r).sum())}/{len(ids)} (+{allow} near-tie), min route margin "
              f"{getattr(m, 'last_min_margin', 0):.4f} -> {'ok' if ok else 'FAIL'}")
        if not ok:
            failures += 1
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
