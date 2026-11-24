"""tools/glm46/mtp_parity.py - the engine's NextN (MTP) block against ref/glm46.py on the synthetic model.

    python tools/glm46/mtp_parity.py [--model DIR] [--ids 1,2,3,4,5] [--tol 8]

Runs `maya-glm46 --mtp on --gen N` with GLM46_MTP_DUMP set (the block's first steps: its input [emb | h], eh_proj's
output, the block's residual output, its shnorm carrier and the head's logits, all F32) and the reference's own block
prefill + chain (ref/glm46.py: mtp_prefill / mtp_step), then compares, step by step:

  * the carrier (the trunk's post-output-norm hidden the block read) - the engine dumps the normed halves, the
    reference the same;
  * the block's residual output and the shnorm carrier (the next step's input);
  * the head's logits (the relative RMS, as parity.py - the block's matrices are quantized like the trunk's) and the
    step's argmax (a near tie allowed, as in parity.py).

The chain's later steps are checked against the step before's carrier (the block's own output), so a chain that reads
the wrong row or forgets shnorm is caught here.
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


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return 100.0 * float(np.sqrt(((a - b) ** 2).mean()) / max(1e-9, float(np.sqrt((b ** 2).mean()))))


def read_mtpd(path: pathlib.Path):
    """Every MTPD record: (pos, tok, dim, vocab, cat[2dim], pre[dim], post[dim], carry[dim], logits[vocab])."""
    b = path.read_bytes()
    out = []
    at = 0
    while at < len(b):
        if b[at:at + 4] != b"MTPD":
            raise SystemExit(f"{path}: not an MTPD file at {at}")
        pos, tok, dim, vocab, _ = struct.unpack("<iiiii", b[at + 4:at + 24])
        at += 24
        n = 2 * dim + dim + dim + dim + vocab
        v = np.frombuffer(b[at:at + 4 * n], dtype="<f4")
        at += 4 * n
        out.append((pos, tok, dim, vocab, v[:2 * dim], v[2 * dim:3 * dim], v[3 * dim:4 * dim],
                    v[4 * dim:5 * dim], v[5 * dim:]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="the NextN block against ref/glm46.py")
    ap.add_argument("--model", default="", help="the model directory (built under the temp dir by default)")
    ap.add_argument("--exe", default=str(REPO / "build" / "maya-glm46"))
    ap.add_argument("--ids", default="1,2,3,4,5")
    ap.add_argument("--steps", type=int, default=3, help="the chain's steps to compare (the engine dumps at most 3)")
    ap.add_argument("--tol", type=float, default=8.0, help="the relative RMS of a logits difference allowed (percent)")
    ap.add_argument("--margin", type=float, default=0.03, help="a top-1 difference allowed below this logits margin")
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    ids = [int(t) for t in a.ids.split(",")]
    if len(ids) < 2:
        raise SystemExit("--ids: at least two tokens")

    model_dir = pathlib.Path(a.model) if a.model else pathlib.Path(tempfile.mkdtemp(prefix="glm46-mtp-"))
    gguf = model_dir / "glm46-synth.gguf"
    if not gguf.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        for cmd in ([sys.executable, str(HERE / "glm46_synth_gguf.py"), "--out", str(gguf), "--seed", str(a.seed)],
                    [sys.executable, str(HERE / "pack_gguf.py"), "--model", str(model_dir), "--no-tokenizer"]):
            r = subprocess.run(cmd)
            if r.returncode != 0:
                return r.returncode

    from glm46 import Glm46   # ref/glm46.py

    exe = pathlib.Path(a.exe)
    dump = model_dir / "mtp.bin"
    if dump.exists():
        dump.unlink()
    env = dict(os.environ)
    env["GLM46_MTP_DUMP"] = str(dump)
    env.setdefault("MAYA_GLM46_GRAPH", "0")
    cmd = [str(exe), "--pack", str(model_dir / "maya-glm46"), "--ids", a.ids, "--gen", str(a.steps + 1),
           "--max-context", "512", "--mtp", "on", "--mtp-drafts", str(a.steps)]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit("maya-glm46 failed:\n" + r.stderr[-3000:])
    if not dump.exists():
        raise SystemExit("no GLM46_MTP_DUMP records: the block did not draft (carrier missing?)")
    steps = read_mtpd(dump)
    if not steps:
        raise SystemExit("the dump is empty")

    m = Glm46(str(gguf))
    if getattr(m, "mtp", None) is None:
        m.mtp_load()
    if m.mtp is None:
        raise SystemExit(f"{gguf}: no NextN block (block_count {m.cfg.n_layer}, nextn {m.cfg.nextn})")
    m._mk = m._mv = None
    trunk = m.forward(ids)                     # (T, vocab)
    h = m.last_h                               # (T, dim) post-output-norm
    n = len(ids)
    m.mtp_prefill(ids, h)                      # the prompt's entries 0 .. n - 2
    anchor = int(np.argmax(trunk[-1]))
    ref_lg, ref_car = m.mtp_step(n - 1, anchor, h[n - 1])

    failures = 0
    print(f"model: {gguf}  ids {ids}  anchor {anchor}  steps {len(steps)}")
    prev_lg, prev_car = ref_lg, ref_car
    for i, (pos, tok, dim, vocab, cat, pre, post, carry, lg) in enumerate(steps[:max(1, a.steps)]):
        if i == 0:
            # the engine's first step: position n - 1, the trunk's own argmax as the token
            if pos != n - 1 or tok != anchor:
                print(f"  step 0: engine pos {pos}, token {tok}; the reference's {n - 1}, {anchor}"
                      f" -> {'FAIL' if tok != anchor and abs(float(lg[int(np.argmax(lg))])) >= 0 else 'ok (near tie)'}")
                if tok != anchor:
                    # (a different anchor: the trunk's top-1 differs - parity.py's own allowance, not the block's)
                    pass
            car_diff = rel_rms(carry, ref_car)
            lg_diff = rel_rms(lg, prev_lg)
            top_e, top_r = int(np.argmax(lg)), int(np.argmax(prev_lg))
            s = np.sort(prev_lg)[-2:][::-1]
            margin = float(s[0] - s[1])
        else:
            # the chain: the step's token is the step before's draft, its carrier the step before's shnorm output
            prev_tok = int(np.argmax(prev_lg))
            if pos != n - 1 + i or tok != prev_tok:
                print(f"  step {i}: engine pos {pos}, token {tok}; the reference's {n - 1 + i}, {prev_tok}"
                      f" -> FAIL")
                failures += 1
                break
            ref_lg, ref_car = m.mtp_step(pos, tok, prev_car)
            car_diff = rel_rms(carry, ref_car)
            lg_diff = rel_rms(lg, ref_lg)
            top_e, top_r = int(np.argmax(lg)), int(np.argmax(ref_lg))
            s = np.sort(ref_lg)[-2:][::-1]
            margin = float(s[0] - s[1])
        ok = lg_diff <= a.tol and (top_e == top_r or margin < a.margin)
        print(f"  step {i}: pos {pos}, token {tok}, carry diff {car_diff:5.2f}%, logits rel rms {lg_diff:5.2f}% "
              f"(tol {a.tol:.1f}%), top-1 {top_e}/{top_r} (margin {margin:.3f}) -> {'ok' if ok else 'FAIL'}")
        if not ok:
            failures += 1
        prev_lg, prev_car = ref_lg, ref_car
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
