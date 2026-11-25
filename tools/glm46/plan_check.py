"""tools/glm46/plan_check.py - the memory plan's two per-layer estimates against the pack itself.

    python tools/glm46/plan_check.py [--model DIR] [--exe build/maya-glm46]

`maya-glm46` sizes its layer split and its staging slots from, per layer, its dense bytes and one expert's blob
(plan_memory).  A GGUF pack keeps the routed experts in three stacked tensors (`ffn_gate_exps.weight`,
`ffn_up_exps.weight`, `ffn_down_exps.weight`); the fork's inherited estimates counted those as dense (a MoE layer's
~1.7 GB of experts as weights: the split then charged a card tens of GB too many and the start stopped with "not
enough GPU memory for the dense part") and read an EXL3 pack's tensor names for the blob (0 for a GGUF).  This runs
the engine with GLM46_PLAN_TRACE=1 and compares the trace with the pack's own tensor table:

  * dense: every `blk.*` tensor but the three stacked expert tensors, for all layers (the trace sums the plan's
    per-layer dense bytes);
  * one expert's blob: each stacked tensor's bytes / num_local_experts, rounded up to 256, summed.

Then the plan must fit the card (the engine loads and answers).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
ALIGN = 256


def pack_facts(pack: pathlib.Path) -> tuple[float, float, int]:
    """(dense bytes over all layers, one expert's blob bytes, layers) from tensors.txt and glm46.txt."""
    geom = {}
    for line in (pack / "glm46.txt").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition(" ")
        geom[k] = v
    n_exp = int(geom["num_local_experts"])
    dense = 0
    blob = {}
    layers = set()
    for line in (pack / "tensors.txt").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if not f or f[0] != "t":
            continue
        name, nbytes = f[1], int(f[5])
        m = re.match(r"blk\.(\d+)\.", name)
        if not m:
            continue
        layers.add(int(m.group(1)))
        role = re.search(r"(ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.", name)
        if role:
            per = (nbytes + n_exp - 1) // n_exp
            blob[role.group(1)] = (per + ALIGN - 1) // ALIGN * ALIGN
        else:
            dense += nbytes
    return dense, sum(blob.values()), len(layers)


def main() -> int:
    ap = argparse.ArgumentParser(description="the engine's plan estimates against the pack's tensors")
    ap.add_argument("--model", default="", help="the model directory (the synthetic one is built by default)")
    ap.add_argument("--exe", default=str(REPO / "build" / "maya-glm46"))
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()

    model = pathlib.Path(a.model) if a.model else pathlib.Path(tempfile.mkdtemp(prefix="glm46-plan-"))
    gguf = model / "glm46-synth.gguf"
    if not (model / "maya-glm46" / "tensors.txt").exists():
        model.mkdir(parents=True, exist_ok=True)
        for cmd in ([sys.executable, str(HERE / "glm46_synth_gguf.py"), "--out", str(gguf), "--seed", str(a.seed)],
                    [sys.executable, str(HERE / "pack_gguf.py"), "--model", str(model), "--no-tokenizer"]):
            if subprocess.run(cmd).returncode != 0:
                return 1
    pack = model / "maya-glm46"
    dense, blob, layers = pack_facts(pack)
    env = dict(os.environ, GLM46_PLAN_TRACE="1")
    r = subprocess.run([a.exe, "--pack", str(pack), "--ids", "1,2,3", "--gen", "0", "--max-context", "512"],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit("maya-glm46 did not start:\n" + r.stderr[-2500:])
    m = re.search(r"glm46 plan \((\d+) positions, (\d+) staging slots\): CUDA0: dense ([\d.]+) ", r.stderr)
    b = re.search(r"slots ([\d.]+) \(([\d.]+) MB each\)", r.stderr)
    if not m or not b:
        raise SystemExit("no plan trace in the engine's output:\n" + r.stderr[-1500:])
    got_dense, got_blob = float(m.group(3)) * (1 << 30), float(b.group(2)) * 1e6
    ok_d = abs(got_dense - dense) <= max(2e6, 0.01 * dense)          # (the trace rounds to 0.01 GB)
    ok_b = abs(got_blob - blob) <= max(0.02e6, 0.01 * blob)
    print(f"  pack: {pack} ({layers} layers, {m.group(1)} positions, {m.group(2)} slots)")
    print(f"  dense       engine {got_dense / 2**20:9.2f} MB, tensors {dense / 2**20:9.2f} MB -> {'ok' if ok_d else 'FAIL'}")
    print(f"  expert blob engine {got_blob / 2**20:9.2f} MB, tensors {blob / 2**20:9.2f} MB -> {'ok' if ok_b else 'FAIL'}")
    print("ALL PASS" if ok_d and ok_b else "FAILED")
    return 0 if ok_d and ok_b else 1


if __name__ == "__main__":
    sys.exit(main())
