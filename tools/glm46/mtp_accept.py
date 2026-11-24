"""tools/glm46/mtp_accept.py - the verify / commit / chain path on a model whose MTP block drafts the trunk's own
next token (tools/glm46/glm46_synth_gguf.py --mtp-identity).

    python tools/glm46/mtp_accept.py [--model DIR] [--ids 1,2,3,4,5,6,7] [--gen 24]

The synthetic output is highly repetitive, so the identity block's drafts (the trunk's next-token argmax read one
step early) are accepted: the engine's rounds then keep a prefix longer than one row, `commit_rows` moves the
sequence's carrier to the last kept row and the chain after it reads the block's own output - the parts a random
block (mtp_parity.py's model) never exercises.  Two checks:

  * `--mtp on` emits exactly the tokens `--mtp off` does (the verify window decides every one: drafting may only
    change the speed), for --mtp-drafts 1 .. 3;
  * at least one draft is kept (the accepted / committed path ran).

The block's math itself is checked against ref/glm46.py by tools/glm46/mtp_parity.py.
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


def run(exe: pathlib.Path, pack: pathlib.Path, ids: str, gen: int, extra: list[str]):
    cmd = [str(exe), "--pack", str(pack), "--ids", ids, "--gen", str(gen), "--max-context", "512", *extra]
    env = dict(os.environ)
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit(f"maya-glm46 failed ({' '.join(extra)}):\n" + r.stderr[-2500:])
    out = [ln for ln in r.stdout.splitlines() if re.fullmatch(r"-?\d+( -?\d+)*", ln.strip())]
    tokens = out[-1].strip() if out else ""
    kept = total = 0
    for ln in r.stderr.splitlines():
        m = re.search(r"(\d+) of (\d+) drafts kept", ln)
        if m:
            kept, total = int(m.group(1)), int(m.group(2))
    return tokens, kept, total, r.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description="the MTP verify / commit path on the identity-block synthetic model")
    ap.add_argument("--model", default="", help="the model directory (built under the temp dir by default)")
    ap.add_argument("--exe", default=str(REPO / "build" / "maya-glm46"))
    ap.add_argument("--ids", default="1,2,3,4,5,6,7")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()

    model_dir = pathlib.Path(a.model) if a.model else pathlib.Path(tempfile.mkdtemp(prefix="glm46-mtpacc-"))
    gguf = model_dir / "glm46-synth.gguf"
    if not gguf.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        for cmd in ([sys.executable, str(HERE / "glm46_synth_gguf.py"), "--out", str(gguf), "--seed", str(a.seed),
                     "--mtp-identity"],
                    [sys.executable, str(HERE / "pack_gguf.py"), "--model", str(model_dir), "--no-tokenizer"]):
            if subprocess.run(cmd).returncode != 0:
                return 1
    pack = model_dir / "maya-glm46"
    exe = pathlib.Path(a.exe)
    base, _, _, _ = run(exe, pack, a.ids, a.gen, ["--mtp", "off"])
    failures = 0
    for n in (1, 2, 3, 8):
        got, kept, total, err = run(exe, pack, a.ids, a.gen, ["--mtp", "on", "--mtp-drafts", str(n)])
        # (a round may overshoot --gen by its kept drafts, as DSpark's rounds did: the first --gen tokens are the answer)
        same = got != "" and base != "" and got.split()[:a.gen] == base.split()[:a.gen]
        ok = same and kept > 0
        print(f"  drafts {n}: tokens {'same' if same else 'DIFFER'}, {kept} of {total} kept -> {'ok' if ok else 'FAIL'}")
        if not ok:
            failures += 1
            if not same:
                print(f"    --mtp off: {base}\n    --mtp on:  {got}")
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
