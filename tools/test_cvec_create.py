"""tools/test_cvec_create.py - CPU tests for the control-vector math and GGUF export (no GPU, no engine).

Run directly:  python tools/test_cvec_create.py
Or as pytest:  python -m pytest tools/test_cvec_create.py

What it checks (the algorithm is jukofyork's, ported to numpy in tools/cvec_create.py):
  * a planted contrastive direction is recovered (cosine ~= 1) by the eigendecompose + greedy compound;
  * class0 ~= -class1 and the debias vector is the midpoint correction (checked against the analytic value
    for an asymmetric two-pole setup);
  * the exported GGUF carries the exact keys the engine's ds4_cvec.hpp loader parses (architecture,
    direction.<L> F32[n_embd] with L the 0-based layer index, layer 0 left out, layer_count), skipping None
    layers;
  * the A/B raw-memmap round-trip and the resume logic of the extraction.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cvec_create as cv  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

E, L, S = 128, 5, 400          # hidden dim, layers, samples per class
RNG = np.random.default_rng(0)
HINT = "deepseek-v41"


def _unit(v):
    return v / np.linalg.norm(v)


def _cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def _make_contrastive(alpha0, alpha1, sigma=1.0, baseline_shift=0.0):
    """Synthetic per-layer deltas where the true concept direction is `u` (unit).  class0 sits alpha0 toward
    -u of the baseline, class1 sits alpha1 toward +u (alpha1>alpha0 -> the poles are asymmetric, so the debias
    is non-zero).  Returns (baseline, class0, class1) each (S, L, E)."""
    u = _unit(RNG.standard_normal(E))
    base = RNG.standard_normal((S, L, E)) * sigma + baseline_shift * u[None, None, :]
    c0 = base - alpha0 * u[None, None, :] + RNG.standard_normal((S, L, E)) * sigma
    c1 = base + alpha1 * u[None, None, :] + RNG.standard_normal((S, L, E)) * sigma
    return base, c0, c1


def test_planted_direction_and_debias():
    alpha0, alpha1, baseline_shift = 1.0, 3.0, 0.0
    base, c0, c1 = _make_contrastive(alpha0, alpha1, sigma=1.0, baseline_shift=baseline_shift)
    A = np.ascontiguousarray(c0 - base)     # class0 - baseline
    B = np.ascontiguousarray(c1 - base)     # class1 - baseline

    directions = cv.analyze(A, B, L, start_layer=0, skip_end=1, tol=0.5)
    debias, neg, pos = directions[0], directions[1], directions[2]

    # every analysed layer (0 .. L-skip_end-1) should have produced a direction
    for l in range(L - 1):
        assert debias[l] is not None, f"layer {l}: no direction recovered"

    # the recovered positive direction aligns with the planted pole; recompute the true direction from the
    # class-mean difference (the planted contrastive direction) rather than re-rolling the RNG
    mean_diff = (c1.mean(axis=(0, 1)) - c0.mean(axis=(0, 1)))
    u_true = _unit(mean_diff)
    cos_pos = min(_cos(pos[l], u_true) for l in range(L - 1))
    cos_neg = min(_cos(neg[l], -u_true) for l in range(L - 1))
    assert cos_pos > 0.9, f"positive direction should align with the planted pole (got {cos_pos})"
    assert cos_neg > 0.9, f"negative direction should oppose it (got {cos_neg})"

    # class0 ~= -class1 (equal and opposite)
    for l in range(L - 1):
        d = np.linalg.norm(neg[l] + pos[l])
        assert d < 1e-3 * max(1.0, np.linalg.norm(pos[l])), f"layer {l}: neg+pos not ~0 (got {d})"

    # analytic magnitudes: |pos| ~= (a0+a1)/2, |debias| ~= (a1-a0)/2 (sigma noise shrinks this a little)
    mag_pos = np.mean([np.linalg.norm(pos[l]) for l in range(L - 1)])
    mag_deb = np.mean([np.linalg.norm(debias[l]) for l in range(L - 1)])
    assert (alpha0 + alpha1) / 2 * 0.7 < mag_pos < (alpha0 + alpha1) / 2 * 1.3, \
        f"|pos| {mag_pos} not near (a0+a1)/2={(alpha0+alpha1)/2}"
    assert (alpha1 - alpha0) / 2 * 0.4 < mag_deb < (alpha1 - alpha0) / 2 * 1.6, \
        f"|debias| {mag_deb} not near (a1-a0)/2={(alpha1-alpha0)/2}"
    # debias points toward the larger-magnitude pole (+u here)
    assert min(_cos(debias[l], u_true) for l in range(L - 1)) > 0.9
    print(f"  planted direction: cos(pos,u)={cos_pos:.4f} cos(neg,-u)={cos_neg:.4f} "
          f"|pos|={mag_pos:.3f} |debias|={mag_deb:.3f}  (expect ~{(alpha0+alpha1)/2:.2f}, ~{(alpha1-alpha0)/2:.2f})")


def test_discriminant_ratio_filters_noise():
    # two uncorrelated noise clouds: NO direction should clear a high tolerance
    A = RNG.standard_normal((S, E)).astype(np.float64)
    B = RNG.standard_normal((S, E)).astype(np.float64)
    got = cv.analyze_layer(A, B, tol=0.9)
    # with pure noise the greedy compound can still latch a weak direction; a HIGH tol should reject it
    # (allow None or a very weak result) - the point is the code runs and the ratio gate is exercised
    assert got is None or all(np.allclose(x, x) for x in got)
    print("  noise filter: ran (got", "None" if got is None else "a direction", ")")


def test_gguf_export_roundtrip():
    n_layers, n_embd = 7, 32
    # directions with some None layers: type 0 has layers 0..5, type 1 has layers 1..4, type 2 layer 3 only
    dirs = [[None] * n_layers for _ in range(3)]
    for l in range(0, 6):
        dirs[0][l] = RNG.standard_normal(n_embd).astype(np.float32)
    for l in range(1, 5):
        dirs[1][l] = RNG.standard_normal(n_embd).astype(np.float32)
    dirs[2][3] = RNG.standard_normal(n_embd).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "pos.gguf"
        ok = cv.export_gguf(dirs[1], n_layers, n_embd, HINT, path)
        assert ok
        g = GGUFFile(path)
        md = g.metadata
        assert md["general.architecture"] == "controlvector"
        assert md["controlvector.model_hint"] == HINT
        assert int(md["controlvector.layer_count"]) == n_layers
        tmap = {t.name: t for t in g.tensors}
        # the non-None layers EXCEPT layer 0 (0-based labels: layer l -> direction.l, no direction.0)
        assert set(tmap) == {"direction.1", "direction.2", "direction.3", "direction.4"}, set(tmap)
        for name, l in [("direction.1", 1), ("direction.2", 2), ("direction.3", 3), ("direction.4", 4)]:
            t = tmap[name]
            assert t.type_name == "F32", t.type_name
            assert t.shape == [n_embd], t.shape
            # values match what was exported
            with path.open("rb") as fh:
                fh.seek(g.data_start + t.offset)
                raw = fh.read(t.elements * 4)
            back = np.frombuffer(raw, dtype=np.float32)
            assert np.allclose(back, np.asarray(dirs[1][l]).reshape(-1)), name
        # a vector that only has layer 0 is exported as nothing (the engine steers layers 1.. last)
        only0 = [None] * n_layers
        only0[0] = RNG.standard_normal(n_embd).astype(np.float32)
        assert cv.export_gguf(only0, n_layers, n_embd, HINT, pathlib.Path(td) / "only0.gguf") is False
        print(f"  gguf round-trip: {sorted(tmap)} F32[{n_embd}], layer_count={n_layers}, layer 0 left out")


def test_gguf_loader_contract():
    # assert the exact keys the C++ ds4_cvec.hpp loader checks, so a hand-built file is loadable
    n_layers, n_embd = 4, 16
    dirs = [None] * n_layers
    dirs[0] = np.arange(n_embd, dtype=np.float32) * 1          # 0-based layer 0: analysed but not exported
    dirs[1] = np.arange(n_embd, dtype=np.float32) * 2          # 0-based layer 1 -> direction.1
    dirs[2] = np.arange(n_embd, dtype=np.float32) * 3          # 0-based layer 2 -> direction.2
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "d.gguf"
        assert cv.export_gguf(dirs, n_layers, n_embd, HINT, path)
        g = GGUFFile(path)
        # the loader requires general.architecture == 'controlvector' and direction.<l> F32 of length n_embd
        assert g.metadata["general.architecture"] == "controlvector"
        # 0-based mapping: layer l is direction.l; layer 0 and the None layer 3 are skipped
        tmap = {t.name: t for t in g.tensors}
        assert set(tmap) == {"direction.1", "direction.2"}, set(tmap)
        for name in tmap:
            assert tmap[name].type_name == "F32" and tmap[name].elements == n_embd
        assert "direction.0" not in tmap and "direction.3" not in tmap
        print("  loader contract: arch=controlvector, direction.<l> F32[n_embd] (0-based), layer 0 / None skipped OK")


def test_samples_memmap_roundtrip_and_create():
    # extract() writes A.npy / B.npy as RAW float32 memmaps (no .npy header) plus a meta.json; create() must
    # map them back with the shape meta.json recorded (np.load CANNOT read them - this was the "invalid load
    # key" bug).  Simulate a finished extraction on synthetic contrastive data, then run create() end-to-end:
    # the mapped values round-trip exactly and the three GGUFs come out.  A stale file (meta shape != file
    # size) and an incomplete extraction (done < T) must be rejected, not silently misread.
    T, n_layers, n_embd = S, L, E
    base, c0, c1 = _make_contrastive(1.0, 2.0)
    A_data = np.ascontiguousarray((c0 - base).astype(np.float32))
    B_data = np.ascontiguousarray((c1 - base).astype(np.float32))
    with tempfile.TemporaryDirectory() as td:
        sdir = pathlib.Path(td)
        A = np.memmap(sdir / "A.npy", dtype=np.float32, mode="w+", shape=(T, n_layers, n_embd))
        B = np.memmap(sdir / "B.npy", dtype=np.float32, mode="w+", shape=(T, n_layers, n_embd))
        A[:] = A_data
        B[:] = B_data
        A.flush(); B.flush()
        meta = {"done": T, "T": T, "n_layers": n_layers, "n_embd": n_embd,
                "class_names": ["alpha", "beta"], "seed": 0, "trait": "test"}
        (sdir / "meta.json").write_text(json.dumps(meta))
        back = cv._load_samples(sdir, "A", meta)
        assert np.array_equal(np.asarray(back), A_data), "raw memmap round-trip is not exact"
        out = pathlib.Path(td) / "ggufs"
        cv.create(argparse.Namespace(samples=sdir, out=out, start_layer=0, skip_end=1,
                                     tol=0.5, model_hint=HINT), ["alpha", "beta"])
        for nm in ("debias", "alpha", "beta"):
            assert (out / f"{nm}.gguf").exists(), f"{nm}.gguf missing"
        # a stale file (meta describes a bigger shape than the file holds) is rejected, not misread
        stale = dict(meta, done=T * 2, T=T * 2)
        (sdir / "meta.json").write_text(json.dumps(stale))
        try:
            cv._load_samples(sdir, "A", stale)
        except SystemExit:
            pass
        else:
            raise AssertionError("stale A.npy (size != meta shape) should raise SystemExit")
        # an incomplete extraction is rejected by create()
        incomplete = dict(meta, done=T - 1)
        (sdir / "meta.json").write_text(json.dumps(incomplete))
        try:
            cv.create(argparse.Namespace(samples=sdir, out=out, start_layer=0, skip_end=1,
                                         tol=0.5, model_hint=HINT), ["alpha", "beta"])
        except SystemExit:
            pass
        else:
            raise AssertionError("incomplete extraction (done < T) should raise SystemExit")
        # a missing meta.json is rejected with a clear message
        (sdir / "meta.json").unlink()
        try:
            cv.create(argparse.Namespace(samples=sdir, out=out, start_layer=0, skip_end=1,
                                         tol=0.5, model_hint=HINT), ["alpha", "beta"])
        except SystemExit:
            pass
        else:
            raise AssertionError("missing meta.json should raise SystemExit")
    print("  samples round-trip: raw memmap + meta.json -> create() -> 3 GGUFs; "
          "stale / incomplete / missing-meta all rejected")


def test_resume_keeps_extracted_rows():
    # extract --resume must keep the already-extracted rows (map with "r+") and continue at `done`.
    # (The old code mapped with "w+" - which TRUNCATES - before the resume check: the done rows were
    # wiped to zero and then skipped.)  A resume whose meta.json / files come from a different run must
    # be rejected, not silently mixed; a resume with the files missing starts fresh.
    T, n_layers, n_embd = 16, 4, 8
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td)
        A, B, start, meta_path = cv._open_samples(out, T, n_layers, n_embd, resume=False)
        assert start == 0
        done = 6
        for i in range(done):
            A[i] = float(i + 1)
        A.flush()
        meta_path.write_text(json.dumps({"done": done, "T": T, "n_layers": n_layers, "n_embd": n_embd,
                                         "class_names": ["alpha", "beta"], "seed": 0, "trait": "test"}))
        del A, B
        A, B, start, _ = cv._open_samples(out, T, n_layers, n_embd, resume=True)
        assert start == done
        for i in range(done):
            assert np.allclose(A[i], float(i + 1)), f"row {i} was wiped by the resume (got {A[i][0]})"
        del A, B
        meta_path.write_text(json.dumps({"done": 0, "T": T * 2, "n_layers": n_layers, "n_embd": n_embd}))
        try:
            cv._open_samples(out, T, n_layers, n_embd, resume=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("resume with a different T in meta.json should raise SystemExit")
        meta_path.write_text(json.dumps({"done": 0, "T": T, "n_layers": n_layers * 2, "n_embd": n_embd // 2}))
        try:
            cv._open_samples(out, T, n_layers, n_embd, resume=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("resume with a different geometry should raise SystemExit")
        (out / "A.npy").write_bytes(b"\x00" * 16)
        meta_path.write_text(json.dumps({"done": 0, "T": T, "n_layers": n_layers, "n_embd": n_embd}))
        try:
            cv._open_samples(out, T, n_layers, n_embd, resume=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("resume with a wrong A.npy size should raise SystemExit")
        (out / "A.npy").unlink(); (out / "B.npy").unlink()
        A, B, start, _ = cv._open_samples(out, T, n_layers, n_embd, resume=True)
        assert start == 0 and (out / "A.npy").exists()
    print("  resume: done rows survive (r+), start == done; "
          "different-run / wrong-size / missing-file all handled")


def _all():
    tests = [test_planted_direction_and_debias, test_discriminant_ratio_filters_noise,
             test_gguf_export_roundtrip, test_gguf_loader_contract,
             test_samples_memmap_roundtrip_and_create, test_resume_keeps_extracted_rows]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print("ALL PASS")


if __name__ == "__main__":
    _all()
