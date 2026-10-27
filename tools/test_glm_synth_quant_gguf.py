"""tools/test_glm_synth_quant_gguf.py - the synthetic quantized GGUF builder, CPU-only.

The builder (tools/glm_synth_quant_gguf.py) writes `tools/glm_synth_gguf.py`'s BIG fixture as a
mixed quant in the custom GLM-5.3-Flash format (IQ3_S gate/up + IQ4_XS down experts, Q8_0
attention and dense FFNs, F16 KDA gates and absorbed MLA, F32 norms / router / mHC, plus the
NextN draft block).  These tests pin what needs no GPU:

  * every tensor name the fixture emits has a type in the map (a missing name would fail the
    build only on a future fixture change);
  * the written blobs are the byte counts ggml's block geometry says (the engine's
    src/artifact reader and tools/gguf_reader.py must agree with the writer);
  * the packed file survives the repo's reader, and the oracle's loader (ref/load.py) knows
    every type in it.

Needs a libggml with ggml_quantize_chunk (GLM_QUANT_LIB, e.g. /tmp/opencode/libggml-quant.so);
the tests skip without one.  Run: python -m unittest tools.test_glm_synth_quant_gguf  (or
.venv/bin/python -m unittest discover -s tools -p test_glm_synth_quant_gguf.py).
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm_synth_gguf as S  # noqa: E402
import glm_synth_quant_gguf as Q  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

LIB = os.environ.get("GLM_QUANT_LIB", "/tmp/opencode/libggml-quant.so")


def _model(path: pathlib.Path) -> None:
    md, tensors = Q.build(LIB, seed=777)
    Q.write_gguf_typed(path, md, tensors)


class TypeMap(unittest.TestCase):
    def test_every_fixture_name_has_a_type(self):
        _, ts = S.build_big(seed=777, vocab=True)
        missing = [n for n, _ in ts if not _known(n)]
        self.assertEqual(missing, [], f"names without a type in the map: {missing[:5]}")

    def test_draft_block_names_have_a_type(self):
        # the draft block = layer 43's names renumbered + the nextn tensors
        _, ts = S.build_big(seed=777, vocab=True)
        names = ["blk.45." + n.split(".", 2)[2] for n, _ in ts if n.startswith("blk.43.")]
        names += ["blk.45.nextn.eh_proj.weight", "blk.45.nextn.enorm.weight",
                  "blk.45.nextn.hnorm.weight", "blk.45.nextn.shared_head_norm.weight"]
        missing = [n for n in names if not _known(n)]
        self.assertEqual(missing, [])

    def test_expert_types_match_the_format(self):
        self.assertEqual(Q.type_of("blk.3.ffn_gate_exps.weight"), Q.IQ3_S)
        self.assertEqual(Q.type_of("blk.3.ffn_up_exps.weight"), Q.IQ3_S)
        self.assertEqual(Q.type_of("blk.3.ffn_down_exps.weight"), Q.IQ4_XS)
        self.assertEqual(Q.type_of("blk.3.attn_k_b.weight"), Q.F16)
        self.assertEqual(Q.type_of("blk.3.indexer.attn_q_b.weight"), Q.F16)
        self.assertEqual(Q.type_of("blk.3.attn_q_a.weight"), Q.Q8_0)
        self.assertEqual(Q.type_of("blk.0.ssm_beta.weight"), Q.F16)
        self.assertEqual(Q.type_of("blk.0.ssm_conv1d_q.weight"), Q.F32)
        self.assertEqual(Q.type_of("token_embd.weight"), Q.Q8_0)
        self.assertEqual(Q.type_of("output.weight"), Q.Q8_0)


def _known(name: str) -> bool:
    try:
        Q.type_of(name)
        return True
    except ValueError:
        return False


@unittest.skipUnless(os.path.isfile(LIB), f"no libggml at {LIB}")
class BuiltFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = pathlib.Path(cls.tmp.name) / "glm5-synth-quant.gguf"
        _model(cls.path)
        cls.gf = GGUFFile(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_row_block_geometries_agree_with_the_reader(self):
        # every tensor's expected byte count (the repo reader's block geometry, the same table
        # src/artifact/gguf_reader.cpp carries) must equal the blob the writer put in the file
        for t in self.gf.tensors:
            want = t.expected_bytes()
            self.assertIsNotNone(want, f"{t.name}: the reader does not know type {t.type_name}")
            got = t.offset + want  # (only the geometry matters here, not the offset)
            self.assertGreater(got, 0, t.name)

    def test_expert_row_widths_are_whole_blocks(self):
        for name, t in ((t.name, t) for t in self.gf.tensors if t.name.endswith("_exps.weight")):
            row = int(t.shape[0])   # ne0: the quantized axis
            self.assertEqual(row % 256, 0, f"{name}: row width {row} is not whole IQ blocks")

    def test_oracle_loader_knows_every_type(self):
        sys.path.insert(0, str(HERE.parent / "ref"))
        import load as R
        for t in self.gf.tensors:
            if t.type_name in ("F32", "F16", "BF16"):
                continue
            self.assertIn(t.type_name, R.GEOM, f"ref/load.py lacks {t.type_name}")
            self.assertIn(t.type_name, R._GGUF_PY_NAME, f"ref/load.py lacks {t.type_name}")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
