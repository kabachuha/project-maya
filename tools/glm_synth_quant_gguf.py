"""tools/glm_synth_quant_gguf.py - the synthetic glm5-next model as a QUANTIZED GGUF.

`tools/glm_synth_gguf.py`'s BIG fixture (the 45-layer real-cadence geometry) written with the
per-tensor types of a real mixed GGUF quant, so the pack (tools/iq_pack.py) and the engine can be
smoke-tested against the oracle without the 320 GB download.  The type map mirrors a
llama.cpp-quantized GLM-5.3-Flash release (file_type "FP8-IQ4_XS"):

  * routed experts   gate/up IQ3_S, down IQ4_XS (every MoE layer, one type per tensor);
  * attention, the dense FFNs, the shared experts, the head and the embedding  Q8_0;
  * the KDA gates and the absorbed MLA projections, the indexer's q_b/k  F16;
  * everything else (norms, mHC mixes, the router, the indexer's compressors)  F32.

The NextN/MTP block (blk.45, a DSA MoE layer's tensor set plus nextn.eh_proj / nextn.{e,h}norm /
nextn.shared_head_norm, as llama.cpp's conversion writes it) rides along, so the pack drops it
and the engine's load_mtp reads it from the GGUF like on a real file.

Quantization goes through ggml's own quantizer (ggml_quantize_chunk of libggml, via ctypes) - the
same code llama.cpp's quantize tool uses, so the blocks are canonical.  IQ3_S and IQ4_XS quantize
without an importance matrix here (a smoke test of the FORMAT, not of quantizer quality).

Usage:
    python tools/glm_synth_quant_gguf.py --out glm5-synth-quant.gguf --lib PATH/libggml-quant.so

`--lib`: a libggml with ggml_quantize_chunk (build third_party/llama.cpp/ggml, or
    tools/ds4/requant_gguf.py's --lib default: ~/llama.cpp/build/bin/libggml-base.so).
"""
from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_synth_gguf as S  # noqa: E402  (its kv encoders, geometry and tensor builder)

# ggml type ids this writer emits (ggml.h's enum order)
F32, F16, Q8_0, IQ3_S, IQ4_XS = 0, 1, 8, 21, 23

ALIGN = S.ALIGN


def _s(text: str) -> bytes:
    return S._s(text)


# ------------------------------------------------------------------ the type map
F16_OF = ("ssm_beta.weight", "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight",
          "attn_k_b.weight", "attn_v_b.weight", "indexer.attn_k.weight", "indexer.attn_q_b.weight",
          "attn_kv_a_mqa.weight")   # the released quant keeps the absorbed KV projection unquantized
F32_SUFFIX = ("attn_norm.weight", "ffn_norm.weight", "output_norm.weight", "hc_attn_fn.weight",
              "hc_attn_base.weight", "hc_attn_scale.weight", "hc_ffn_fn.weight", "hc_ffn_base.weight",
              "hc_ffn_scale.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_conv1d_q.weight",
              "ssm_conv1d_k.weight", "ssm_conv1d_v.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
              "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight",
              "indexer_compressor_gate.weight", "indexer_compressor_ape.weight", "ffn_gate_inp.weight",
              "exp_probs_b.bias", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight")
Q8_SUFFIX = ("token_embd.weight", "output.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
             "attn_q_a.weight", "attn_q_b.weight", "attn_output.weight",
             "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_gate_shexp.weight",
             "ffn_up_shexp.weight", "ffn_down_shexp.weight", "nextn.eh_proj.weight")


def type_of(name: str) -> int:
    """The target format's type of one tensor, by its name (the MoE experts aside)."""
    if name.endswith(F16_OF):
        return F16
    if name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
        return IQ3_S
    if name.endswith("ffn_down_exps.weight"):
        return IQ4_XS
    if name.endswith(F32_SUFFIX):
        return F32
    if name.endswith(Q8_SUFFIX):
        return Q8_0
    raise ValueError(f"no type in the map for {name}")


# ------------------------------------------------------------------ quantization
class Quantizer:
    """ggml_quantize_chunk of a libggml, via ctypes (tools/ds4/requant_gguf.py's pattern)."""

    def __init__(self, lib_path: str):
        lib = ctypes.CDLL(lib_path)
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib = lib
        self._init: dict[int, bool] = {}

    def quantize(self, tid: int, w: np.ndarray, n_per_row: int) -> bytes:
        if tid not in self._init:
            self.lib.ggml_quantize_init(tid)
            self._init[tid] = True
        flat = np.ascontiguousarray(w, dtype="<f4").ravel()
        if flat.size % n_per_row:
            raise ValueError(f"{flat.size} elements are not whole {n_per_row}-wide rows")
        out = np.empty(flat.size // n_per_row * _row_bytes(tid, n_per_row), dtype=np.uint8)
        got = self.lib.ggml_quantize_chunk(tid, flat.ctypes.data, out.ctypes.data, 0,
                                           flat.size // n_per_row, n_per_row, None)
        if got != out.size:
            raise ValueError(f"ggml_quantize_chunk wrote {got} bytes, expected {out.size}")
        return out.tobytes()


def _row_bytes(tid: int, n_per_row: int) -> int:
    if tid == Q8_0:
        el, by = 32, 34
    elif tid == IQ3_S:
        el, by = 256, 110
    elif tid == IQ4_XS:
        el, by = 256, 136
    else:
        raise ValueError(f"no block geometry for type {tid}")
    if n_per_row % el:
        raise ValueError(f"row width {n_per_row} is not whole {el}-element blocks")
    return n_per_row // el * by


def to_blob(tid: int, w: np.ndarray, n_per_row: int, q: Quantizer) -> bytes:
    if tid == F32:
        return np.ascontiguousarray(w, dtype="<f4").tobytes()
    if tid == F16:
        return np.ascontiguousarray(w, dtype="<f2").tobytes()
    return q.quantize(tid, w, n_per_row)


# ------------------------------------------------------------------ the writer
def write_gguf_typed(path: Path, metadata: list[tuple[str, bytes]],
                     tensors: list[tuple[str, list[int], int, bytes]]) -> None:
    """GGUF v3 with pre-built blobs: (name, ne (ggml order), type id, bytes) per tensor."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    kvs = b"".join(_s(k) + v for k, v in metadata)
    infos, offset = b"", 0
    body = []
    for name, ne, tid, blob in tensors:
        infos += _s(name) + struct.pack("<I", len(ne))
        infos += b"".join(struct.pack("<Q", d) for d in ne)
        infos += struct.pack("<I", tid) + struct.pack("<Q", offset)
        body.append(blob)
        offset += len(blob)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
    data_pad = (len(header) + len(kvs) + len(infos) + ALIGN - 1) // ALIGN * ALIGN
    pad = b"\0" * (data_pad - len(header) - len(kvs) - len(infos))
    body = b"".join(b.ljust(len(b) + (-len(b) % ALIGN), b"\0") for b in body)
    path.write_bytes(header + kvs + infos + pad + body)


# ------------------------------------------------------------------ the model
def build(lib_path: str, seed: int = 777) -> tuple[list[tuple[str, bytes]], list[tuple[str, list[int], int, bytes]]]:
    """The BIG fixture's metadata (the NextN block counted in) and its tensors, quantized."""
    md, ts = S.build_big(seed=seed, vocab=True)

    def md_i(name: str) -> int:
        return next(i for i, (k, _) in enumerate(md) if k == name)

    # the NextN block rides inside block_count, like llama.cpp's conversion (46 with nextn 1)
    md[md_i("glm5-next.block_count")] = ("glm5-next.block_count", S.kv_u32(S.BIG["n_layers"] + 1))
    md.append(("glm5-next.nextn_predict_layers", S.kv_u32(1)))
    md[md_i("glm5-next.attention.head_count_kv")] = (
        "glm5-next.attention.head_count_kv", S.arr_i32([*S.BIG["n_head_kv"], 1]))
    md[md_i("glm5-next.attention.indexer.types")] = (
        "glm5-next.attention.indexer.types", S.arr_i32([1] * (S.BIG["n_layers"] + 1)))
    md[md_i("glm5-next.swiglu_clamp_exp")] = (
        "glm5-next.swiglu_clamp_exp", S.arr_f32([10.0] * (S.BIG["n_layers"] + 1)))
    md[md_i("glm5-next.swiglu_clamp_shexp")] = (
        "glm5-next.swiglu_clamp_shexp", S.arr_f32([10.0] * (S.BIG["n_layers"] + 1)))

    q = Quantizer(lib_path)
    out: list[tuple[str, list[int], int, bytes]] = []
    for name, arr in ts:
        tid = type_of(name)
        ne = [int(d) for d in reversed(arr.shape)]        # ne0 varies fastest
        n_per_row = ne[0]
        out.append((name, ne, tid, to_blob(tid, arr, n_per_row, q)))

    # blk.<n_layers>: the draft block, a DSA MoE layer's tensor set plus the nextn tensors -
    # layer 43 is one (43 % 4 == 3); its tensors renamed, the MoE shapes re-rolled to the block
    draft: list[tuple[str, np.ndarray]] = []
    for name, arr in ts:
        if name.startswith("blk.43."):
            draft.append(("blk.%d.%s" % (S.BIG["n_layers"], name.split(".", 2)[2]), arr))
    rng = np.random.default_rng(seed + 1)
    n_embd = S.BIG["n_embd"]
    draft += [
        ("blk.%d.nextn.eh_proj.weight" % S.BIG["n_layers"],
         (rng.standard_normal((2 * n_embd, n_embd)) * 0.05).astype(np.float32)),
        ("blk.%d.nextn.enorm.weight" % S.BIG["n_layers"], (1.0 + rng.standard_normal(n_embd) * 0.02).astype(np.float32)),
        ("blk.%d.nextn.hnorm.weight" % S.BIG["n_layers"], (1.0 + rng.standard_normal(n_embd) * 0.02).astype(np.float32)),
        ("blk.%d.nextn.shared_head_norm.weight" % S.BIG["n_layers"],
         (1.0 + rng.standard_normal(n_embd) * 0.02).astype(np.float32)),
    ]
    for name, arr in draft:
        tid = type_of(name)
        ne = [int(d) for d in reversed(arr.shape)]
        out.append((name, ne, tid, to_blob(tid, arr, ne[0], q)))
    return md, out


def selftest(path: Path) -> None:
    sys.path.insert(0, str(HERE))
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    by = {t.name: t for t in gf.tensors}
    md = gf.metadata
    assert md["general.architecture"] == "glm5-next", "architecture key lost"
    assert md["glm5-next.block_count"] == S.BIG["n_layers"] + 1 and md["glm5-next.nextn_predict_layers"] == 1, "nextn"
    kv = md["glm5-next.attention.head_count_kv"]
    assert len(kv) == S.BIG["n_layers"] + 1 and kv[-1] == 1, "the draft block's KV entry"
    assert by["blk.3.ffn_gate_exps.weight"].type_name == "IQ3_S", "gate exps type"
    assert by["blk.3.ffn_up_exps.weight"].type_name == "IQ3_S", "up exps type"
    assert by["blk.3.ffn_down_exps.weight"].type_name == "IQ4_XS", "down exps type"
    assert by["blk.3.attn_k_b.weight"].type_name == "F16", "absorbed MLA type"
    assert by["blk.3.attn_kv_a_mqa.weight"].type_name == "F16", "the KV projection's type"
    assert by["blk.3.attn_q_a.weight"].type_name == "Q8_0", "attention projection type"
    assert by["token_embd.weight"].type_name == "Q8_0" and by["output.weight"].type_name == "Q8_0", "head/embed type"
    assert "blk.%d.nextn.eh_proj.weight" % S.BIG["n_layers"] in by, "the draft block present"
    assert by["blk.0.attn_q.weight"].type_name == "Q8_0", "KDA projection type"
    assert by["blk.0.ssm_beta.weight"].type_name == "F16", "KDA gate type"
    print(f"selftest ok: {len(gf.tensors)} tensors, {path.stat().st_size / 1e6:.2f} MB (quant)")


def main() -> int:
    ap = argparse.ArgumentParser(description="write the synthetic glm5-next GGUF as a quantized mixed file")
    ap.add_argument("--out", default="glm5-synth-quant.gguf")
    ap.add_argument("--lib", required=True, help="a libggml with ggml_quantize_chunk (see the module docstring)")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    md, tensors = build(args.lib, args.seed)
    write_gguf_typed(Path(args.out), md, tensors)
    selftest(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
