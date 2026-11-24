"""tools/glm46/glm46_synth_gguf.py - a tiny, structurally complete synthetic glm4moe (GLM-4.5+/GLM-4.6) GGUF.

The smoke-test model: the tensor set and the per-tensor types of the shipped mixed quant
(Q8_0 attention / dense / shared experts / head, IQ3_S gate+up experts, IQ4_XS down experts, F32
norms, router and biases, exactly ReadyArt/GLM-4.6-Derestricted-Q8_0-IQ3_S-IQ3_S-IQ4_XS's map),
with the dimensions scaled down and values from a seeded RNG.

Geometry (small enough for one 3090; the kernels' hard bounds respected):
    dim 512, 12 query heads / 1 KV head (kG = 12, the GLM-4.5/4.6 ratio), head 128, rope 64,
    9 trunk layers (3 dense + 6 MoE), 8 experts a token's 2, expert FF 256, dense/shared FF 256,
    vocab 512, theta 1e4, rms eps 1e-5, routed scale 2.5

Quantization goes through ggml's own quantizer (ggml_quantize_chunk of libggml, via ctypes) - the
same code llama.cpp's quantize tool uses, so the blocks are canonical.

Usage:
    python tools/glm46/glm46_synth_gguf.py --out glm46-synth.gguf \
        [--lib ~/llama.cpp/build/bin/libggml-base.so] [--seed 777]
    python tools/glm46/pack_gguf.py --model <dir with the gguf>     (writes maya-glm46/)

The tokenizer is absent on purpose: the engine's one-shot path runs token ids, and the real
tokenizer is a conversion concern (tools/glm46/pack_gguf.py exports it from a real GGUF).
"""
from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

import numpy as np

# GGUF metadata value types (ggml spec order)
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
# ggml tensor type ids this writer emits
GGML_F32, GGML_F16, GGML_Q8_0, GGML_IQ3_S, GGML_IQ4_XS = 0, 1, 8, 21, 23

ALIGN = 32

# the synthetic model
G = dict(
    n_embd=512, n_layers=9, n_head=12, n_kv=1, hd=128, n_rot=64, vocab=512,
    n_exp=8, k_exp=2, ff=256, ff_dense=256, ff_shared=256, dense_lead=3,
    theta=1e4, eps=1e-5, scale=2.5, bos=0, eos=1,
)


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_u32(v): return struct.pack("<I", U32) + struct.pack("<I", v)
def kv_i32(v): return struct.pack("<I", I32) + struct.pack("<i", v)
def kv_f32(v): return struct.pack("<I", F32) + struct.pack("<f", v)
def kv_bool(v): return struct.pack("<I", BOOL) + struct.pack("<B", 1 if v else 0)
def kv_str(v): return struct.pack("<I", STRING) + _s(v)


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


# ------------------------------------------------------------------ quantization (ggml's own)
class Quantizer:
    """ggml_quantize_chunk of a libggml, via ctypes."""

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
    el, by = {GGML_Q8_0: (32, 34), GGML_IQ3_S: (256, 110), GGML_IQ4_XS: (256, 136)}[tid]
    if n_per_row % el:
        raise ValueError(f"row width {n_per_row} is not whole {el}-element blocks")
    return n_per_row // el * by


def to_blob(tid: int, w: np.ndarray, n_per_row: int, q: Quantizer) -> bytes:
    if tid == GGML_F32:
        return np.ascontiguousarray(w, dtype="<f4").tobytes()
    if tid == GGML_F16:
        return np.ascontiguousarray(w, dtype="<f2").tobytes()
    return q.quantize(tid, w, n_per_row)


# ------------------------------------------------------------------ the model
def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2's bytes_to_unicode (the vocabulary's first 256 symbols a real gpt2/glm4 vocab starts with)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + \
         list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def vocab_tokens() -> list[str]:
    b2u = _gpt2_bytes_to_unicode()
    toks = [b2u[b] for b in range(256)]
    toks += [f"t{i}" for i in range(256, G["vocab"])]
    return toks


def metadata() -> list[tuple[str, bytes]]:
    md = [
        ("general.architecture", kv_str("glm4moe")),
        ("general.name", kv_str("GLM-4.6 synthetic")),
        ("general.size_label", kv_str("synth")),
        ("glm4moe.block_count", kv_u32(G["n_layers"])),
        ("glm4moe.context_length", kv_u32(4096)),
        ("glm4moe.embedding_length", kv_u32(G["n_embd"])),
        ("glm4moe.attention.head_count", kv_u32(G["n_head"])),
        ("glm4moe.attention.head_count_kv", kv_u32(G["n_kv"])),
        ("glm4moe.attention.key_length", kv_u32(G["hd"])),
        ("glm4moe.attention.value_length", kv_u32(G["hd"])),
        ("glm4moe.attention.layer_norm_rms_epsilon", kv_f32(G["eps"])),
        ("glm4moe.rope.dimension_count", kv_u32(G["n_rot"])),
        ("glm4moe.rope.freq_base", kv_f32(G["theta"])),
        ("glm4moe.feed_forward_length", kv_u32(G["ff_dense"])),
        ("glm4moe.expert_feed_forward_length", kv_u32(G["ff"])),
        ("glm4moe.expert_count", kv_u32(G["n_exp"])),
        ("glm4moe.expert_used_count", kv_u32(G["k_exp"])),
        ("glm4moe.expert_shared_count", kv_u32(1)),
        ("glm4moe.expert_gating_func", kv_u32(2)),
        ("glm4moe.expert_weights_scale", kv_f32(G["scale"])),
        ("glm4moe.expert_weights_norm", kv_bool(True)),
        ("glm4moe.leading_dense_block_count", kv_u32(G["dense_lead"])),
        ("glm4moe.nextn_predict_layers", kv_u32(0)),
        ("tokenizer.ggml.model", kv_str("gpt2")),
        ("tokenizer.ggml.pre", kv_str("glm4")),
        ("tokenizer.ggml.tokens", struct.pack("<I", ARRAY) + struct.pack("<I", STRING) + struct.pack("<Q", G["vocab"]) +
         b"".join(_s(t) for t in vocab_tokens())),
        ("tokenizer.ggml.merges", struct.pack("<I", ARRAY) + struct.pack("<I", STRING) + struct.pack("<Q", 1) +
         _s("\u0120 \u0120")),
        ("tokenizer.ggml.token_type", struct.pack("<I", ARRAY) + struct.pack("<I", I32) + struct.pack("<Q", G["vocab"]) +
         b"".join(struct.pack("<i", 1) for _ in range(G["vocab"]))),
        ("tokenizer.ggml.bos_token_id", kv_u32(G["bos"])),
        ("tokenizer.ggml.eos_token_id", kv_u32(G["eos"])),
    ]
    return md


def build(lib: str, seed: int) -> tuple[list[tuple[str, bytes]], list[tuple[str, list[int], int, bytes]]]:
    rng = np.random.default_rng(seed)
    q = Quantizer(lib)
    ts: list[tuple[str, list[int], int, bytes]] = []

    def add(name: str, arr: np.ndarray, tid: int) -> None:
        ne = [int(d) for d in reversed(arr.shape)]
        ts.append((name, ne, tid, to_blob(tid, arr, ne[0], q)))

    def w(*shape, s=0.05):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def norm(n):
        return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)

    add("token_embd.weight", w(G["vocab"], G["n_embd"], s=0.02), GGML_Q8_0)
    add("output_norm.weight", norm(G["n_embd"]), GGML_F32)
    add("output.weight", w(G["vocab"], G["n_embd"], s=0.02), GGML_Q8_0)
    for i in range(G["n_layers"]):
        P = f"blk.{i}."
        add(P + "attn_norm.weight", norm(G["n_embd"]), GGML_F32)
        add(P + "post_attention_norm.weight", norm(G["n_embd"]), GGML_F32)
        add(P + "attn_q_norm.weight", norm(G["hd"]), GGML_F32)
        add(P + "attn_k_norm.weight", norm(G["hd"]), GGML_F32)
        add(P + "attn_q.weight", w(G["n_head"] * G["hd"], G["n_embd"]), GGML_Q8_0)
        add(P + "attn_q.bias", w(G["n_head"] * G["hd"], s=0.02), GGML_F32)
        add(P + "attn_k.weight", w(G["n_kv"] * G["hd"], G["n_embd"]), GGML_Q8_0)
        add(P + "attn_k.bias", w(G["n_kv"] * G["hd"], s=0.02), GGML_F32)
        add(P + "attn_v.weight", w(G["n_kv"] * G["hd"], G["n_embd"]), GGML_Q8_0)
        add(P + "attn_v.bias", w(G["n_kv"] * G["hd"], s=0.02), GGML_F32)
        add(P + "attn_output.weight", w(G["n_embd"], G["n_head"] * G["hd"]), GGML_Q8_0)
        if i < G["dense_lead"]:
            add(P + "ffn_gate.weight", w(G["ff_dense"], G["n_embd"]), GGML_Q8_0)
            add(P + "ffn_up.weight", w(G["ff_dense"], G["n_embd"]), GGML_Q8_0)
            add(P + "ffn_down.weight", w(G["n_embd"], G["ff_dense"]), GGML_Q8_0)
        else:
            add(P + "ffn_gate_inp.weight", w(G["n_exp"], G["n_embd"], s=0.1), GGML_F32)
            add(P + "exp_probs_b.bias", (2.0 - 4.0 * np.arange(G["n_exp"]) / max(1, G["n_exp"] - 1)).astype(np.float32), GGML_F32)
            add(P + "ffn_gate_exps.weight", w(G["n_exp"], G["ff"], G["n_embd"]), GGML_IQ3_S)
            add(P + "ffn_up_exps.weight", w(G["n_exp"], G["ff"], G["n_embd"]), GGML_IQ3_S)
            add(P + "ffn_down_exps.weight", w(G["n_exp"], G["n_embd"], G["ff"]), GGML_IQ4_XS)
            add(P + "ffn_gate_shexp.weight", w(G["ff_shared"], G["n_embd"]), GGML_Q8_0)
            add(P + "ffn_up_shexp.weight", w(G["ff_shared"], G["n_embd"]), GGML_Q8_0)
            add(P + "ffn_down_shexp.weight", w(G["n_embd"], G["ff_shared"]), GGML_Q8_0)
    return metadata(), ts


def selftest(path: Path) -> None:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from gguf_reader import GGUFFile  # noqa: E402
    gf = GGUFFile(path)
    by = {t.name: t for t in gf.tensors}
    md = gf.metadata
    assert md["general.architecture"] == "glm4moe", "architecture key"
    assert md["glm4moe.block_count"] == G["n_layers"], "block count"
    assert md["glm4moe.expert_count"] == G["n_exp"] and md["glm4moe.expert_used_count"] == G["k_exp"], "experts"
    assert by["blk.0.attn_q.weight"].type_name == "Q8_0", "attention projection type"
    assert by["blk.0.ffn_gate.weight"].type_name == "Q8_0", "dense FFN type"
    assert by[f"blk.{G['dense_lead']}.ffn_gate_exps.weight"].type_name == "IQ3_S", "gate experts type"
    assert by[f"blk.{G['dense_lead']}.ffn_up_exps.weight"].type_name == "IQ3_S", "up experts type"
    assert by[f"blk.{G['dense_lead']}.ffn_down_exps.weight"].type_name == "IQ4_XS", "down experts type"
    assert by[f"blk.{G['dense_lead']}.ffn_gate_shexp.weight"].type_name == "Q8_0", "shared expert type"
    assert by["token_embd.weight"].type_name == "Q8_0" and by["output.weight"].type_name == "Q8_0", "head/embed type"
    assert by["blk.0.attn_q.bias"].type_name == "F32", "the q bias type"
    print(f"selftest ok: {len(gf.tensors)} tensors, {path.stat().st_size / 1e6:.2f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="write a synthetic glm4moe GGUF (the smoke-test model)")
    ap.add_argument("--out", default="glm46-synth.gguf")
    ap.add_argument("--lib", default="/home/khrapov/AI4/llama.cpp/build/bin/libggml-base.so",
                    help="a libggml with ggml_quantize_chunk")
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    md, tensors = build(a.lib, a.seed)
    write_gguf_typed(Path(a.out), md, tensors)
    selftest(Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
