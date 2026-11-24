"""tools/glm46/pack_gguf.py - maya-glm46's pack for a GLM-4.5+/GLM-4.6 GGUF set (llama.cpp's `glm4moe`
architecture: ReadyArt/GLM-4.6-Derestricted-GGUF and the like), beside the model's files (no weights copied).

    python tools/glm46/pack_gguf.py --model ~/models/GLM-4.6-Q8_0-IQ3_S-IQ3_S-IQ4_XS

writes <model>/maya-glm46/:
  glm46.txt      the geometry under config.json's names (from the GGUF's glm4moe.* keys; the trunk's layer count is
                 block_count less nextn_predict_layers - the NextN block is not served yet), format gguf, quant_name,
                 model_dir, qkv_bias / norms_plus_one
  tensors.txt    every trunk tensor as the GGUF stores it, `file <i> <shard>` lines then
                 `t <name> <dtype> <file> <offset> <bytes> <dims...>` - dims outermost first (a matrix [out][in]),
                 dtype ggml's type name (Q8_0, IQ3_S, IQ4_XS ...), offset absolute in the shard
  tokenizer/     exported from the shard that holds the vocabulary, with the GGUF's chat template
                 (tools/strata_tokenizer.py)

The quant format this serves is the shipped mixed one - Q8_0 attention / dense FFN / shared experts / head,
IQ3_S gate+up experts, IQ4_XS down experts, F32 norms, router and biases - but the pack is type-agnostic: whatever
types the GGUF carries are handed to the engine, which refuses what its kernels do not cover.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
import gguf_reader as G  # noqa: E402


def shard_names(first: pathlib.Path) -> list[pathlib.Path]:
    """-00001-of-0000N beside the first: every shard of the set, in order (a lone file: itself)."""
    name = first.name
    marker = "-00001-of-"
    i = name.find(marker)
    if i < 0:
        return [first]
    total = int(name[i + len(marker):i + len(marker) + 5])
    stem = name[:i]
    out = []
    for k in range(1, total + 1):
        out.append(first.parent / f"{stem}-{k:05d}-of-{total:05d}.gguf")
    missing = [p for p in out if not p.is_file()]
    if missing:
        raise SystemExit("the shard set is incomplete, missing: " + ", ".join(p.name for p in missing))
    return out


def geometry_of(shards: list[pathlib.Path]) -> tuple[dict, dict]:
    """The glm4moe.* numbers and the tokenizer summary, from the first shard's metadata."""
    g = G.GGUFFile(shards[0])
    md = g.metadata
    if md.get("general.architecture") != "glm4moe":
        raise SystemExit(f"{shards[0].name}: general.architecture is {md.get('general.architecture')!r}, not 'glm4moe'")
    p = "glm4moe."

    def need(k):
        if p + k not in md:
            raise SystemExit(f"{shards[0].name}: {p}{k} is missing")
        return md[p + k]

    block = int(need("block_count"))
    nextn = int(md.get(p + "nextn_predict_layers", 0))
    trunk = block - nextn
    if trunk < 1:
        raise SystemExit(f"block_count {block} with nextn {nextn} leaves no trunk")
    dense = int(md.get(p + "leading_dense_block_count", 0))
    n_head = int(need("attention.head_count"))
    n_kv = int(need("attention.head_count_kv"))
    head_dim = int(md.get(p + "attention.key_length", int(need("embedding_length")) // n_head))
    rot = int(md.get(p + "rope.dimension_count", head_dim // 2))
    n_exp = int(need("expert_count"))
    k_exp = int(need("expert_used_count"))
    ff = int(need("expert_feed_forward_length"))
    ff_dense = int(need("feed_forward_length"))
    ff_shared = int(md.get(p + "expert_shared_count", 1)) * ff   # (the shared expert's width = one expert's)
    geom = {
        "arch": "glm4moe",
        "format": "gguf",
        "quant_name": str(md.get("general.size_label", md.get("general.name", ""))) or "gguf",
        "hidden_size": int(need("embedding_length")),
        "num_hidden_layers": trunk,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "rotary_dim": rot,
        "vocab_size": int(md.get(p + "vocab_size", md.get("tokenizer.ggml.tokens", [0]) and
                                 len(md.get("tokenizer.ggml.tokens", []))) or 0),
        "intermediate_size": ff,
        "dense_intermediate_size": ff_dense,
        "shared_intermediate_size": ff_shared,
        "num_local_experts": n_exp,
        "num_experts_per_tok": k_exp,
        "rope_theta": float(need("rope.freq_base")),
        "rms_norm_eps": float(need("attention.layer_norm_rms_epsilon")),
        "routed_scaling_factor": float(md.get(p + "expert_weights_scale", 1.0)),
        "moe_layer_freq": ",".join("0" if i < dense else "1" for i in range(trunk)),
        "bos_token_id": int(md.get("tokenizer.ggml.bos_token_id", 1)),
        "eos_token_id": int(md.get("tokenizer.ggml.eos_token_id", 2)),
        "eot_token_id": int(md.get("tokenizer.ggml.eot_token_id", md.get("tokenizer.ggml.eos_token_id", 2))),
        "qkv_bias": 1 if md.get("glm4moe.attention.qkv_bias", True) is not False else 0,
        "norms_plus_one": 0,
        "model_dir": str(shards[0].parent),
    }
    if geom["vocab_size"] <= 0:
        raise SystemExit("the GGUF has no tokenizer.ggml.tokens to size the vocabulary from")
    return geom, md


def main() -> int:
    ap = argparse.ArgumentParser(description="write maya-glm46's pack beside a GLM-4.5+/GLM-4.6 GGUF set")
    ap.add_argument("--model", required=True, help="the model directory (or any shard of the set)")
    ap.add_argument("--out", default="", help="the pack directory (default: <model dir>/maya-glm46)")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the tokenizer export")
    a = ap.parse_args()
    m = pathlib.Path(a.model)
    first = m if m.suffix == ".gguf" else next(iter(sorted(m.glob("*.gguf"))), None)
    if first is None:
        raise SystemExit(f"{m}: no .gguf file")
    shards = shard_names(first)
    geom, md = geometry_of(shards)
    out = pathlib.Path(a.out) if a.out else first.parent / "maya-glm46"
    out.mkdir(parents=True, exist_ok=True)

    # the trunk tensors of every shard: the names, types, absolute offsets and shapes the engine reads
    lines = [f"file {i} {p.name}" for i, p in enumerate(shards)]
    trunk = geom["num_hidden_layers"]
    kept = 0
    dense_types = set()
    expert_types = set()
    for fi, p in enumerate(shards):
        g = G.GGUFFile(p)
        for t in g.tensors:
            if t.name.startswith("blk."):
                if int(t.name.split(".", 2)[1]) >= trunk:
                    continue   # (the NextN block: not served; its tensors stay in the GGUF)
            nbytes = t.expected_bytes()
            if nbytes is None:
                raise SystemExit(f"{p.name}: {t.name} is {t.type_name}, no block geometry")
            off = g.data_start + t.offset
            dims = [int(d) for d in reversed(t.shape)]   # outermost first, as the engine's pack reads them
            lines.append(f"t {t.name} {t.type_name} {fi} {off} {nbytes} " + " ".join(str(d) for d in dims))
            kept += 1
            if ".ffn_" in t.name and "_exps." in t.name:
                expert_types.add(t.type_name)
            elif not t.name.startswith("token_embd.") and t.name != "output.weight":
                dense_types.add(t.type_name)
    (out / "tensors.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    geom["gguf.dense"] = ",".join(sorted(dense_types))
    geom["gguf.experts"] = ",".join(sorted(expert_types))
    (out / "glm46.txt").write_text("".join(f"{k} {v}\n" for k, v in geom.items()), encoding="utf-8")

    if not a.no_tokenizer:
        tok = out / "tokenizer"
        tok.mkdir(exist_ok=True)
        rc = subprocess.run([sys.executable, str(REPO / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                             "--out", str(tok)], check=False).returncode
        if rc != 0:
            print("NOTE: the tokenizer export failed (one-shot --ids runs still work)", file=sys.stderr)
    print(f"pack: {out} - {kept} tensors from {len(shards)} shard(s); trunk {trunk} layers "
          f"({geom['num_hidden_layers'] - len([1 for x in geom['moe_layer_freq'].split(',') if x == '0'])} MoE), "
          f"dim {geom['hidden_size']}, {geom['num_attention_heads']}/{geom['num_key_value_heads']} heads, "
          f"{geom['num_local_experts']} experts ({geom['num_experts_per_tok']} a token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
