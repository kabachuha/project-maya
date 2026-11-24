"""ref/glm46.py - the glm4moe (GLM-4.5+/GLM-4.6) reference forward pass, one op at a time.

The specification maya-glm46's kernels are parity-tested against, written from llama.cpp's
src/models/glm4-moe.cpp (the implementation this architecture's GGUF conversion follows) and the
transformers reference (modeling_glm4_moe.py).  Conventions as ref/glm.py keeps them:

  * activations are (features, tokens), features varying fastest, like a ggml tensor's ne[0];
  * weights come from ref/load.py's `tensor()` as (out, in), so `y = W @ x` IS `ggml_mul_mat`;
  * splitting a (dim*n_head, T) projection follows ggml's reshape order (see split_heads);
  * the one pass is teacher-forced over the whole sequence; every query attends the positions
    0 .. its own (a plain causal mask), the caches built in place.

The rope is llama.cpp's NEOX pairing - channel i with i + n_rot/2, frequency i - which the glm4moe
conversion keeps (llama.cpp's rope_neox is the half-split form despite the name).

Usage:
    python ref/glm46.py MODEL.gguf --tokens 1,2,3,4 [--dump FILE] [--greedy N]
    (--dump writes "MM3L" + int32[3] + ids + pos + float logits, the engine's --dump-logits layout)
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from load import Gguf  # noqa: E402


class Glm46Config:
    def __init__(self, g: Gguf):
        md = g.g.metadata
        p = "glm4moe."

        def num(k, d=None):
            if p + k in md:
                return md[p + k]
            if d is None:
                raise SystemExit(f"{p}{k} is missing")
            return d

        self.n_embd = int(num("embedding_length"))
        self.n_layer = int(num("block_count"))
        self.nextn = int(num("nextn_predict_layers", 0))
        self.n_head = int(num("attention.head_count"))
        self.n_kv = int(num("attention.head_count_kv"))
        self.hd = int(num("attention.key_length", self.n_embd // self.n_head))
        self.n_rot = int(num("rope.dimension_count", self.hd // 2))
        self.n_vocab = len(md["tokenizer.ggml.tokens"])
        self.n_exp = int(num("expert_count"))
        self.k_exp = int(num("expert_used_count"))
        self.ff = int(num("expert_feed_forward_length"))
        self.ff_dense = int(num("feed_forward_length"))
        self.ff_shared = self.ff * int(num("expert_shared_count", 1))
        self.dense_lead = int(num("leading_dense_block_count", 0))
        self.theta = float(num("rope.freq_base"))
        self.eps = float(num("attention.layer_norm_rms_epsilon"))
        self.w_scale = float(num("expert_weights_scale", 1.0))
        self.norm_w = int(num("expert_weights_norm", True))
        self.gating_sigmoid = int(num("expert_gating_func", 2)) == 2
        self.trunk = self.n_layer - self.nextn


# ------------------------------------------------------------------ the ops (features, tokens)
def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)) * w[:, None]


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return silu(gate) * up


def split_heads(x: np.ndarray, dim: int, n_head: int) -> np.ndarray:
    """(dim*n_head, T) -> (head, dim, T), ggml's reshape order: element (e, h) at row e + dim*h."""
    return x.reshape(n_head, dim, x.shape[1])


def per_head_rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps)) * w[None, :, None]


def rope(x: np.ndarray, pos: np.ndarray, n_rot: int, inv: np.ndarray) -> np.ndarray:
    """llama.cpp's rope_neox on (head, dim, T): channel i with i + n_rot/2, frequency i."""
    out = x.copy()
    half = n_rot // 2
    for i in range(half):
        a = pos[None, :] * inv[i]                       # (T,)
        c, s = np.cos(a), np.sin(a)
        x0 = x[:, i, :]          # (head, T)
        x1 = x[:, i + half, :]
        out[:, i, :] = x0 * c - x1 * s
        out[:, i + half, :] = x0 * s + x1 * c
    return out


def causal_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    """(head, dim, T) queries / keys / values -> (head, dim, T), one pass over the sequence."""
    nh, hd, T = q.shape
    o = np.zeros_like(v)
    mask = np.tril(np.ones((T, T), dtype=bool))          # query t attends keys <= t
    for h in range(nh):
        s = (q[h].T @ k[h]) * scale                      # (Tq, Tk)
        s = np.where(mask, s, -np.inf)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        e /= e.sum(axis=1, keepdims=True)
        o[h] = v[h] @ e.T
    return o


class Glm46:
    def __init__(self, path):
        self.g = Gguf(path)
        self.cvec = []   # (layer -> direction[N]) applied after that layer's residual write
        self.cfg = Glm46Config(self.g)
        c = self.cfg
        self.inv = 1.0 / (c.theta ** (np.arange(0, c.n_rot, 2, dtype=np.float64) / c.n_rot))
        self.emb = self.g.tensor("token_embd.weight")                 # (vocab, n_embd)
        self.out_w = self.g.tensor("output.weight")                   # (vocab, n_embd)
        self.out_norm = self.g.tensor("output_norm.weight")
        self.L = []
        for i in range(c.trunk):
            P = f"blk.{i}."
            d = {
                "attn_norm": self.g.tensor(P + "attn_norm.weight"),
                "post1": self.g.tensor(P + "post_attention_norm.weight"),
                "q": self.g.tensor(P + "attn_q.weight"),           # (n_head*hd, n_embd)
                "qb": self.g.tensor(P + "attn_q.bias"),
                "k": self.g.tensor(P + "attn_k.weight"),
                "kb": self.g.tensor(P + "attn_k.bias"),
                "v": self.g.tensor(P + "attn_v.weight"),
                "vb": self.g.tensor(P + "attn_v.bias"),
                "o": self.g.tensor(P + "attn_output.weight"),
                "qn": self.g.tensor(P + "attn_q_norm.weight"),
                "kn": self.g.tensor(P + "attn_k_norm.weight"),
                "moe": i >= c.dense_lead,
            }
            if d["moe"]:
                d["gate"] = self.g.tensor(P + "ffn_gate_inp.weight")   # (n_exp, n_embd)
                d["gbias"] = self.g.tensor(P + "exp_probs_b.bias")
                d["sg"] = self.g.tensor(P + "ffn_gate_shexp.weight")
                d["su"] = self.g.tensor(P + "ffn_up_shexp.weight")
                d["sd"] = self.g.tensor(P + "ffn_down_shexp.weight")
            else:
                d["mg"] = self.g.tensor(P + "ffn_gate.weight")
                d["mu"] = self.g.tensor(P + "ffn_up.weight")
                d["md"] = self.g.tensor(P + "ffn_down.weight")
            self.L.append(d)

    def mtp_load(self) -> None:
        """The pack's NextN (MTP) block: blk.<trunk>.* as a full MoE layer, beside nextn.eh_proj / enorm / hnorm /
        shared_head_norm (the head is the trunk's output.weight)."""
        c = self.cfg
        if c.trunk >= c.n_layer:
            self.mtp = None
            return
        P = f"blk.{c.trunk}."
        d = {
            "attn_norm": self.g.tensor(P + "attn_norm.weight"),
            "post1": self.g.tensor(P + "post_attention_norm.weight"),
            "q": self.g.tensor(P + "attn_q.weight"),
            "qb": self.g.tensor(P + "attn_q.bias"),
            "k": self.g.tensor(P + "attn_k.weight"),
            "kb": self.g.tensor(P + "attn_k.bias"),
            "v": self.g.tensor(P + "attn_v.weight"),
            "vb": self.g.tensor(P + "attn_v.bias"),
            "o": self.g.tensor(P + "attn_output.weight"),
            "qn": self.g.tensor(P + "attn_q_norm.weight"),
            "kn": self.g.tensor(P + "attn_k_norm.weight"),
            "gate": self.g.tensor(P + "ffn_gate_inp.weight"),
            "gbias": self.g.tensor(P + "exp_probs_b.bias"),
            "sg": self.g.tensor(P + "ffn_gate_shexp.weight"),
            "su": self.g.tensor(P + "ffn_up_shexp.weight"),
            "sd": self.g.tensor(P + "ffn_down_shexp.weight"),
            "eh": self.g.tensor(P + "nextn.eh_proj.weight"),
            "enorm": self.g.tensor(P + "nextn.enorm.weight"),
            "hnorm": self.g.tensor(P + "nextn.hnorm.weight"),
            "shnorm": self.g.tensor(P + "nextn.shared_head_norm.weight"),
        }
        self.mtp = d
        self._mk = None   # the block's own K / V rows, (n_kv, hd, T)
        self._mv = None

    def _mtp_block(self, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """The block's rows x (n_embd, T) at positions pos, appended to its own K/V; the residual out (n_embd, T)."""
        c, L = self.cfg, self.mtp
        h = rms_norm(x, L["attn_norm"], c.eps)
        q = L["q"] @ h + L["qb"][:, None]
        k = L["k"] @ h + L["kb"][:, None]
        v = L["v"] @ h + L["vb"][:, None]
        qh = rope(per_head_rms(split_heads(q, c.hd, c.n_head), L["qn"], c.eps), pos, c.n_rot, self.inv)
        kh = rope(per_head_rms(split_heads(k, c.hd, c.n_kv), L["kn"], c.eps), pos, c.n_rot, self.inv)
        vh = split_heads(v, c.hd, c.n_kv)
        self._mk = kh if self._mk is None else np.concatenate([self._mk, kh], axis=2)
        self._mv = vh if self._mv is None else np.concatenate([self._mv, vh], axis=2)
        groups = c.n_head // c.n_kv
        att = causal_attn(qh, self._mk.repeat(groups, axis=0), np.repeat(self._mv, groups, axis=0), 1.0 / np.sqrt(c.hd))
        x = x + L["o"] @ att.reshape(c.n_head * c.hd, att.shape[-1])
        h2 = rms_norm(x, L["post1"], c.eps)
        logits = L["gate"] @ h2
        probs = 1.0 / (1.0 + np.exp(-logits)) if c.gating_sigmoid else np.exp(logits)
        sel = probs + L["gbias"][:, None]
        w = np.zeros((h2.shape[1], c.n_exp), dtype=np.float64)
        for t in range(h2.shape[1]):
            order = np.argsort(-sel[:, t], kind="stable")
            top = order[:c.k_exp]
            wt = probs[top, t]
            if c.norm_w:
                wt = wt / max(wt.sum(), 6.103515625e-5)
            w[t, top] = wt * c.w_scale
        y = np.zeros_like(h2)
        for e in range(c.n_exp):
            rows = np.nonzero(w[:, e] > 0.0)[0]
            if rows.size == 0:
                continue
            gg = self.expert(c.trunk, "gate", e) @ h2[:, rows]
            uu = self.expert(c.trunk, "up", e) @ h2[:, rows]
            dd = self.expert(c.trunk, "down", e) @ swiglu(gg, uu)
            y[:, rows] += dd * w[rows, e][None, :]
        y += L["sd"] @ swiglu(L["sg"] @ h2, L["su"] @ h2)
        return x + y

    def mtp_prefill(self, ids: list[int], h_rows: np.ndarray) -> None:
        """The prompt's entries p = 0 .. T-2: [emb(ids[p+1]) | h_p] run through the block (its K/V built)."""
        c, L = self.cfg, self.mtp
        rows = len(ids) - 1
        if rows <= 0:
            return
        e = rms_norm(self.emb[ids[1:rows + 1]].T.astype(np.float64), L["enorm"], c.eps)
        hh = rms_norm(h_rows[:rows].T.astype(np.float64), L["hnorm"], c.eps)
        R = L["eh"] @ np.concatenate([e, hh], axis=0)
        self._mtp_block(R, np.arange(rows, dtype=np.float64))

    def mtp_step(self, p: int, tok: int, carrier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One chain step at position p: the block reads (h_p = carrier, the token) and drafts; (logits, carry)."""
        c, L = self.cfg, self.mtp
        e = rms_norm(self.emb[[tok]].T.astype(np.float64), L["enorm"], c.eps)
        hh = rms_norm(carrier[:, None].astype(np.float64), L["hnorm"], c.eps)
        R = L["eh"] @ np.concatenate([e, hh], axis=0)
        x = self._mtp_block(R, np.array([float(p)]))
        carry = rms_norm(x, L["shnorm"], c.eps)
        return (self.out_w @ carry).T[0], carry.T[0]

    def expert(self, i, role, e):
        P = f"blk.{i}."
        name = {"gate": P + "ffn_gate_exps.weight", "up": P + "ffn_up_exps.weight",
                "down": P + "ffn_down_exps.weight"}[role]
        return self.g.expert(name, e)

    def load_cvec(self, path: str, scale: float, mode: int, first: int, last: int) -> None:
        """llama.cpp's loader: every direction.<l> of the file times its scale, summed into layer l (0-based;
        l 0 never steered); project normalises the direction and keeps its norm as the scale."""
        g = Gguf(path)
        dirs = {}
        for t in g.g.tensors:
            if not t.name.startswith("direction."):
                continue
            l = int(t.name.split(".")[1])
            if l < 1 or l >= self.cfg.trunk:
                continue
            v = g.tensor(t.name).astype(np.float64)
            dirs[l] = dirs.get(l, np.zeros_like(v)) + scale * v
        for l in range(max(1, first), min(self.cfg.trunk - 1, last) + 1):
            if l not in dirs:
                continue
            d = dirs[l]
            if mode == 0:
                nrm = float(np.linalg.norm(d))
                if nrm <= 0:
                    continue
                self.cvec.append((l, d / nrm, nrm, 0))
            else:
                self.cvec.append((l, d, 1.0, 1))

    def forward(self, ids: list[int], dump_layers: str | None = None, dump_routes: str | None = None) -> np.ndarray:
        c = self.cfg
        T = len(ids)
        dumpf = None
        self.last_min_margin = float("inf")   # the smallest #2/#3 selection-score margin met (a near tie a quantized
                                              # kernel may break the other way - the parity tool allows it)
        pos = np.arange(T, dtype=np.float64)
        x = self.emb[ids].T.astype(np.float64)                        # (n_embd, T)
        for i, L in enumerate(self.L):
            h = rms_norm(x, L["attn_norm"], c.eps)
            q = L["q"] @ h + L["qb"][:, None]
            k = L["k"] @ h + L["kb"][:, None]
            v = L["v"] @ h + L["vb"][:, None]
            qh = rope(per_head_rms(split_heads(q, c.hd, c.n_head), L["qn"], c.eps), pos, c.n_rot, self.inv)
            kh = rope(per_head_rms(split_heads(k, c.hd, c.n_kv), L["kn"], c.eps), pos, c.n_rot, self.inv)
            vh = split_heads(v, c.hd, c.n_kv)                          # (n_kv, hd, T)
            groups = c.n_head // c.n_kv
            vg = np.repeat(vh, groups, axis=0)
            att = causal_attn(qh, kh.repeat(groups, axis=0), vg, 1.0 / np.sqrt(c.hd))
            o = att.reshape(c.n_head * c.hd, T)                        # join heads back
            x = x + L["o"] @ o
            h2 = rms_norm(x, L["post1"], c.eps)
            if L["moe"]:
                logits = L["gate"] @ h2                                 # (n_exp, T)
                probs = 1.0 / (1.0 + np.exp(-logits)) if c.gating_sigmoid else np.exp(logits)
                sel = probs + L["gbias"][:, None]
                w = np.zeros((T, c.n_exp), dtype=np.float64)
                if dump_routes:
                    with open(dump_routes, "ab") as fr:
                        for t in range(T):
                            tp = np.argsort(-sel[:, t], kind="stable")[:c.k_exp]
                            fr.write(struct.pack("<ii", i, T))
                            fr.write(np.asarray(tp, dtype="<i4").tobytes())
                for t in range(T):
                    order = np.argsort(-sel[:, t], kind="stable")
                    if c.k_exp + 1 <= c.n_exp:
                        self.last_min_margin = min(self.last_min_margin,
                                                   float(sel[order[c.k_exp - 1], t] - sel[order[c.k_exp], t]))
                    top = order[:c.k_exp]
                    wt = probs[top, t]
                    if c.norm_w:
                        wt = wt / max(wt.sum(), 6.103515625e-5)
                    wt = wt * c.w_scale
                    w[t, top] = wt
                y = np.zeros_like(h2)
                for e in range(c.n_exp):
                    rows = np.nonzero(w[:, e] > 0.0)[0]
                    if rows.size == 0:
                        continue
                    g = self.expert(i, "gate", e) @ h2[:, rows]
                    u = self.expert(i, "up", e) @ h2[:, rows]
                    d = self.expert(i, "down", e) @ swiglu(g, u)
                    y[:, rows] += d * w[rows, e][None, :]
                sg = L["sg"] @ h2
                su = L["su"] @ h2
                y += L["sd"] @ swiglu(sg, su)
                x = x + y
            else:
                g = L["mg"] @ h2
                u = L["mu"] @ h2
                x = x + L["md"] @ swiglu(g, u)
            for (cl, cd, cs, cm) in self.cvec:
                if cl == i:
                    if cm == 0:
                        dot = cd @ x                       # (T,)
                        x -= cs * cd[:, None] * dot[None, :]
                    else:
                        x += cs * cd[:, None]
            if dump_layers:
                dumpf = open(dump_layers, "wb" if i == 0 else "ab") if dumpf is None else dumpf
                dumpf.write(struct.pack("<ii", i, T))
                dumpf.write(np.ascontiguousarray(x.T, dtype="<f4").tobytes())
        h = rms_norm(x, self.out_norm, c.eps)
        self.last_h = h.T.copy()                                       # (T, n_embd) the MTP's carrier
        if dumpf is not None:
            dumpf.close()
        return (self.out_w @ h).T                                      # (T, vocab)


def main() -> int:
    ap = argparse.ArgumentParser(description="glm4moe reference forward pass")
    ap.add_argument("model", help="a glm4moe GGUF (real, or tools/glm46/glm46_synth_gguf.py's)")
    ap.add_argument("--tokens", default="1,2,3,4", help="comma-separated token ids")
    ap.add_argument("--dump", help="write the logits as the engine's --dump-logits layout (MM3L)")
    ap.add_argument("--greedy", type=int, default=0, help="print this many greedy next ids")
    ap.add_argument("--dump-layers", help="write {layer,T}+residual(T,dim) after every layer")
    ap.add_argument("--dump-routes", help="write {layer,T}+top-k a token (the router's picks)")
    ap.add_argument("--cvec", action="append", default=[], help="FILE:SCALE (repeatable; llama.cpp's control vector)")
    ap.add_argument("--cvec-mode", choices=("add", "project"), default="add")
    ap.add_argument("--cvec-range", nargs=2, type=int, default=None, help="A B (default 1 .. the last trunk layer)")
    a = ap.parse_args()
    ids = [int(t) for t in a.tokens.split(",") if t.strip()]
    m = Glm46(a.model)
    if a.dump_routes:
        pathlib.Path(a.dump_routes).write_bytes(b"")
    for spec in a.cvec:
        path, _, sc = spec.rpartition(":")
        m.load_cvec(path, float(sc), 0 if a.cvec_mode == "project" else 1,
                    *(a.cvec_range if a.cvec_range else (1, m.cfg.trunk - 1)))
    logits = m.forward(ids, a.dump_layers, a.dump_routes)
    if a.dump:
        rows = len(ids)
        with open(a.dump, "wb") as f:
            f.write(b"MM3L")
            f.write(struct.pack("<iii", rows, logits.shape[1], len(ids)))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
            f.write(struct.pack(f"<{rows}i", *range(rows)))
            f.write(np.ascontiguousarray(logits, dtype="<f4").tobytes())
    if a.greedy:
        cur = list(ids)
        out = []
        for _ in range(a.greedy):
            lg = m.forward(cur)
            nxt = int(np.argmax(lg[-1]))
            out.append(nxt)
            cur.append(nxt)
        print("greedy:", " ".join(str(t) for t in out))
    print("last row top-5:", np.round(np.sort(logits[-1])[-5:][::-1], 4).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
