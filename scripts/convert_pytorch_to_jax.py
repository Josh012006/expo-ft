"""
Convert RLinf π₀/π₀.₅ PyTorch safetensors checkpoint to JAX orbax format.
Inverts: expo_ft/agents/vla/openpi/examples/convert_jax_model_to_pytorch.py

Usage:
    python scripts/convert_pytorch_to_jax.py \
        --input  assets/RLinf-Pi05-ManiSkill-25Main-SFT/model.safetensors \
        --output assets/RLinf-Pi05-ManiSkill-25Main-SFT/params \
        [--pi0]           # use π₀ layout (default: π₀.₅)
        [--precision float32]
"""

import argparse
import pathlib
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.traverse_util import unflatten_dict
import safetensors.torch as st


def to_np(tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


# ── Vision encoder (PaliGemma SigLIP) ────────────────────────────────────────

def remap_paligemma(sd: dict) -> dict:
    out = {}

    # patch_embedding: JAX kernel (kH, kW, C_in, C_out), PyTorch weight (C_out, C_in, kH, kW)
    # convert_script: state_dict[pytorch] = state_dict[jax].transpose(3,2,0,1)
    # inverse: pytorch (C_out, C_in, kH, kW) → jax (kH, kW, C_in, C_out) = transpose(2,3,1,0)
    w = to_np(sd.pop("paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"))
    out["img/embedding/kernel"] = np.transpose(w, (2, 3, 1, 0))
    out["img/embedding/bias"] = to_np(
        sd.pop("paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"))

    # pos_embedding: JAX (1, num_positions, hidden), PyTorch (num_positions, hidden)
    pe = to_np(sd.pop("paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"))
    out["img/pos_embedding"] = pe.reshape(1, *pe.shape)

    # Count vision layers
    num_vis = sum(1 for k in sd if "vision_tower.vision_model.encoder.layers." in k and ".layer_norm1.weight" in k)

    # LayerNorm: JAX scale 1D, PyTorch weight 1D — NO transpose for 1D
    out["img/Transformer/encoderblock/LayerNorm_0/scale"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"))
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/LayerNorm_0/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"))
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/LayerNorm_1/scale"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"))
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/LayerNorm_1/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"))
        for i in range(num_vis)])

    # MLP: JAX kernel (in, out), PyTorch weight (out, in) → transpose
    out["img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight")).T
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight")).T
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        for i in range(num_vis)])

    # Attention projections: JAX kernel (in, out), PyTorch weight (out, in) → transpose
    # q/k/v/out all follow (out_dim, in_dim) in PyTorch → .T in JAX
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight")).T
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight")).T
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight")).T
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight")).T
        for i in range(num_vis)])
    out["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))
        for i in range(num_vis)])

    # encoder_norm: 1D, no transpose
    out["img/Transformer/encoder_norm/scale"] = to_np(
        sd.pop("paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"))
    out["img/Transformer/encoder_norm/bias"] = to_np(
        sd.pop("paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"))

    # multimodal projector: 2D kernel → transpose
    out["img/head/kernel"] = to_np(
        sd.pop("paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight")).T
    out["img/head/bias"] = to_np(
        sd.pop("paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"))

    # LLM text decoder
    out["llm/embedder/input_embedding"] = to_np(
        sd.pop("paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"))

    num_llm = sum(1 for k in sd if "language_model.layers." in k and ".self_attn.q_proj.weight" in k)

    # Get dims from first layer
    q0 = to_np(sd["paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"])
    hidden = q0.shape[1]
    total_q_dim = q0.shape[0]   # num_heads * head_dim
    head_dim_q = to_np(sd["paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight"]).shape[0]
    num_heads = total_q_dim // head_dim_q

    # q_einsum: JAX (L, num_heads, hidden, head_dim)
    # convert_script: q_proj = q_einsum[i].transpose(0,2,1).reshape(num_heads*head_dim, hidden)
    # inverse: q_proj (num_heads*head_dim, hidden) → reshape(num_heads, head_dim, hidden) → transpose(0,2,1)
    q_einsums = []
    for i in range(num_llm):
        q = to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"))
        q_einsums.append(q.reshape(num_heads, head_dim_q, hidden).transpose(0, 2, 1))  # (num_heads, hidden, head_dim)
    out["llm/layers/attn/q_einsum/w"] = np.stack(q_einsums)  # (L, num_heads, hidden, head_dim)

    # kv_einsum: JAX (L, 2, 1, head_dim, hidden) — note: NOT transposed in original!
    # convert_script: k_proj = kv_einsum[i,0,0].transpose() → PyTorch (hidden, head_dim).T = (head_dim, hidden)
    # Wait: k_proj.weight shape is (256, 1024) = (head_dim, hidden)
    # So kv_einsum[i,0,0] = k_proj.T → shape (hidden, head_dim)... but JAX expects (head_dim, hidden)?
    # Let's re-read: k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
    # kv_einsum[i,0,0].transpose() = k_proj → k_proj = (head_dim, hidden) → kv_einsum[i,0,0] = (hidden, head_dim)
    # So JAX kv_einsum shape per layer: (2, 1, hidden, head_dim)
    # inverse: k_proj (head_dim, hidden) → .T → (hidden, head_dim) = kv_einsum[i,0,0]
    kv_einsums = []
    for i in range(num_llm):
        k = to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"))
        v = to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"))
        # k shape: (head_dim, hidden) → .T → (hidden, head_dim)
        kv = np.stack([k.T[np.newaxis], v.T[np.newaxis]])  # (2, 1, hidden, head_dim)
        kv_einsums.append(kv)
    out["llm/layers/attn/kv_einsum/w"] = np.stack(kv_einsums)  # (L, 2, 1, hidden, head_dim)

    # attn_vec_einsum: JAX (L, num_heads, head_dim, hidden)
    # convert_script: o_proj = attn_vec_einsum[i].transpose(2,0,1).reshape(num_heads*head_dim, hidden)
    # JAX shape: (num_heads, head_dim, hidden) → transpose(2,0,1) → (hidden, num_heads, head_dim) → reshape(num_heads*head_dim, hidden)??
    # That doesn't work. Let me reread:
    # "o_proj_weight_reshaped = attn_vec_einsum[i].transpose(2,0,1).reshape(num_heads*head_dim, hidden)"
    # attn_vec_einsum[i]: if shape is (head_dim, num_heads, hidden):
    #   .transpose(2,0,1) → (hidden, head_dim, num_heads) — still wrong shape
    # Actually looking at the code again in the original:
    # o_proj_weight_reshaped = (attn_vec_einsum[i].transpose(2,0,1).reshape(num_heads*head_dim, hidden))
    # state_dict[f"...o_proj.weight"] = o_proj_weight_reshaped  ← this IS the weight directly (no extra .T)
    # o_proj.weight = (1024, 2048) = (hidden, num_heads*head_dim)
    # So: attn_vec_einsum[i].transpose(2,0,1).reshape(num_heads*head_dim, hidden) should = (hidden, num_heads*head_dim)
    # That means reshape gives (hidden, num_heads*head_dim)... only if attn_vec after transpose is (hidden, num_heads*head_dim)
    # attn_vec_einsum[i] shape must be such that transpose(2,0,1) gives (hidden, *, *) then reshape to (hidden, num_heads*head_dim)
    # If attn_vec_einsum[i] = (num_heads, head_dim, hidden): transpose(2,0,1) = (hidden, num_heads, head_dim) → reshape(hidden, num_heads*head_dim)
    # Then o_proj_weight = (hidden, num_heads*head_dim) ← matches (1024, 2048) ✓
    # inverse: o_proj (1024, 2048) = (hidden, num_heads*head_dim)
    #   → reshape(hidden, num_heads, head_dim) → transpose(1,2,0) → (num_heads, head_dim, hidden)
    o_einsums = []
    for i in range(num_llm):
        o = to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"))
        # o shape: (hidden, num_heads*head_dim) = (1024, 2048)
        o_einsums.append(o.reshape(hidden, num_heads, head_dim_q).transpose(1, 2, 0))  # (num_heads, head_dim, hidden)
    out["llm/layers/attn/attn_vec_einsum/w"] = np.stack(o_einsums)

    # MLP: JAX gating_einsum (L, 2, in, out), linear (L, in, out)
    # convert_script: gate_proj = gating_einsum[i,0].transpose() → PyTorch (out, in)
    # inverse: gate_proj (out, in) → .T → (in, out) = gating_einsum[i,0]
    gate_list, up_list, down_list = [], [], []
    for i in range(num_llm):
        gate_list.append(to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight")).T)
        up_list.append(to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight")).T)
        down_list.append(to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight")).T)

    out["llm/layers/mlp/gating_einsum"] = np.stack([np.stack([g, u]) for g, u in zip(gate_list, up_list)])
    out["llm/layers/mlp/linear"] = np.stack(down_list)

    # LayerNorm LLM: 1D scale, no transpose
    out["llm/layers/pre_attention_norm/scale"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"))
        for i in range(num_llm)])
    out["llm/layers/pre_ffw_norm/scale"] = np.stack([
        to_np(sd.pop(f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"))
        for i in range(num_llm)])

    out["llm/final_norm/scale"] = to_np(
        sd.pop("paligemma_with_expert.paligemma.model.language_model.norm.weight"))

    return out


# ── Gemma expert ──────────────────────────────────────────────────────────────

def remap_gemma_expert(sd: dict, num_expert: int, pi05: bool) -> dict:
    out = {}
    num_layers = sum(1 for k in sd if "gemma_expert.model.layers." in k and ".self_attn.q_proj.weight" in k)

    q0 = to_np(sd[f"paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"])
    hidden = q0.shape[1]
    total_q = q0.shape[0]
    head_dim = to_np(sd[f"paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj.weight"]).shape[0]
    num_heads = total_q // head_dim

    q_list, kv_list, o_list = [], [], []
    gate_list, up_list, down_list = [], [], []
    ln_in_b, ln_in_k, ln_post_b, ln_post_k = [], [], [], []

    for i in range(num_layers):
        # q einsum
        q = to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"))
        q_list.append(q.reshape(num_heads, head_dim, hidden).transpose(0, 2, 1))  # (num_heads, hidden, head_dim)

        # kv einsum
        k = to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"))
        v = to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"))
        kv_list.append(np.stack([k.T[np.newaxis], v.T[np.newaxis]]))  # (2, 1, hidden, head_dim)

        # attn_vec einsum
        o = to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"))
        o_list.append(o.reshape(hidden, num_heads, head_dim).transpose(1, 2, 0))  # (num_heads, head_dim, hidden)

        # MLP
        gate_list.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight")).T)
        up_list.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight")).T)
        down_list.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight")).T)

        # LayerNorms
        if pi05:
            # π₀.₅ uses adaptive norm with Dense layers
            ln_in_b.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias")))
            ln_in_k.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight")).T)
            ln_post_b.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias")))
            ln_post_k.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight")).T)
        else:
            ln_in_b.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight")))
            ln_post_b.append(to_np(sd.pop(f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight")))

    out[f"llm/layers/attn/q_einsum_{num_expert}/w"] = np.stack(q_list)
    out[f"llm/layers/attn/kv_einsum_{num_expert}/w"] = np.stack(kv_list)
    out[f"llm/layers/attn/attn_vec_einsum_{num_expert}/w"] = np.stack(o_list)
    out[f"llm/layers/mlp_{num_expert}/gating_einsum"] = np.stack([np.stack([g, u]) for g, u in zip(gate_list, up_list)])
    out[f"llm/layers/mlp_{num_expert}/linear"] = np.stack(down_list)

    if pi05:
        out[f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias"]   = np.stack(ln_in_b)
        out[f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel"] = np.stack(ln_in_k)
        out[f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias"]         = np.stack(ln_post_b)
        out[f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel"]       = np.stack(ln_post_k)
        out[f"llm/final_norm_{num_expert}/Dense_0/bias"]   = to_np(sd.pop("paligemma_with_expert.gemma_expert.model.norm.dense.bias"))
        out[f"llm/final_norm_{num_expert}/Dense_0/kernel"] = to_np(sd.pop("paligemma_with_expert.gemma_expert.model.norm.dense.weight")).T
    else:
        out[f"llm/layers/pre_attention_norm_{num_expert}/scale"] = np.stack(ln_in_b)
        out[f"llm/layers/pre_ffw_norm_{num_expert}/scale"]       = np.stack(ln_post_b)
        out[f"llm/final_norm_{num_expert}/scale"] = to_np(sd.pop("paligemma_with_expert.gemma_expert.model.norm.weight"))

    return out


# ── Projection layers ─────────────────────────────────────────────────────────

def remap_projections(sd: dict, pi05: bool) -> dict:
    """JAX kernel = (in, out), PyTorch weight = (out, in) → transpose"""
    out = {}
    keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"] if pi05 else \
           ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]
    for key in keys:
        out[f"{key}/kernel"] = to_np(sd.pop(f"{key}.weight")).T
        out[f"{key}/bias"]   = to_np(sd.pop(f"{key}.bias"))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str, pi05: bool, precision: str):
    print(f"Loading: {input_path}")
    raw = st.load_file(input_path)
    sd = dict(raw)  # mutable copy
    print(f"  {len(sd)} tensors")

    jax_params = {}
    print("Remapping PaliGemma...")
    jax_params.update(remap_paligemma(sd))
    print("Remapping Gemma expert...")
    jax_params.update(remap_gemma_expert(sd, num_expert=1, pi05=pi05))
    print("Remapping projections...")
    jax_params.update(remap_projections(sd, pi05=pi05))

    # Skip tied lm_head weights
    for k in list(sd.keys()):
        if "lm_head" in k:
            print(f"  Skipping tied: {k}")
            sd.pop(k)

    if sd:
        print(f"WARNING: {len(sd)} unmapped keys:")
        for k in sorted(sd): print(f"  {k}")

    # Convert to JAX arrays
    dtype = jnp.bfloat16 if precision == "bfloat16" else jnp.float32
    print(f"Converting to {precision}...")
    jax_params = {k: jnp.array(v, dtype=dtype) for k, v in jax_params.items()}

    # Nest and wrap under "params"
    nested = unflatten_dict({tuple(k.split("/")): v for k, v in jax_params.items()})
    params = {"params": nested}

    # Save
    out = pathlib.Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving orbax to: {out}")
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str(out), params)

    print(f"Done — {len(jax_params)} tensors saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,  help="Path to model.safetensors")
    parser.add_argument("--output",    required=True,  help="Output orbax directory")
    parser.add_argument("--pi0",       action="store_true", help="Use π₀ layout (default: π₀.₅)")
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        pi05=not args.pi0,
        precision=args.precision,
    )
