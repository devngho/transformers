## branch: flax-4d-attention

> [!WARNING]
> It's very **experimental**. Be careful!

HF's missing flax support

### Main feature

- Flax 4d attention mask for correct packing (by position_ids or directly providing mask)
- TPU flash attention
- Supported models: BERT, XLM-Roberta, GPT2, Llama, Mistral
- RoPE correction for flax (longrope...)
- lazy causal mask to prevent oom

### Notice

You should use `input_shape` and `mesh` to prevent error by jax tpu flash attention kernel 🤗🤗
```python
model = FlaxLlamaForCausalLM.from_pretrained(
    "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    jnp.bfloat16, # dtype
    mesh, # should be added (mesh) if you use flash_attention_tpu.
    (1, 128), # should be added (input_shape for initial forward pass) if you use flash_attention_tpu
    from_pt=True,
    attn_implementation='flash_attention_tpu' # or eager (defaults to eager)
)
output = model(**input_ids).logits
```
