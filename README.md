## flax-4d-attention

- Flax 4d attention mask for correct packing
- TPU flash attention
- Supported models: GPT2, Llama, Mistral

### Notice

You should use `input_shape` to prevent error by jax tpu flash attention kernel 🤗🤗
```python
model = FlaxLlamaForCausalLM.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM", from_pt=True, dtype=jnp.bfloat16, input_shape=(1, 128))
output = model(**input_ids).logits
```