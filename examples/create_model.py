"""Минимальный пример создания собственной конфигурации mimiLLM."""

from mimillm import ByteTokenizer, ModelConfig, create_model


config = ModelConfig(
    context_length=64,
    d_model=32,
    n_layers=2,
    n_heads=4,
    d_mlp=96,
    batch_size=1,
    steps=100,
)
model = create_model(config)
tokenizer = ByteTokenizer()
tokens = tokenizer.encode("Hello, mimiLLM!", add_bos=True)
logits = model([tokens])

print(f"parameters={model.parameter_count():,}")
print(f"logits_shape={logits.shape}")
