# Чек-лист перед слиянием dev5

- [x] CPU optimizer regression
- [x] checkpoint v1/v2 regression
- [x] Transformer/config regression
- [x] raw TokenDataset regression
- [x] mmap token shard regression
- [x] streaming corpus converter smoke test
- [x] training smoke test
- [x] pipeline smoke test
- [x] CUDA accumulation test добавлен и автоматически skip без NVIDIA
- [ ] CUDA accumulation test должен быть отдельно запущен на реальной NVIDIA перед дорогим pretrain

Этот файл фиксирует границу того, что доказано CI, и того, что требует реального CUDA-устройства.
