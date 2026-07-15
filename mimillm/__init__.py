"""mimiLLM: понятные строительные блоки для decoder-only языковых моделей."""

from .api import LanguageModel, ModelConfig, create_model, load_model, save_model
from .attention import MultiHeadCausalSelfAttention
from .backend import get_backend, reset_backend
from .backend_cuda import CudaBackend, is_available as cuda_is_available
from .checkpoint import CheckpointData, load_checkpoint, save_checkpoint
from .dataset import (
    TokenDataset,
    discover_question_files,
    discover_text_files,
    load_qa_text,
    load_text_documents,
)
from .generation import (
    answer_question, generate, generate_response, generate_text, sample_token,
)
from .hailo import (
    HailoHefInfo,
    HailoRuntimeInfo,
    hailo_is_available,
    inspect_hailo_hef,
    inspect_hailo_runtime,
)
from .layers import Embedding, FeedForward, Linear, RMSNorm, ReLU
from .losses import cross_entropy
from .module import Module
from .optim import AdamW, Optimizer, SGD
from .parameter import Parameter
from .safetensors import load_safetensors, save_safetensors
from .tensor import Tensor, is_grad_enabled, no_grad
from .tokenizer import (
    BpeTokenizer,
    ByteTokenizer,
    UnicodeByteTokenizer,
    create_tokenizer,
    detokenize,
    load_tokenizer,
    pretokenize,
    save_tokenizer,
    tokenize,
    train_bpe_tokenizer,
)
from .training import (
    TrainingResult,
    train_from_config,
    train_model,
    train_tokenizer_from_config,
    validation_loss,
)
from .transformer import DecoderTransformer, TransformerBlock, TransformerConfig


__all__ = [
    "AdamW",
    "BpeTokenizer",
    "ByteTokenizer",
    "CheckpointData",
    "CudaBackend",
    "DecoderTransformer",
    "Embedding",
    "FeedForward",
    "HailoHefInfo",
    "HailoRuntimeInfo",
    "LanguageModel",
    "Linear",
    "ModelConfig",
    "Module",
    "MultiHeadCausalSelfAttention",
    "Optimizer",
    "Parameter",
    "RMSNorm",
    "ReLU",
    "SGD",
    "Tensor",
    "TokenDataset",
    "TransformerBlock",
    "TransformerConfig",
    "TrainingResult",
    "UnicodeByteTokenizer",
    "answer_question",
    "create_model",
    "create_tokenizer",
    "cross_entropy",
    "cuda_is_available",
    "discover_question_files",
    "discover_text_files",
    "detokenize",
    "generate",
    "generate_response",
    "generate_text",
    "get_backend",
    "hailo_is_available",
    "is_grad_enabled",
    "inspect_hailo_hef",
    "inspect_hailo_runtime",
    "load_checkpoint",
    "load_model",
    "load_qa_text",
    "load_safetensors",
    "load_tokenizer",
    "load_text_documents",
    "no_grad",
    "pretokenize",
    "reset_backend",
    "sample_token",
    "save_checkpoint",
    "save_model",
    "save_safetensors",
    "save_tokenizer",
    "tokenize",
    "train_from_config",
    "train_bpe_tokenizer",
    "train_model",
    "train_tokenizer_from_config",
    "validation_loss",
]

__version__ = "0.7.1"
