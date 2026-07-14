"""mimiLLM: понятные строительные блоки для decoder-only языковых моделей."""

from .api import LanguageModel, ModelConfig, create_model, load_model, save_model
from .attention import MultiHeadCausalSelfAttention
from .backend import get_backend, reset_backend
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
from .layers import Embedding, FeedForward, Linear, RMSNorm, ReLU
from .losses import cross_entropy
from .module import Module
from .optim import AdamW, Optimizer, SGD
from .parameter import Parameter
from .safetensors import load_safetensors, save_safetensors
from .tensor import Tensor, is_grad_enabled, no_grad
from .tokenizer import ByteTokenizer
from .training import TrainingResult, train_from_config, train_model, validation_loss
from .transformer import DecoderTransformer, TransformerBlock, TransformerConfig


__all__ = [
    "AdamW",
    "ByteTokenizer",
    "CheckpointData",
    "DecoderTransformer",
    "Embedding",
    "FeedForward",
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
    "answer_question",
    "create_model",
    "cross_entropy",
    "discover_question_files",
    "discover_text_files",
    "generate",
    "generate_response",
    "generate_text",
    "get_backend",
    "is_grad_enabled",
    "load_checkpoint",
    "load_model",
    "load_qa_text",
    "load_safetensors",
    "load_text_documents",
    "no_grad",
    "reset_backend",
    "sample_token",
    "save_checkpoint",
    "save_model",
    "save_safetensors",
    "train_from_config",
    "train_model",
    "validation_loss",
]

__version__ = "0.3.4"
