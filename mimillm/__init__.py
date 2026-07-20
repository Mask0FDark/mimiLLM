"""mimiLLM: понятные строительные блоки для decoder-only языковых моделей."""

from .api import LanguageModel, ModelConfig, create_model, load_model, save_model
from .audit import (
    DatasetAuditReport,
    audit_dataset,
    normalize_training_text,
    save_dataset_audit,
)
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
from .diagnostics import (
    DEFAULT_ANSWER,
    DEFAULT_QUESTION,
    run_one_pair_sft_acceptance,
)
from .generation import (
    answer_question, generate, generate_response, generate_text, sample_token,
)
from .evaluation import (
    DialogueCheckResult,
    DialogueEvaluationReport,
    evaluate_dialogues,
    save_dialogue_evaluation,
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
from .pipeline import PipelineQualityError, PipelineResult, PipelineStage, train_pipeline
from .safetensors import load_safetensors, save_safetensors
from .tensor import Tensor, is_grad_enabled, no_grad
from .tokenizer import (
    BpeTokenizer,
    ByteTokenizer,
    TokenizerReport,
    UnicodeByteTokenizer,
    analyze_tokenizer,
    create_tokenizer,
    detokenize,
    format_qa_text,
    load_tokenizer,
    pretokenize,
    save_tokenizer,
    save_tokenizer_report,
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
    "DatasetAuditReport",
    "DEFAULT_ANSWER",
    "DEFAULT_QUESTION",
    "DialogueCheckResult",
    "DialogueEvaluationReport",
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
    "PipelineResult",
    "PipelineQualityError",
    "PipelineStage",
    "RMSNorm",
    "ReLU",
    "SGD",
    "Tensor",
    "TokenDataset",
    "TokenizerReport",
    "TransformerBlock",
    "TransformerConfig",
    "TrainingResult",
    "UnicodeByteTokenizer",
    "answer_question",
    "analyze_tokenizer",
    "audit_dataset",
    "create_model",
    "create_tokenizer",
    "cross_entropy",
    "cuda_is_available",
    "discover_question_files",
    "discover_text_files",
    "detokenize",
    "evaluate_dialogues",
    "format_qa_text",
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
    "normalize_training_text",
    "pretokenize",
    "reset_backend",
    "run_one_pair_sft_acceptance",
    "sample_token",
    "save_checkpoint",
    "save_dataset_audit",
    "save_dialogue_evaluation",
    "save_model",
    "save_safetensors",
    "save_tokenizer",
    "save_tokenizer_report",
    "tokenize",
    "train_from_config",
    "train_bpe_tokenizer",
    "train_model",
    "train_pipeline",
    "train_tokenizer_from_config",
    "validation_loss",
]

__version__ = "0.10.2"
