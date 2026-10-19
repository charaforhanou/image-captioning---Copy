"""Byte-level BPE tokenizer for radiology report text.

Trains a self-contained tokenizer (no external pretrained vocabulary
dependency) on the IU X-Ray report corpus, with the special tokens required
by the MCCAG decoder (section markers, EOS/PAD, etc).
"""

from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[REPORT_START]",
    "[FINDING]",
    "[END_FINDING]",
    "[IMPRESSION]",
    "[END_IMPRESSION]",
    "[EOS]",
    "[MEMORY]",
]


class MedicalTokenizer:
    """Wraps a byte-level BPE :class:`tokenizers.Tokenizer` with helpers.

    Args:
        tokenizer: The underlying trained ``tokenizers.Tokenizer``.
        max_length: Sequences are padded/truncated to this length.
    """

    def __init__(self, tokenizer: Tokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.enable_padding(
            pad_id=self.pad_id, pad_token="[PAD]", length=max_length
        )
        self.tokenizer.enable_truncation(max_length=max_length)

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        vocab_size: int = 8000,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        max_length: int = 512,
        min_frequency: int = 2,
    ) -> "MedicalTokenizer":
        """Train a new byte-level BPE tokenizer from an iterable of texts.

        Args:
            texts: Training corpus (e.g. structured report target sequences).
            vocab_size: Target vocabulary size.
            special_tokens: Tokens guaranteed to receive their own ids.
            max_length: Padding/truncation length for ``encode``.
            min_frequency: Minimum pair frequency for BPE merges.

        Returns:
            A fitted :class:`MedicalTokenizer`.
        """
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        logger.info(f"Trained BPE tokenizer with vocab_size={tokenizer.get_vocab_size()}")
        return cls(tokenizer, max_length=max_length)

    @classmethod
    def load(cls, save_dir: Union[str, Path], max_length: int = 512) -> "MedicalTokenizer":
        """Load a previously trained tokenizer from ``save_dir/tokenizer.json``."""
        path = Path(save_dir) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer, max_length=max_length)

    def save(self, save_dir: Union[str, Path]) -> None:
        """Save the tokenizer to ``save_dir/tokenizer.json``."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(save_dir / "tokenizer.json"))
        logger.info(f"Saved tokenizer to {save_dir / 'tokenizer.json'}")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Unknown special token: {token!r}")
        return token_id

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id("[PAD]")

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id("[UNK]")

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id("[EOS]")

    @property
    def report_start_id(self) -> int:
        return self.tokenizer.token_to_id("[REPORT_START]")

    @property
    def finding_id(self) -> int:
        return self.tokenizer.token_to_id("[FINDING]")

    @property
    def end_finding_id(self) -> int:
        return self.tokenizer.token_to_id("[END_FINDING]")

    @property
    def impression_id(self) -> int:
        return self.tokenizer.token_to_id("[IMPRESSION]")

    @property
    def end_impression_id(self) -> int:
        return self.tokenizer.token_to_id("[END_IMPRESSION]")

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode ``text`` en ``input_ids``/``attention_mask`` paddés/tronqués.

        Encadre le texte par [REPORT_START] au début et [EOS] à la fin, pour que
        l'entraînement et la génération (beam search) partagent le même format :
        le modèle apprend à démarrer après [REPORT_START] et à terminer sur [EOS].
        """
        text_framed = "[REPORT_START] " + text.strip() + " [EOS]"
        encoding = self.tokenizer.encode(text_framed)
        return {
            "input_ids": torch.tensor(encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
        }
    
    
    def decode(self, ids: Union[Sequence[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token ids back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def section_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the per-token FINDINGS(0)/IMPRESSION(1) section id.

        Used by the decoder's report-type embedding. Tokens before
        ``[IMPRESSION]`` (inclusive of the section's own special tokens) are
        labeled 0; tokens from ``[IMPRESSION]`` onward are labeled 1.

        Args:
            input_ids: ``(..., seq_len)`` tensor of token ids.

        Returns:
            ``(..., seq_len)`` LongTensor of 0/1 section ids.
        """
        is_impression = input_ids == self.impression_id
        # cumulative sum > 0 once [IMPRESSION] has been seen (inclusive)
        section = torch.cumsum(is_impression.long(), dim=-1)
        section = (section > 0).long()
        return section