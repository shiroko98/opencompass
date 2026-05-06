from typing import Dict, List, Optional, Union
import os.path as osp
import re
import numpy as np
import torch
import torch.nn.functional as F

from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.model import RWKV

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList


PromptType = Union[PromptList, str]


class Mobius(BaseModel):
    """RWKV / Mobius model wrapper for OpenCompass (>=0.5.x)."""

    def __init__(
        self,
        path: str,
        tokenizer_path: Optional[str] = None,
        max_seq_len: int = 2048,
        generation_kwargs: Optional[Dict] = None,
        meta_template: Optional[Dict] = None,
        strategy: str = "cuda fp16",
        stop_words: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Args:
            path: model path
            tokenizer_path: rwkv vocab path
            max_seq_len: max context length
            generation_kwargs: sampling args
            meta_template: OpenCompass meta template
            strategy: ChatRWKV rwkv package strategy string
            stop_words: strings used to truncate generated text
            **kwargs: ignored (for OpenCompass compatibility)
        """
        super().__init__(path=path, max_seq_len=max_seq_len, meta_template=meta_template)

        self.logger = get_logger()
        self.max_seq_len = max_seq_len
        self.strategy = strategy
        self.stop_words = stop_words or []

        # ---- load model ----
        self.model = RWKV(model=path, strategy=strategy)

        self.pipeline = PIPELINE(self.model, self._normalize_tokenizer_path(tokenizer_path))

        self.tokenizer = self.pipeline
        self.template_parser = LMTemplateParser(meta_template)

        # ---- generation args (safe defaults) ----
        generation_kwargs = generation_kwargs or {}

        self.args = PIPELINE_ARGS(
            temperature=generation_kwargs.get("temperature", 0.8),
            top_p=generation_kwargs.get("top_p", 0.8),
            top_k=generation_kwargs.get("top_k", 0),
            alpha_frequency=generation_kwargs.get("alpha_frequency", 0.0),
            alpha_presence=generation_kwargs.get("alpha_presence", 0.0),
            alpha_decay=0.996,
            token_ban=[0],
            token_stop=[],
            chunk_len=256,
        )

        self.logger.info("Mobius (RWKV) model initialized.")

    def _normalize_tokenizer_path(self, tokenizer_path: Optional[str]) -> str:
        # ChatRWKV's PIPELINE special-cases magic tokenizer names like
        # `rwkv_vocab_v20230424`. Passing the `.txt` path falls through to
        # `Tokenizer.from_file(...)`, which expects a JSON tokenizer file.
        if tokenizer_path is None:
            return 'rwkv_vocab_v20230424'
        basename = osp.basename(tokenizer_path)
        if re.fullmatch(r'rwkv_vocab_v\d+\.txt', basename):
            return osp.splitext(basename)[0]
        return tokenizer_path

    def _apply_stop_words(self, text: str, stop_words: Optional[List[str]]) -> str:
        stop_words = [word for word in (stop_words or []) if word]
        if not stop_words:
            return text

        stop_positions = [
            text.find(word) for word in stop_words if text.find(word) != -1
        ]
        if not stop_positions:
            return text
        return text[:min(stop_positions)]

    # =========================
    # Generation (GEN datasets)
    # =========================
    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        stopping_criteria: Optional[List[str]] = None,
    ) -> List[str]:
        results = []
        effective_stop_words = list(self.stop_words)
        if stopping_criteria:
            effective_stop_words.extend(stopping_criteria)
        for prompt in inputs:
            out = self.pipeline.generate(
                prompt,
                token_count=max_out_len,
                args=self.args,
            )
            results.append(self._apply_stop_words(out, effective_stop_words).strip())
        return results

    # =========================
    # Perplexity (PPL datasets)
    # =========================
    def get_ppl(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None,
    ) -> List[float]:
        """
        Used by MMLU / CEval / CMMLU PPL evaluation.
        """
        ce_loss_list = []

        if mask_length is None:
            mask_length = [None] * len(inputs)

        for text, text_mask_length in zip(inputs, mask_length):
            tokens_list = self.pipeline.encode(text)
            tokens = tokens_list[-self.max_seq_len:]
            if len(tokens) <= 1:
                ce_loss_list.append(float('inf'))
                continue

            # RWKV forward
            logits, _ = self.model.forward(tokens, None, full_output=True)
            shift_logits = logits[:-1, :].contiguous().float()
            shift_labels = torch.tensor(tokens[1:], device=shift_logits.device, dtype=torch.long)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")

            if text_mask_length is not None:
                start = max(text_mask_length - 1, 0)
                if start >= len(loss):
                    ce_loss_list.append(float('inf'))
                    continue
                loss = loss[start:]

            ce_loss_list.append(loss.mean().detach().cpu().item())

        return np.array(ce_loss_list, dtype=np.float32)

    # =========================
    # Token length helper
    # =========================
    def get_token_len(self, prompt: str) -> int:
        return len(self.pipeline.encode(prompt))
