from __future__ import annotations

import math
import os
from typing import List

from .models import SourceDocument


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._load_error = None

    def rerank(
        self,
        query: str,
        candidates: List[SourceDocument],
        top_k: int,
    ) -> List[SourceDocument]:
        if len(candidates) <= 1:
            return candidates[:top_k]

        model = self._get_model()
        pairs = [(query, candidate.content) for candidate in candidates]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        reranked: List[SourceDocument] = []
        for candidate, score in zip(candidates, scores):
            reranked.append(
                SourceDocument(
                    source_id=candidate.source_id,
                    document_id=candidate.document_id,
                    source_name=candidate.source_name,
                    source_path=candidate.source_path,
                    source_type=candidate.source_type,
                    chunk_index=candidate.chunk_index,
                    segment_label=candidate.segment_label,
                    content=candidate.content,
                    score=self._normalize_score(score),
                )
            )

        reranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return reranked[:top_k]

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise RuntimeError(
                f"Cross-Encoder reranker 加载失败: {self.model_name}"
            ) from self._load_error

        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as exc:
            self._load_error = exc
            raise RuntimeError(
                f"Cross-Encoder reranker 加载失败: {self.model_name}。"
                "首次运行通常需要联网下载模型，或改用本地已缓存模型。"
            ) from exc

    @staticmethod
    def _normalize_score(score) -> float:
        value = float(score)
        if 0.0 <= value <= 1.0:
            return value
        return 1.0 / (1.0 + math.exp(-value))
