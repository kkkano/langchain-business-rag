from __future__ import annotations

from threading import Lock

from langchain_core.caches import InMemoryCache
from langchain_core.globals import get_llm_cache, set_llm_cache

from .config import Settings
from .models import CacheStats


class ObservableInMemoryCache(InMemoryCache):
    def __init__(self):
        super().__init__()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._writes = 0

    def lookup(self, prompt: str, llm_string: str):
        value = super().lookup(prompt, llm_string)
        with self._lock:
            if value is None:
                self._misses += 1
            else:
                self._hits += 1
        return value

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        with self._lock:
            self._writes += 1
        super().update(prompt, llm_string, return_val)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._writes = 0

    def snapshot(self, enabled: bool = True) -> CacheStats:
        with self._lock:
            return CacheStats(
                enabled=enabled,
                backend="InMemoryCache",
                entries=len(self._cache),
                hits=self._hits,
                misses=self._misses,
                writes=self._writes,
            )


class LLMCacheService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache = None
        self.configure()

    def configure(self) -> None:
        if not self.settings.enable_llm_cache:
            set_llm_cache(None)
            self._cache = None
            return

        current_cache = get_llm_cache()
        if isinstance(current_cache, ObservableInMemoryCache):
            self._cache = current_cache
            return

        self._cache = ObservableInMemoryCache()
        set_llm_cache(self._cache)

    def reset(self) -> CacheStats:
        if self._cache is None:
            return self.snapshot()
        self._cache.clear()
        return self.snapshot()

    def snapshot(self) -> CacheStats:
        if self._cache is None:
            return CacheStats(enabled=False, backend="disabled")
        return self._cache.snapshot(enabled=True)
