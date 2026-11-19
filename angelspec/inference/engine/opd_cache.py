"""Request-local prefix-cache bounds for OPD hidden capture."""

from contextvars import ContextVar
from functools import wraps

CAPTURE_START = "angelspec_opd_capture_start"
_active_bound = ContextVar("angelspec_opd_cache_bound", default=None)


def capture_bound(request, alignment):
    sampling = getattr(request, "sampling_params", None)
    extra = getattr(sampling, "extra_args", None) or {}
    if CAPTURE_START not in extra:
        return None
    start = extra[CAPTURE_START]
    if type(start) is not int or not 0 <= start < request.num_prompt_tokens:
        raise ValueError(f"Invalid OPD capture start {start!r}")
    return start // alignment * alignment


def install_cache_bound():
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager, "_angelspec_opd_capture_bound", False):
        return
    original = KVCacheManager.get_computed_blocks
    partial = getattr(KVCacheManager, "maybe_extend_partial_block_hit", None)

    @wraps(original)
    def get_computed_blocks(self, request, *args, **kwargs):
        bound = capture_bound(request, self.coordinator.scheduler_block_size)
        if bound is None:
            return original(self, request, *args, **kwargs)
        token = _active_bound.set((self.coordinator, bound))
        try:
            result = original(self, request, *args, **kwargs)
            if result[1] > bound:
                raise RuntimeError(f"OPD prefix hit {result[1]} exceeds capture bound {bound}")
            return result
        finally:
            _active_bound.reset(token)

    def maybe_extend(self, request, computed_blocks, num_new_computed_tokens):
        if _active_bound.get() is not None:
            return num_new_computed_tokens
        return partial(self, request, computed_blocks, num_new_computed_tokens)

    def patch_coordinator(cls):
        method = cls.__dict__.get("find_longest_cache_hit")
        if method is not None:

            @wraps(method)
            def find_hit(self, block_hashes, max_cache_hit_length):
                active = _active_bound.get()
                if active is not None and active[0] is self:
                    max_cache_hit_length = min(max_cache_hit_length, active[1])
                return method(self, block_hashes, max_cache_hit_length)

            cls.find_longest_cache_hit = find_hit
        for child in cls.__subclasses__():
            patch_coordinator(child)

    patch_coordinator(KVCacheCoordinator)
    KVCacheManager.get_computed_blocks = get_computed_blocks
    if partial is not None:
        KVCacheManager.maybe_extend_partial_block_hit = wraps(partial)(maybe_extend)
    KVCacheManager._angelspec_opd_capture_bound = True
