"""Run trajectory extraction on the encoder's fast TGN path (bita/fast, level 1).

Extraction (HostTrajectoryExtractor) drives the encoder in streaming mode --
update_memory_for / get_host_embeddings / store_interactions, window by
window, on the CPU in worker processes -- and it was still on the reference
code: per-node Python message lists, ~200k small torch.stack calls per 100k
records. That is the code the encoder's training path replaced with
bita/fast; its level 1 is an eager re-expression whose forward is
bit-identical to the reference, and FastTGNMixin already implements the
streaming calls.

Measured on cached real captures (one CPU thread):
  CTU-13 #7   114k records   8.8 s ->  5.1 s
  CTU-13 #4   1.1M records 114.6 s -> 55.6 s
with every extracted column identical (features, node ids, hosts, windows,
risk): tests/test_fast_extraction.py.

CYBERWORLD_FAST_EXTRACT=0 keeps the reference path. An encoder the fast path
does not support (no memory, another aggregator) is left on the reference
path, with a note.
"""

import os


def fast_extraction_enabled() -> bool:
    return os.environ.get("CYBERWORLD_FAST_EXTRACT", "1") not in ("0", "false", "False", "")


def enable_fast_extraction(tgn, log=print):
    """Switch `tgn` (fresh, before any use) to the fast streaming path, level 1."""
    if tgn is None or not fast_extraction_enabled():
        return tgn
    import sys
    bita = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bita")
    if bita not in sys.path:
        sys.path.insert(0, bita)
    try:
        from fast import enable_fast_tgn
        enable_fast_tgn(tgn, level=1)
    except (ValueError, ImportError) as e:
        if log is not None:
            log(f"fast extraction unavailable ({e}); using the reference path")
    return tgn
