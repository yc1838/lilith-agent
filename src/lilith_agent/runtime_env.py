from __future__ import annotations

import os

SAFE_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def apply_safe_thread_env() -> None:
    for key, value in SAFE_THREAD_ENV.items():
        os.environ[key] = value
