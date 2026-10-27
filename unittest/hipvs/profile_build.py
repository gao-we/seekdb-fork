#!/usr/bin/env python3
"""Trace the same bridge used by seekdb; profiler timings are diagnostic only."""
import ctypes as C
import sys
import numpy as np

bridge = C.CDLL(sys.argv[1])
bridge.seekdb_hipvs_hnsw_build.argtypes = [C.c_void_p, C.c_int64, C.c_int, C.c_int, C.c_int, C.c_int]
bridge.seekdb_hipvs_hnsw_build.restype = C.c_void_p
bridge.seekdb_hipvs_hnsw_free.argtypes = [C.c_void_p]
x = np.random.default_rng(20261027).standard_normal((10000, 128), dtype=np.float32)
h = bridge.seekdb_hipvs_hnsw_build(x.ctypes.data, len(x), 128, 16, 200, 16)
if not h:
    raise RuntimeError('hipVS construction failed; refusing to profile CPU fallback')
bridge.seekdb_hipvs_hnsw_free(h)
