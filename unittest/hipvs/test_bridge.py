#!/usr/bin/env python3
"""GPU bridge correctness, persistence, filtering and unavailable-GPU tests."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bridge', required=True)
    p.add_argument('--hidden-gpu-child', action='store_true')
    args = p.parse_args()
    h = C.CDLL(args.bridge)
    h.seekdb_hipvs_hnsw_build.argtypes = [C.c_void_p, C.c_int64, C.c_int, C.c_int, C.c_int, C.c_int]
    h.seekdb_hipvs_hnsw_build.restype = C.c_void_p
    h.seekdb_hipvs_hnsw_free.argtypes = [C.c_void_p]
    h.seekdb_hipvs_hnsw_search.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int64)]
    h.seekdb_hipvs_hnsw_verify.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int]
    h.seekdb_hipvs_hnsw_save.argtypes = [C.c_void_p, C.c_char_p]
    h.seekdb_hipvs_hnsw_load.argtypes = [C.c_char_p, C.c_int]
    h.seekdb_hipvs_hnsw_load.restype = C.c_void_p
    h.seekdb_hipvs_hnsw_error.restype = C.c_char_p
    rng = np.random.default_rng(20261027)
    x = rng.standard_normal((1024, 64), dtype=np.float32)
    if args.hidden_gpu_child:
        assert not h.seekdb_hipvs_hnsw_build(x.ctypes.data, len(x), 64, 16, 200, 4)
        assert b'fallback disabled' in h.seekdb_hipvs_hnsw_error()
        print('hidden_gpu: explicit failure, no CPU fallback')
        return
    callback_type = C.CFUNCTYPE(C.c_int, C.c_uint64, C.c_void_p)
    even = callback_type(lambda label, _: int(label % 2 == 0))
    recalls = []
    for iteration in range(3):
        handle = h.seekdb_hipvs_hnsw_build(x.ctypes.data, len(x), 64, 16, 200, 4)
        assert handle, h.seekdb_hipvs_hnsw_error()
        def query(q, callback=None):
            ids = np.zeros(10, dtype=np.uint64)
            dist = np.zeros(10, dtype=np.float32)
            count = C.c_int64()
            rc = h.seekdb_hipvs_hnsw_search(handle, q.ctypes.data, 10, 512, callback, None,
                                            ids.ctypes.data, dist.ctypes.data, C.byref(count))
            assert rc == 0 and count.value == 10, h.seekdb_hipvs_hnsw_error()
            assert len(set(map(int, ids))) == 10 and np.all(ids < len(x))
            assert np.all(np.isfinite(dist)) and np.all(dist >= 0)
            expected = np.sum((x[ids] - q) ** 2, axis=1)
            np.testing.assert_allclose(dist, expected, rtol=1e-5, atol=1e-4)
            return ids, dist
        try:
            assert h.seekdb_hipvs_hnsw_verify(handle, x.ctypes.data, len(x), 64) == 0
            changed = x.copy()
            changed[0, 0] += 1
            assert h.seekdb_hipvs_hnsw_verify(handle, changed.ctypes.data, len(x), 64) != 0
            assert h.seekdb_hipvs_hnsw_verify(handle, x.ctypes.data, len(x) + 1, 64) != 0
            # Every input row must be present and correctly mapped in the graph.
            for i, q in enumerate(x):
                ids, dist = query(q)
                assert ids[0] == i and dist[0] < 1e-5
            hits = 0
            queries = rng.standard_normal((100, 64), dtype=np.float32)
            for q in queries:
                ids, _ = query(q)
                gt = np.argsort(np.sum((x - q) ** 2, axis=1))[:10]
                hits += len(set(map(int, ids)) & set(map(int, gt)))
                ids, _ = query(q, even)
                assert np.all(ids % 2 == 0)
            recalls.append(hits / 1000)
            assert recalls[-1] >= .95
            before = query(x[0])
            with tempfile.TemporaryDirectory() as tmp:
                path = os.fsencode(Path(tmp) / 'hnsw.bin')
                assert h.seekdb_hipvs_hnsw_save(handle, path) == 0
                h.seekdb_hipvs_hnsw_free(handle)
                handle = h.seekdb_hipvs_hnsw_load(path, 64)
                assert handle, h.seekdb_hipvs_hnsw_error()
                after = query(x[0])
                np.testing.assert_array_equal(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1])
        finally:
            h.seekdb_hipvs_hnsw_free(handle)
    env = os.environ | {'ROCR_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': ''}
    subprocess.run([sys.executable, __file__, '--bridge', args.bridge, '--hidden-gpu-child'], env=env, check=True)
    print(json.dumps({'repeated_builds': 3, 'all_vector_checks_per_round': 1024,
                      'recall_at_10': recalls, 'persistence': 'PASS', 'payload_completeness': 'PASS', 'filter': 'PASS', 'hidden_gpu': 'PASS'}))


if __name__ == '__main__':
    main()
