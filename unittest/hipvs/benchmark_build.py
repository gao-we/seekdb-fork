#!/usr/bin/env python3
"""End-to-end SQL build benchmark. Never substitutes batch query timings.

Requires numpy and pymysql. Each run uses a fresh table, imports raw vectors
before timing, starts before CREATE VECTOR INDEX, includes a forced index read,
then validates recall against exact FP32 vector ground truth. SQL failures abort.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import time

import numpy as np
import pymysql


def vector_literal(v):
    # Nine significant digits round-trip IEEE float32 values through SQL.
    return '[' + ','.join(format(float(x), '.9g') for x in v) + ']'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--port', type=int, default=2881)
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--nq', type=int, default=100)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--m', type=int, default=16)
    p.add_argument('--ef-construction', type=int, default=200)
    p.add_argument('--ef-search', type=int, default=512)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--recall-floor', type=float, default=.95)
    p.add_argument('--target-speedup', type=float)
    p.add_argument('--output', required=True)
    p.add_argument('--backend', choices=['both', 'vsag', 'cuvs'], default='both')
    p.add_argument('--gpu-log', type=Path)
    args = p.parse_args()
    if args.rounds < 3:
        p.error('at least three rounds are required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    base = rng.standard_normal((args.n, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.nq, args.dim), dtype=np.float32)
    # Deliberately non-contiguous, large signed SQL primary keys.
    ids = (rng.permutation(args.n).astype(np.int64) * 17 + 10**12)
    ids[::2] *= -1
    base_lit = [vector_literal(v) for v in base]
    query_lit = [vector_literal(v) for v in queries]
    exact = np.empty((args.nq, args.k), dtype=np.int64)
    for i, q in enumerate(queries):
        d = np.sum((base - q) ** 2, axis=1, dtype=np.float64)
        exact[i] = ids[np.argsort(d, kind='stable')[:args.k]]
    np.savez(out / 'dataset.npz', base=base, queries=queries, ids=ids, truth=exact)
    report = {'config': vars(args).copy(), 'data_sha256': hashlib.sha256(base.tobytes() + queries.tobytes() + ids.tobytes()).hexdigest(),
              'preparation_seconds': time.perf_counter() - start, 'runs': [],
              'official_acceptance': 'UNDETERMINED: official evaluation configuration not supplied'}
    report['config']['gpu_log'] = str(args.gpu_log) if args.gpu_log else None
    db = pymysql.connect(host='127.0.0.1', port=args.port, user='root', autocommit=True,
                         read_timeout=1800, write_timeout=1800, charset='utf8mb4')
    cur = db.cursor()
    cur.execute('select version()')
    report['seekdb_version'] = cur.fetchone()[0]
    cur.execute('alter system set ob_vector_memory_limit_percentage=30')
    cur.execute('set session ob_hnsw_ef_search=%s', (args.ef_search,))
    cur.execute("show variables like 'ob_hnsw_ef_search'")
    report['effective_session_ef_search'] = int(cur.fetchone()[1])
    # Isolated database; previous experiment data is never reused.
    dbname = 'hipvs_build_' + str(time.time_ns())
    report['database'] = dbname
    cur.execute(f'create database {dbname}')
    cur.execute(f'use {dbname}')
    valid_ids = set(map(int, ids))
    backends = ['vsag', 'cuvs'] if args.backend == 'both' else [args.backend]

    def persist():
        (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    def query(table, v, k, where=''):
        cur.execute(f"select id, l2_distance(v, '{v}') d from {table} {where} "
                    f"order by l2_distance(v, '{v}') approximate limit {k}")
        return cur.fetchall()

    persist()
    for round_no in range(args.rounds):
        for backend in backends if round_no % 2 == 0 else backends[::-1]:
            table = f't_{backend}_{round_no}'
            t = time.perf_counter()
            cur.execute(f'create table {table}(id bigint primary key, v vector({args.dim}))')
            for begin in range(0, args.n, 256):
                cur.executemany(f'insert into {table} values (%s,%s)',
                                [(int(ids[i]), base_lit[i]) for i in range(begin, min(begin + 256, args.n))])
            cur.execute(f'select count(*),count(distinct id) from {table}')
            assert cur.fetchone() == (args.n, args.n)
            imported = time.perf_counter() - t
            ddl = (f'create vector index idx on {table}(v) with '
                   f'(distance=l2,type=hnsw,lib={backend},m={args.m},'
                   f'ef_construction={args.ef_construction},ef_search={args.ef_search})')
            log_stat = args.gpu_log.stat() if args.gpu_log and args.gpu_log.exists() else None
            log_offset = log_stat.st_size if log_stat else 0
            started = time.perf_counter()
            cur.execute(ddl)
            ddl_s = time.perf_counter() - started
            # Force the actual vector index. This read includes any deferred build.
            cur.execute(f"explain select id from {table} order by l2_distance(v, '{query_lit[0]}') approximate limit {args.k}")
            plan = '\n'.join(str(r) for r in cur.fetchall())
            (out / f'{table}-plan.txt').write_text(plan + '\n')
            if 'VECTOR' not in plan.upper():
                raise RuntimeError('Query plan did not confirm vector index use: ' + plan)
            ready = query(table, query_lit[0], args.k)
            if len(ready) != args.k:
                raise RuntimeError('Index not ready: insufficient neighbors')
            elapsed = time.perf_counter() - started
            evidence = ''
            if args.gpu_log and args.gpu_log.exists():
                # seekdb can rotate its log while a long build is running.
                original = next((path for path in args.gpu_log.parent.glob(args.gpu_log.name + '*')
                                 if log_stat and path.stat().st_ino == log_stat.st_ino), args.gpu_log)
                with original.open('rb') as f:
                    f.seek(log_offset)
                    evidence = f.read().decode('utf-8', errors='replace')
                if original != args.gpu_log:
                    evidence += args.gpu_log.read_bytes().decode('utf-8', errors='replace')
                (out / f'{table}-build.log').write_text(evidence)
            if backend == 'cuvs' and ('HIPVS_HNSW_BUILD ' not in evidence or 'fallback=0 ready=1' not in evidence):
                raise RuntimeError('GPU build evidence missing; refusing to count this as GPU acceleration')
            hits = 0
            all_results = []
            for i, v in enumerate(query_lit):
                rows = query(table, v, args.k)
                got = [int(r[0]) for r in rows]
                if len(got) != args.k or len(set(got)) != len(got) or not set(got) <= valid_ids:
                    raise RuntimeError('Incomplete, duplicate or illegal document IDs')
                if not all(np.isfinite(float(r[1])) and float(r[1]) >= 0 for r in rows):
                    raise RuntimeError('Invalid distances')
                hits += len(set(got) & set(map(int, exact[i])))
                all_results.append([list(r) for r in rows])
            recall = hits / (args.nq * args.k)
            (out / f'{table}-queries.json').write_text(json.dumps(all_results) + '\n')
            # Probe every imported row through ANN once per backend, final round.
            # This catches missing rows and row-offset / document-ID mapping bugs.
            completeness = None
            if round_no == args.rounds - 1:
                completeness = 0
                for i, v in enumerate(base_lit):
                    rows = query(table, v, 1)
                    if not rows or int(rows[0][0]) not in valid_ids:
                        raise RuntimeError(f'Invalid self-query document ID at row {i}: {rows}')
                    completeness += int(int(rows[0][0]) == int(ids[i]) and float(rows[0][1]) < 1e-4)
                # Check restrictive SQL filtering against exact filtered SQL results.
                for v in query_lit[:10]:
                    approx = query(table, v, args.k, 'where id > 0')
                    if any(int(r[0]) <= 0 for r in approx):
                        raise RuntimeError('Filtered query returned an excluded ID')
            cur.execute("select table_id from oceanbase.__all_table where table_name=%s and database_id="
                        "(select database_id from oceanbase.__all_database where database_name=%s)", (table, dbname))
            data_table_id = cur.fetchone()[0]
            deadline = time.monotonic() + 30
            processed = None
            while time.monotonic() < deadline:
                cur.execute('select sync_info from oceanbase.__all_virtual_vector_index_info where data_table_id=%s', (data_table_id,))
                statuses = [r[0] for r in cur.fetchall()]
                counts = [int(m.group(1)) for status in statuses
                          if (m := re.search(r'snap_cnt=(\d+);', status))]
                if args.n in counts:
                    processed = args.n
                    break
                time.sleep(.1)  # administrative counters refresh asynchronously
            if processed != args.n:
                raise RuntimeError('Snapshot did not process all imported vectors: ' + str(statuses))
            if backend == 'cuvs' and f'validated_rows={args.n} ' not in evidence:
                raise RuntimeError('Missing full GPU graph / FP32 payload completeness verification')
            entry = {'round': round_no + 1, 'backend': backend, 'table': table,
                     'import_seconds': imported, 'ddl_seconds': ddl_s, 'snapshot_processed_rows': processed,
                     'end_to_end_build_seconds': elapsed, 'recall_at_k': recall,
                     'quality_pass': recall >= args.recall_floor,
                     'self_query_count': args.n if completeness is not None else None, 'self_query_hits': completeness, 'ddl': ddl}
            report['runs'].append(entry)
            persist()
            print(json.dumps(entry), flush=True)
            if recall < args.recall_floor:
                raise RuntimeError(f'Recall {recall} below experimental floor {args.recall_floor}')
    if args.backend == 'both':
        medians = {b: statistics.median(r['end_to_end_build_seconds'] for r in report['runs'] if r['backend'] == b) for b in backends}
        speedup = medians['vsag'] / medians['cuvs']
        report['medians_seconds'] = medians
        report['build_speedup'] = speedup
        report['experimental_quality_pass'] = all(r['quality_pass'] for r in report['runs'])
        if args.target_speedup is not None:
            report['configured_target_pass'] = speedup >= args.target_speedup and report['experimental_quality_pass']
    persist()
    print(json.dumps({k: v for k, v in report.items() if k not in ('runs',)}, indent=2), flush=True)
    db.close()


if __name__ == '__main__':
    main()
