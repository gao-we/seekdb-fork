#!/usr/bin/env python3
"""Exercise bitmap deletes and a batched mutable delta through normal SQL."""
import argparse
import json
import time
import numpy as np
import pymysql
from benchmark_build import vector_literal

p = argparse.ArgumentParser(__doc__)
p.add_argument('--port', type=int, default=2882)
a = p.parse_args()
db = pymysql.connect(host='127.0.0.1', port=a.port, user='root', autocommit=True, read_timeout=120)
c = db.cursor()
name = 'hipvs_mutations_' + str(time.time_ns())
c.execute(f'create database {name}')
c.execute(f'use {name}')
c.execute('set session ob_hnsw_ef_search=512')
c.execute('create table docs(id bigint primary key,v vector(64))')
rng = np.random.default_rng(20261027)
x = rng.standard_normal((2048, 64), dtype=np.float32)
c.executemany('insert into docs values (%s,%s)', [(10**12 + i * 17, vector_literal(v)) for i,v in enumerate(x)])
c.execute('create vector index idx on docs(v) with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=512)')

def query(q):
    v = vector_literal(q)
    c.execute(f"select id from docs order by l2_distance(v,'{v}') approximate limit 10")
    return [int(r[0]) for r in c.fetchall()]

assert query(x[0])[0] == 10**12
removed = {10**12 + i * 17 for i in range(32)}
c.execute(f'delete from docs where id < {10**12 + 32 * 17}')
for q in x[:64]:
    assert not set(query(q)) & removed
# A supported GPU batch size; tiny NN-Descent shapes are explicitly unsupported.
y = rng.standard_normal((512, 64), dtype=np.float32) * np.float32(.01)
c.executemany('insert into docs values (%s,%s)', [(2*10**12 + i * 19, vector_literal(v)) for i,v in enumerate(y)])
for i,q in enumerate(y[:64]):
    result = query(q)
    assert result[0] == 2*10**12 + i*19, (i, result)
    assert not set(result) & removed
print(json.dumps({'database': name, 'deleted_ids': 32, 'delete_queries': 64,
                  'inserted_batch': 512, 'freshness_queries': 64, 'status': 'PASS'}))
