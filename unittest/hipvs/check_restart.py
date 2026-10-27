#!/usr/bin/env python3
"""Compare persisted SQL ANN results with a completed benchmark after restart."""
import argparse
import json
from pathlib import Path
import numpy as np
import pymysql
from benchmark_build import vector_literal

p = argparse.ArgumentParser(__doc__)
p.add_argument('--results', type=Path, required=True)
p.add_argument('--port', type=int, default=2882)
a = p.parse_args()
r = json.loads((a.results / 'results.json').read_text())
data = np.load(a.results / 'dataset.npz')
db = pymysql.connect(host='127.0.0.1', port=a.port, user='root', database=r['database'], autocommit=True)
c = db.cursor()
c.execute('set session ob_hnsw_ef_search=%s', (r['config']['ef_search'],))
checks = []
for run in r['runs']:
    # Every round has a separate persisted graph and ID map.
    expected = json.loads((a.results / (run['table'] + '-queries.json')).read_text())
    for i, q in enumerate(data['queries']):
        v = vector_literal(q)
        c.execute(f"select id,l2_distance(v,'{v}') from {run['table']} order by l2_distance(v,'{v}') approximate limit {r['config']['k']}")
        got = c.fetchall()
        assert [int(x[0]) for x in got] == [int(x[0]) for x in expected[i]], (run['table'], i)
        np.testing.assert_allclose([float(x[1]) for x in got], [float(x[1]) for x in expected[i]], rtol=1e-6, atol=1e-6)
    checks.append({'table': run['table'], 'queries': len(data['queries']), 'status': 'PASS'})
print(json.dumps({'restart_checks': checks}, indent=2))
