# hipVS HNSW construction on AMD GPUs

`WITH (distance=l2,type=hnsw,lib=cuvs)` selects an optional FP32 build backend.
`lib=vsag` retains the original CPU implementation. Enable the backend with
`OB_BUILD_CUVS=ON` (default OFF).

The bulk construction path buffers the vectors scanned by seekdb, builds the
base graph with hipVS CAGRA/NN-Descent on the AMD GPU, then uses
`cuvsHnswFromCagra` with a full CPU hierarchy. It converts the result into a
CPU-searchable hnswlib HNSW. The GPU path does **not** also call VSAG Add/Build.
The CPU upper hierarchy, all copies, synchronization, temporary-file conversion,
and persistence are real build work and must be included in measurements.

Queries run through seekdb's ordinary SQL vector-index operator. Row offsets are
mapped to seekdb vector IDs and then to SQL primary keys. The backend supports
predicate/deletion bitmap filtering and stores extra information with each ID.
Both the graph and the ID mapping are persisted; loading a persisted index does
not require constructing a replacement graph or an available GPU.

## Build

Install the standard seekdb development dependencies with `./build.sh init`.
On Ubuntu, `rpm2cpio`, `cpio`, and `m4` are also needed for dependency extraction
and parser generation. Install the Rust version pinned in
`rust/rust-toolchain.toml`. Preinstalled hipVS headers/libraries must match ROCm
and the device architecture (the tested image has gfx1100-only code objects).

```bash
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh /work/bridge/libseekdb_cuvs_bridge.so
./build.sh release -DOB_BUILD_CUVS=ON \
  -DCUVS_BRIDGE_LIB=/work/bridge/libseekdb_cuvs_bridge.so --make -j12
```

The bridge C ABI isolates the ROCm compiler from seekdb's compiler.
The linker hides symbols from `libvsag_static.a` with `--exclude-libs`, and the
bridge hides its C++ symbols. This prevents VSAG's modified hnswlib from
interposing into hipVS's differently laid-out hnswlib implementation. The new
`seekdb_hipvs_hnsw_*` symbols manage GPU construction and the native HNSW graph.
The inherited `seekdb_cuvs_*` symbols and explicit `dbms_vector.batch_knn`
operator remain a separate CAGRA query demonstration. Its query timings are
never construction measurements.

## Use and timing boundary

Import raw data into a table with no vector index, then build:

```sql
CREATE VECTOR INDEX idx ON items(embedding)
WITH (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=512);
SELECT id FROM items
ORDER BY l2_distance(embedding, '[...]') APPROXIMATE LIMIT 10;
```

Snapshot finalization and serialization complete pending GPU construction.
If a mutable delta has received new vectors, its next query completes the
pending build before searching. A benchmark must therefore include the first
forced vector-index query after DDL, even if CREATE INDEX returns earlier.
The backend serializes operations per index and uses a 32 MiB build worker stack
because hipVS requires more stack space than seekdb's regular workers provide.
The initial worker uses 16 host threads for upper-level construction.
Set the session variable `ob_hnsw_ef_search` explicitly for both backends; its
default can override the ef_search value written in the index declaration.

`HIPVS_HNSW_BUILD ... fallback=0 ready=1` records a completed hipVS build.
`HIPVS_HNSW_BUILD_FAILED` / `HIPVS_HNSW_ERROR` report failures. Unsupported
builds and absent GPUs return errors; **there is no CPU build fallback**.
Filters use the constructed HNSW and do not trigger a second VSAG build.

The initial implementation supports dense FP32 L2 HNSW, `2 <= m <= 64` and
at least two vectors in a nonempty build. Tiny NN-Descent shapes may be rejected
by hipVS; rejection propagates instead of silently changing algorithms.
Quantized HNSW, HGRAPH, other distance types, multi-GPU operation and importing
old VSAG-formatted `lib=cuvs` snapshots are unsupported. Use a fresh GPU index.
Normal `lib=vsag` persistence is unchanged.
After restarting the observer, run `python unittest/hipvs/check_restart.py
--results /work/results-10000 --port 2881` to compare every saved test query
with its result before restart.

## Validation

```bash
python -m pip install numpy pymysql
python unittest/hipvs/test_bridge.py \
  --bridge /work/bridge/libseekdb_cuvs_bridge.so
python unittest/hipvs/benchmark_build.py --port 2881 --n 10000 --dim 128 \
  --rounds 3 --nq 100 --k 10 --recall-floor .95 \
  --gpu-log /work/observer/log/seekdb.log --output /work/results-10000
```

The bridge test repeats construction three times, checks every vector, verifies
Recall@10 against exact neighbors, filters, save/load equality and explicit
failure with a hidden GPU. The SQL benchmark uses fresh tables each round,
imports identical FP32 values before timing, checks the vector execution plan,
includes first-query readiness, verifies GPU evidence, records all raw timings,
and uses CPU/GPU medians. It records every document's ANN self-query result in the final round and uses
independent queries for recall. ANN self-query misses are recorded as search
quality, rather than treated as proof of missing stored data. Completeness is
checked separately: snapshot processed count must equal N, and the GPU bridge
verifies all N graph labels and their bit-exact FP32 vectors against the input
both after construction and after persistence loading. Set `--target-speedup` only from
the published evaluation configuration; `.95` is an experimental quality floor,
not a claim about the competition threshold. Preserve the JSON results and
query plans. Run profiler tracing separately from timed trials.

Public environment references:

- [seekdb GPU integration prerequisite (PR #1314)](https://github.com/oceanbase/seekdb/pull/1314)
- [AMD hipVS build/install](https://github.com/AMD-Ecosystem/hipVS/blob/release/rocmds-26.03/docs_amd/install/build.rst)
- [AMD hipVS usage](https://github.com/AMD-Ecosystem/hipVS/blob/release/rocmds-26.03/docs_amd/how-to/using-hipVS.rst)
- [ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)

The private Yuque article requires group authentication. The supplied executed
Notebook documents batch query acceleration and is not a build-speedup result.
