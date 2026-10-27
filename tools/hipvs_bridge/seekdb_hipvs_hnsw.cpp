#include "seekdb_hipvs_hnsw.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <hnswlib/hnswlib.h>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
thread_local std::string last_error;
void check(cuvsError_t code)
{
  if (code != CUVS_SUCCESS) {
    const char *s = cuvsGetLastErrorText();
    throw std::runtime_error(s ? s : "hipVS failure");
  }
}
struct TempFile {
  char name[64] = "/tmp/seekdb-hipvs-hnsw-XXXXXX";
  TempFile() { int fd = mkstemp(name); if (fd < 0) throw std::runtime_error("mkstemp"); close(fd); }
  ~TempFile() { unlink(name); }
};
struct NativeHnsw {
  hnswlib::L2Space<float> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;
  explicit NativeHnsw(int dim) : space(dim) {}
};
struct BuildState {
  cuvsResources_t res = 0;
  cuvsCagraIndexParams_t cp = nullptr;
  cuvsCagraIndex_t cagra = nullptr;
  cuvsHnswIndexParams_t hp = nullptr;
  cuvsHnswIndex_t hnsw = nullptr;
  ~BuildState() {
    if (hnsw) cuvsHnswIndexDestroy(hnsw);
    if (hp) cuvsHnswIndexParamsDestroy(hp);
    if (cagra) cuvsCagraIndexDestroy(cagra);
    if (cp) cuvsCagraIndexParamsDestroy(cp);
    if (res) cuvsResourcesDestroy(res);
  }
};
struct Filter : hnswlib::BaseFilterFunctor {
  int (*valid)(uint64_t, void *);
  void *ctx;
  Filter(int (*v)(uint64_t, void *), void *c) : valid(v), ctx(c) {}
  bool operator()(hnswlib::labeltype label) override { return !valid || valid(label, ctx); }
};
}

extern "C" const char *seekdb_hipvs_hnsw_error() { return last_error.c_str(); }
extern "C" void *seekdb_hipvs_hnsw_build(const float *base, int64_t n, int dim,
                                          int m, int ef_construction, int threads)
{
  try {
    if (!base || n < 2 || dim <= 0 || m < 2 || m > 64 || threads < 1) {
      throw std::runtime_error("unsupported HNSW build shape or parameters (n>=2, 2<=m<=64)");
    }
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      throw std::runtime_error("AMD GPU unavailable; CPU fallback disabled");
    }
    const auto start = std::chrono::steady_clock::now();
    BuildState s;
    check(cuvsResourcesCreate(&s.res));
    check(cuvsCagraIndexParamsCreate(&s.cp));
    check(cuvsCagraIndexCreate(&s.cagra));
    s.cp->metric = L2Expanded;
    s.cp->graph_degree = std::min<int64_t>(2 * m, n - 1);
    s.cp->intermediate_graph_degree = std::min<int64_t>(std::max(4 * m, 64), n - 1);
    s.cp->build_algo = NN_DESCENT;
    s.cp->compression = nullptr;
    int64_t shape[2] = {n, dim};
    DLManagedTensor data{};
    data.dl_tensor.data = const_cast<float *>(base);
    data.dl_tensor.device = {kDLCPU, 0};
    data.dl_tensor.ndim = 2;
    data.dl_tensor.dtype = {kDLFloat, 32, 1};
    data.dl_tensor.shape = shape;
    check(cuvsCagraBuild(s.res, s.cp, &data, s.cagra));
    if (hipDeviceSynchronize() != hipSuccess) throw std::runtime_error("CAGRA GPU synchronization failed");
    check(cuvsHnswIndexParamsCreate(&s.hp));
    check(cuvsHnswIndexCreate(&s.hnsw));
    s.hp->hierarchy = CPU; // GPU constructs the costly base graph; CPU adds upper levels.
    s.hp->ef_construction = ef_construction;
    s.hp->num_threads = threads;
    check(cuvsHnswFromCagra(s.res, s.hp, s.cagra, s.hnsw));
    TempFile file;
    check(cuvsHnswSerialize(s.res, file.name, s.hnsw));
    auto out = std::make_unique<NativeHnsw>(dim);
    out->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(&out->space, file.name);
    if (out->graph->getCurrentElementCount() != static_cast<uint64_t>(n)) {
      throw std::runtime_error("HNSW conversion lost vectors");
    }
    if (seekdb_hipvs_hnsw_verify(out.get(), base, n, dim)) {
      throw std::runtime_error("HNSW label/vector completeness validation failed");
    }
    if (hipDeviceSynchronize() != hipSuccess) throw std::runtime_error("HNSW final synchronization failed");
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    fprintf(stderr, "HIPVS_HNSW_BUILD backend=hipvs gpu_graph=nn_descent hierarchy=cpu n=%ld dim=%d m=%d ef_construction=%d threads=%d fp32=1 fallback=0 ready=1 validated_rows=%ld ms=%.3f\n",
            n, dim, m, ef_construction, threads, n, ms);
    return out.release();
  } catch (const std::exception &e) {
    last_error = e.what();
    fprintf(stderr, "HIPVS_HNSW_BUILD_FAILED fallback=0 error=%s\n", e.what());
    return nullptr;
  }
}
extern "C" void seekdb_hipvs_hnsw_free(void *handle) { delete static_cast<NativeHnsw *>(handle); }
extern "C" int seekdb_hipvs_hnsw_search(void *handle, const float *query, int64_t k, int ef,
                                         int (*valid)(uint64_t, void *), void *ctx,
                                         uint64_t *offsets, float *distances, int64_t *count)
{
  try {
    auto *h = static_cast<NativeHnsw *>(handle);
    if (!h || !query || k <= 0 || !offsets || !distances || !count) throw std::runtime_error("invalid HNSW query");
    h->graph->setEf(std::max<int64_t>(ef, k));
    Filter filter(valid, ctx);
    auto results = h->graph->searchKnn(query, k, valid ? &filter : nullptr);
    *count = results.size();
    for (int64_t i = *count; i > 0; --i) {
      offsets[i - 1] = results.top().second;
      distances[i - 1] = results.top().first;
      results.pop();
    }
    return 0;
  } catch (const std::exception &e) { last_error = e.what(); return -1; }
}
extern "C" int seekdb_hipvs_hnsw_verify(void *handle, const float *base, int64_t n, int dim)
{
  try {
    auto *h = static_cast<NativeHnsw *>(handle);
    if (!h || !base || dim <= 0 || n < 0 ||
        h->graph->getCurrentElementCount() != static_cast<uint64_t>(n) ||
        h->graph->label_lookup_.size() != static_cast<uint64_t>(n) ||
        h->graph->label_offset_ - h->graph->offsetData_ != static_cast<uint64_t>(dim) * sizeof(float)) {
      throw std::runtime_error("HNSW shape/count mismatch");
    }
    for (int64_t i = 0; i < n; ++i) {
      auto it = h->graph->label_lookup_.find(i);
      if (it == h->graph->label_lookup_.end() ||
          memcmp(h->graph->getDataByInternalId(it->second), base + i * dim, dim * sizeof(float))) {
        throw std::runtime_error("HNSW missing/mis-mapped vector or changed FP32 precision");
      }
    }
    return 0;
  } catch (const std::exception &e) { last_error = e.what(); return -1; }
}
extern "C" int seekdb_hipvs_hnsw_save(void *handle, const char *filename)
{
  try { static_cast<NativeHnsw *>(handle)->graph->saveIndex(filename); return 0; }
  catch (const std::exception &e) { last_error = e.what(); return -1; }
}
extern "C" void *seekdb_hipvs_hnsw_load(const char *filename, int dim)
{
  try {
    auto h = std::make_unique<NativeHnsw>(dim);
    h->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(&h->space, filename);
    return h.release();
  } catch (const std::exception &e) { last_error = e.what(); return nullptr; }
}
