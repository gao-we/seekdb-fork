#pragma once

// Included by the VSAG adaptor after seekdb/vsag declarations. The bridge keeps
// ROCm, hipVS and hnswlib types outside seekdb's ABI and compiler toolchain.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

extern "C" {
void *seekdb_hipvs_hnsw_build(const float *, int64_t, int, int, int, int);
void seekdb_hipvs_hnsw_free(void *);
int seekdb_hipvs_hnsw_search(void *, const float *, int64_t, int,
                            int (*)(uint64_t, void *), void *, uint64_t *, float *, int64_t *);
int seekdb_hipvs_hnsw_save(void *, const char *);
int seekdb_hipvs_hnsw_verify(void *, const float *, int64_t, int);
void *seekdb_hipvs_hnsw_load(const char *, int);
const char *seekdb_hipvs_hnsw_error();
}

namespace oceanbase {
namespace common {
namespace obvsag {

class HipvsHnswBackend {
public:
  HipvsHnswBackend(int dim, int m, int ef, int ef_search, uint64_t extra_size)
    : dim_(dim), m_(m), ef_(ef), ef_search_(ef_search), extra_size_(extra_size) {}
  ~HipvsHnswBackend() { seekdb_hipvs_hnsw_free(graph_); }
  HipvsHnswBackend(const HipvsHnswBackend &) = delete;
  HipvsHnswBackend &operator=(const HipvsHnswBackend &) = delete;

  int add(const vsag::DatasetPtr &data)
  {
    return protect([&] {
      std::lock_guard<std::mutex> guard(mu_);
      const int64_t n = data->GetNumElements();
      const float *v = data->GetFloat32Vectors();
      const int64_t *ids = data->GetIds();
      if (n < 0 || data->GetDim() != dim_ || !v || !ids ||
          (extra_size_ && !data->GetExtraInfos())) throw std::runtime_error("invalid HIPVS input");
      std::unordered_set<int64_t> batch_ids;
      for (int64_t i = 0; i < n; ++i) {
        if (positions_.count(ids[i]) || !batch_ids.insert(ids[i]).second) throw std::runtime_error("duplicate HIPVS vector ID");
        for (int d = 0; d < dim_; ++d) {
          if (!std::isfinite(v[i * dim_ + d])) throw std::runtime_error("non-finite HIPVS vector");
        }
      }
      // Reserve all growing containers before publishing any IDs.
      const uint64_t total = ids_.size() + n;
      if (ids_.capacity() < total) {
        const uint64_t capacity = std::max<uint64_t>(total, std::max<uint64_t>(256, ids_.capacity() * 2));
        vectors_.reserve(capacity * dim_);
        ids_.reserve(capacity);
        extras_.reserve(capacity * extra_size_);
        positions_.reserve(capacity);
      }
      for (int64_t i = 0; i < n; ++i) positions_.emplace(ids[i], ids_.size() + i);
      vectors_.insert(vectors_.end(), v, v + n * dim_);
      ids_.insert(ids_.end(), ids, ids + n);
      if (extra_size_) {
        const char *extras = data->GetExtraInfos();
        extras_.insert(extras_.end(), extras, extras + n * extra_size_);
      }
      // No CPU HNSW Add/Build is called on this path.
    });
  }

  int count() const { std::lock_guard<std::mutex> guard(mu_); return ids_.size(); }
  uint64_t estimate(uint64_t n) const
  { return n * (2 * dim_ * sizeof(float) + 4 * m_ * sizeof(uint32_t) + extra_size_ + 96); }
  int bounds(int64_t &min_id, int64_t &max_id)
  {
    std::lock_guard<std::mutex> guard(mu_);
    if (!ids_.empty()) {
      auto mm = std::minmax_element(ids_.begin(), ids_.end());
      min_id = *mm.first;
      max_id = *mm.second;
    }
    return OB_SUCCESS;
  }
  int finish()
  {
    return protect([&] { std::lock_guard<std::mutex> guard(mu_); ensure_built(); });
  }

  int query(const vsag::DatasetPtr &query, int64_t k, const float *&dist,
            const int64_t *&ids, int64_t &result_size, const std::string &parameters, FilterInterface *filter,
            bool reverse, bool need_extras, const char *&extras, vsag::Allocator *allocator)
  {
    return protect([&] {
      std::lock_guard<std::mutex> guard(mu_);
      result_size = 0;
      ids = nullptr;
      dist = nullptr;
      extras = nullptr;
      if (k <= 0 || query->GetDim() != dim_ || !query->GetFloat32Vectors()) {
        throw std::runtime_error("invalid HIPVS query");
      }
      if (ids_.empty()) return;
      ensure_built(); // Readiness benchmark must include this if not finalized by DDL.
      k = std::min<int64_t>(k, ids_.size());
      std::vector<uint64_t> offsets(k);
      std::vector<float> distances(k);
      struct Context { const std::vector<int64_t> *ids; FilterInterface *filter; bool reverse; };
      Context ctx{&ids_, filter, reverse};
      auto valid = [](uint64_t pos, void *arg) -> int {
        auto *c = static_cast<Context *>(arg);
        if (pos >= c->ids->size()) return 0;
        if (!c->filter) return 1;
        bool tested = c->filter->test((*c->ids)[pos]);
        return c->reverse ? tested : !tested;
      };
      int ef = ef_search_;
      auto key = parameters.find("\"ef_search\"");
      if (key != std::string::npos) {
        auto colon = parameters.find(':', key);
        if (colon != std::string::npos) ef = std::max(1, atoi(parameters.c_str() + colon + 1));
      }
      int64_t found = 0;
      if (seekdb_hipvs_hnsw_search(graph_, query->GetFloat32Vectors(), k, ef,
                                  valid, &ctx, offsets.data(), distances.data(), &found)) {
        throw std::runtime_error(seekdb_hipvs_hnsw_error());
      }
      auto alloc = [&](uint64_t bytes) { return allocator ? allocator->Allocate(bytes) : ::malloc(bytes); };
      auto release = [&](void *p) { if (allocator) allocator->Deallocate(p); else ::free(p); };
      int64_t *out_ids = static_cast<int64_t *>(alloc(k * sizeof(int64_t)));
      float *out_dist = static_cast<float *>(alloc(k * sizeof(float)));
      char *out_extra = need_extras && extra_size_ ? static_cast<char *>(alloc(k * extra_size_)) : nullptr;
      if (!out_ids || !out_dist || (need_extras && extra_size_ && !out_extra)) {
        if (out_ids) release(out_ids);
        if (out_dist) release(out_dist);
        if (out_extra) release(out_extra);
        throw std::bad_alloc();
      }
      for (int64_t i = 0; i < found; ++i) {
        if (offsets[i] >= ids_.size() || !std::isfinite(distances[i])) {
          release(out_ids); release(out_dist); if (out_extra) release(out_extra);
          throw std::runtime_error("illegal HIPVS HNSW result");
        }
        out_ids[i] = ids_[offsets[i]];
        out_dist[i] = distances[i];
        if (out_extra) memcpy(out_extra + i * extra_size_, extras_.data() + offsets[i] * extra_size_, extra_size_);
      }
      ids = out_ids; dist = out_dist; result_size = found; extras = out_extra;
    });
  }

  int distance(const float *query, const int64_t *ids, int64_t n,
               const float *&dist, vsag::Allocator *allocator)
  {
    return protect([&] {
      std::lock_guard<std::mutex> guard(mu_);
      if (!query || !ids || n < 0) throw std::runtime_error("invalid HIPVS distance request");
      std::vector<float> result(n);
      for (int64_t i = 0; i < n; ++i) {
        auto it = positions_.find(ids[i]);
        if (it == positions_.end()) throw std::runtime_error("unknown HIPVS vector ID");
        float d = 0;
        for (int j = 0; j < dim_; ++j) {
          float x = query[j] - vectors_[it->second * dim_ + j]; d += x * x;
        }
        result[i] = d;
      }
      float *out = static_cast<float *>(allocator ? allocator->Allocate(n * sizeof(float)) : ::malloc(n * sizeof(float)));
      if (!out && n) throw std::bad_alloc();
      memcpy(out, result.data(), n * sizeof(float)); dist = out;
    });
  }
  int extra(const int64_t *ids, int64_t n, char *out)
  {
    return protect([&] {
      std::lock_guard<std::mutex> guard(mu_);
      if (extra_size_ && !out) throw std::runtime_error("missing HIPVS extra output");
      for (int64_t i = 0; i < n; ++i) {
        auto it = positions_.find(ids[i]);
        if (it == positions_.end()) throw std::runtime_error("unknown HIPVS vector ID");
        if (extra_size_) memcpy(out + i * extra_size_, extras_.data() + it->second * extra_size_, extra_size_);
      }
    });
  }

  int serialize(std::ostream &out)
  {
    return protect([&] {
      std::lock_guard<std::mutex> guard(mu_);
      ensure_built();
      TempFile file;
      uint64_t graph_bytes = 0;
      if (graph_) {
        if (seekdb_hipvs_hnsw_save(graph_, file.name)) throw std::runtime_error(seekdb_hipvs_hnsw_error());
        std::ifstream in(file.name, std::ios::binary | std::ios::ate);
        graph_bytes = in.tellg();
      }
      const uint64_t header[] = {0x3157534e48565048ULL, 1, static_cast<uint64_t>(dim_),
          static_cast<uint64_t>(m_), static_cast<uint64_t>(ef_), static_cast<uint64_t>(ef_search_),
          extra_size_, ids_.size(), graph_bytes};
      out.write(reinterpret_cast<const char *>(header), sizeof(header));
      out.write(reinterpret_cast<const char *>(ids_.data()), ids_.size() * sizeof(int64_t));
      out.write(reinterpret_cast<const char *>(vectors_.data()), vectors_.size() * sizeof(float));
      out.write(extras_.data(), extras_.size());
      if (graph_bytes) { std::ifstream in(file.name, std::ios::binary); out << in.rdbuf(); }
      if (!out) throw std::runtime_error("HIPVS index serialization failed");
    });
  }
  int deserialize(std::istream &in)
  {
    return protect([&] {
      std::lock_guard<std::mutex> guard(mu_);
      uint64_t h[9]{};
      in.read(reinterpret_cast<char *>(h), sizeof(h));
      if (!in || h[0] != 0x3157534e48565048ULL || h[1] != 1 || h[2] != static_cast<uint64_t>(dim_) ||
          h[3] != static_cast<uint64_t>(m_) || h[6] != extra_size_ || h[7] > INT32_MAX ||
          h[8] > (1ULL << 40) || (h[7] > 0 && h[8] == 0)) {
        throw std::runtime_error("invalid HIPVS HNSW persistence header");
      }
      std::vector<int64_t> ids(h[7]);
      std::vector<float> vectors(h[7] * dim_);
      std::vector<char> extras(h[7] * extra_size_);
      in.read(reinterpret_cast<char *>(ids.data()), ids.size() * sizeof(int64_t));
      in.read(reinterpret_cast<char *>(vectors.data()), vectors.size() * sizeof(float));
      in.read(extras.data(), extras.size());
      TempFile file;
      std::ofstream out(file.name, std::ios::binary | std::ios::trunc);
      // SQL worker stacks are small. Keep streaming buffers on the heap.
      std::vector<char> buf(65536);
      for (uint64_t left = h[8]; left;) {
        const uint64_t n = std::min<uint64_t>(left, buf.size());
        in.read(buf.data(), n); out.write(buf.data(), n); left -= n;
        if (!in || !out) throw std::runtime_error("truncated HIPVS HNSW index");
      }
      out.close();
      if (!in) throw std::runtime_error("truncated HIPVS vector payload");
      std::unordered_map<int64_t, uint64_t> positions;
      for (uint64_t i = 0; i < ids.size(); ++i) {
        if (!positions.emplace(ids[i], i).second) throw std::runtime_error("duplicate persisted HIPVS ID");
      }
      void *loaded = h[8] ? seekdb_hipvs_hnsw_load(file.name, dim_) : nullptr;
      if (h[8] && !loaded) throw std::runtime_error(seekdb_hipvs_hnsw_error());
      if (loaded && seekdb_hipvs_hnsw_verify(loaded, vectors.data(), ids.size(), dim_)) {
        seekdb_hipvs_hnsw_free(loaded);
        throw std::runtime_error("persisted HNSW graph does not match all stored FP32 vectors");
      }
      seekdb_hipvs_hnsw_free(graph_); graph_ = loaded;
      ids_.swap(ids); vectors_.swap(vectors); extras_.swap(extras); positions_.swap(positions);
      built_n_ = ids_.size();
    });
  }

private:
  struct TempFile {
    char name[64] = "/tmp/seekdb-hnsw-persist-XXXXXX";
    TempFile() { int fd = mkstemp(name); if (fd < 0) throw std::runtime_error("mkstemp"); close(fd); }
    ~TempFile() { unlink(name); }
  };
  template <class F> int protect(F &&fn)
  {
    try { fn(); return OB_SUCCESS; }
    catch (const std::exception &e) {
      int ret = OB_ERR_VSAG_RETURN_ERROR;
      LOG_WARN("HIPVS_HNSW_ERROR CPU fallback disabled", "what", e.what());
      return OB_ERR_VSAG_RETURN_ERROR;
    } catch (...) { return OB_ERR_VSAG_RETURN_ERROR; }
  }
  void ensure_built()
  {
    if (ids_.empty() || (graph_ && built_n_ == ids_.size())) return;
    struct Job { HipvsHnswBackend *self; void *result; } job{this, nullptr};
    auto run = [](void *arg) -> void * {
      auto *j = static_cast<Job *>(arg);
      auto *s = j->self;
      j->result = seekdb_hipvs_hnsw_build(s->vectors_.data(), s->ids_.size(), s->dim_, s->m_, s->ef_, 16);
      return nullptr;
    };
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    int rc = pthread_attr_setstacksize(&attr, 32UL * 1024 * 1024);
    pthread_t tid;
    if (!rc) rc = pthread_create(&tid, &attr, run, &job);
    pthread_attr_destroy(&attr);
    if (rc) throw std::runtime_error("cannot start HIPVS build worker");
    if (pthread_join(tid, nullptr)) throw std::runtime_error("cannot join HIPVS build worker");
    if (!job.result) throw std::runtime_error("HIPVS GPU build failed; see bridge log; CPU fallback disabled");
    seekdb_hipvs_hnsw_free(graph_); graph_ = job.result; built_n_ = ids_.size();
    LOG_INFO("HIPVS_HNSW_READY", "rows", built_n_, "dim", dim_, "fallback", 0);
  }
  int dim_, m_, ef_, ef_search_;
  uint64_t extra_size_;
  mutable std::mutex mu_;
  std::vector<float> vectors_;
  std::vector<int64_t> ids_;
  std::vector<char> extras_;
  std::unordered_map<int64_t, uint64_t> positions_;
  void *graph_ = nullptr;
  uint64_t built_n_ = 0;
};

} // namespace obvsag
} // namespace common
} // namespace oceanbase
