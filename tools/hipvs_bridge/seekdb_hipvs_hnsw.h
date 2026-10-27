#pragma once
#include <stdint.h>
#if defined(__GNUC__)
#define SEEKDB_HIPVS_API __attribute__((visibility("default")))
#else
#define SEEKDB_HIPVS_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* No CPU build fallback. Returned graph is a full, CPU-searchable HNSW.
 * All pointers remain caller-owned except the opaque handle. */
SEEKDB_HIPVS_API void *seekdb_hipvs_hnsw_build(const float *base, int64_t n, int dim,
                             int m, int ef_construction, int threads);
SEEKDB_HIPVS_API void seekdb_hipvs_hnsw_free(void *handle);
SEEKDB_HIPVS_API int seekdb_hipvs_hnsw_search(void *handle, const float *query, int64_t k, int ef,
                            int (*valid)(uint64_t, void *), void *ctx,
                            uint64_t *offsets, float *distances, int64_t *count);
SEEKDB_HIPVS_API int seekdb_hipvs_hnsw_verify(void *handle, const float *base, int64_t n, int dim);
SEEKDB_HIPVS_API int seekdb_hipvs_hnsw_save(void *handle, const char *filename);
SEEKDB_HIPVS_API void *seekdb_hipvs_hnsw_load(const char *filename, int dim);
SEEKDB_HIPVS_API const char *seekdb_hipvs_hnsw_error(void);
#ifdef __cplusplus
}
#endif
