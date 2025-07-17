#include <faiss/rair/IndexReIVFFastScan.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/rair/IndexReIVFPQFastScan.h>
#include <faiss/impl/FaissAssert.h>

using namespace faiss;
using namespace faiss::rair;

IndexReIVFPQFastScan::IndexReIVFPQFastScan(
    Index* quantizer, size_t d, size_t nlist, size_t M, size_t nbits,
    size_t nrep, MetricType metric, int bbs):
    IndexReIVFFastScan(quantizer, d, nlist, 0, nrep, metric), pq(d, M, nbits) {
  by_residual = false;
  init_fastscan(M, nbits, nlist, metric, bbs);
}

void IndexReIVFPQFastScan::train_encoder(idx_t n, const float* x,
                                         const idx_t* assign) {
  pq.verbose = verbose;
  pq.train(n, x);
}

idx_t IndexReIVFPQFastScan::train_encoder_num_vectors() const {
  return pq.cp.max_points_per_centroid * pq.ksub;
}

void IndexReIVFPQFastScan::encode_vectors(
    idx_t n, const float* x, const idx_t* list_nos, uint8_t* codes,
    bool include_listnos) const {
  pq.compute_codes(x, codes, n);
  FAISS_ASSERT_MSG(!include_listnos, "Not implemented");
}

void
IndexReIVFPQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
  pq.decode(bytes, x, n);
}

void IndexReIVFPQFastScan::compute_LUT(
    size_t n, const float* x, AlignedTable<float>& dis_tables) const {
  dis_tables.resize(n * pq.ksub * pq.M);
  if (metric_type == METRIC_L2) {
    pq.compute_distance_tables(n, x, dis_tables.get());
  } else if (metric_type == METRIC_INNER_PRODUCT) {
    pq.compute_inner_prod_tables(n, x, dis_tables.get());
  } else {
    FAISS_THROW_FMT("metric %d not supported", metric_type);
  }
}