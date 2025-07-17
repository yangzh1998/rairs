#ifndef FAISS_RAIR_INDEXREIVFPQFASTSCAN_H
#define FAISS_RAIR_INDEXREIVFPQFASTSCAN_H

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/rair/IndexReIVFFastScan.h>

namespace faiss {

namespace rair {

struct IndexReIVFPQFastScan : faiss::rair::IndexReIVFFastScan {

  ProductQuantizer pq; ///< produces the codes

  IndexReIVFPQFastScan(Index* quantizer, size_t d, size_t nlist, size_t M,
                       size_t nbits, size_t nrep = 2,
                       MetricType metric = METRIC_L2, int bbs = 32);

  void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

  idx_t train_encoder_num_vectors() const override;

  /// same as the regular IVFPQ encoder.
  /// The codes are not reorganized by blocks at that point
  void encode_vectors(idx_t n, const float* x, const idx_t* list_nos,
      uint8_t* codes, bool include_listno = false) const override;

  void compute_LUT(size_t n, const float* x,
      AlignedTable<float>& dis_tables) const override;

  void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

}; // IndexReIVFPQFastScan

} // namespace faiss::rair

} // namespace faiss


#endif // FAISS_RAIR_INDEXREIVFPQFASTSCAN_H