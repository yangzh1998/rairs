#ifndef FAISS_RAIR_INDEXREIVFFLAT_H
#define FAISS_RAIR_INDEXREIVFFLAT_H

#include <faiss/rair/IndexReIVF.h>

namespace faiss {

namespace rair {

struct IndexReIVFFlat : faiss::rair::IndexReIVF {
  IndexReIVFFlat(Index* quantizer, size_t d, size_t nlist_, size_t nrep = 2,
                 MetricType = METRIC_L2);

  void encode_vectors(idx_t n, const float* x, const idx_t* list_nos,
                      uint8_t* codes, bool include_listnos = false) const override;

  InvertedListScanner* get_InvertedListScanner(
      bool store_pairs, const IDSelector* sel = nullptr) const override;

  void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

  /// TODO: reconstruct_from_offset

}; // IndexReIVFFlat

} // namespace faiss::rair

} // namespace faiss

#endif // FAISS_RAIR_INDEXREIVFFLAT_H
