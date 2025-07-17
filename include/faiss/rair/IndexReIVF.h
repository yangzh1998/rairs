#ifndef FAISS_RAIR_INDEXREIVF_H
#define FAISS_RAIR_INDEXREIVF_H

#include <atomic>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>

namespace faiss {

namespace rair {

/// abstract class for all IndexReIVFxxx
struct IndexReIVF : faiss::IndexIVF {

  size_t nrep;

  /**
   * Redundant strategy determines how to choose the second cluster
   *
   *  0: Naive: loss(r') = r'^2; the second nearest cluster
   *  1(default): RAIR: loss(r') = r'^2 + l <r,r'>
   *  2: SOAR: loss(r') = r'^2 + l (cos<r,r'> r')^2   (l=1 by default)
   */
  int redund_strategy = 1;

  /**
   * Whether a second cluster is still needed
   * when loss of nn-cluster is the least.
   *
   * 0(default): not needed (RAIR*).=
   * 1: still needed
   */
  int strict_redund_mode = 0;

  /**
   * Parameter used for RAIR and SOAR.
   * Same as Naive when lambda=0
   */
  float lambda = 0.75;

  /**
   * Whether avoiding visiting and computing distance to the same point.
   * True by default.
   * This should be set false in Flat if ids are not in natural number sequence.
   */
  bool omit_redund_visit = true;

  IndexReIVF(Index* quantizer, size_t d, size_t nlist, size_t code_size,
             size_t nrep = 2, MetricType metric = METRIC_L2);

  IndexReIVF() = default;

  /**
   * return the indexes of the k selected clusters for query x.
   *
   * @param x           input vectors to search, size n * d
   * @param labels      output labels of the k-selected clusters, size n * k
   */
  virtual void cluster_assign(idx_t n, const float* x, idx_t* labels, idx_t k) const;

  void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

  void add_core(idx_t n, const float* x, const idx_t* xids,
                const idx_t* precomputed_idx,
                void* inverted_list_context = nullptr) override;

  void search_preassigned(
      idx_t n,
      const float* x,
      idx_t k,
      const idx_t* assign,
      const float* centroid_dis,
      float* distances,
      idx_t* labels,
      bool store_pairs,
      const IVFSearchParameters* params = nullptr,
      IndexIVFStats* stats = nullptr) const override;

  void update_vectors(int nv, const idx_t* idx, const float* v) override;

  /// TODO: all reconstruction related methods
  /// TODO: rewrite range_search

}; // IndexReIVF

FAISS_API extern std::atomic<int> omit_ndis;

} // namespace faiss::rair

} // namespace rair

#endif // FAISS_RAIR_INDEXREIVF_H
