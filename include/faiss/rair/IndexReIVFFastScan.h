#ifndef FAISS_RAIR_INDEXREIVFFASTSCAN_H
#define FAISS_RAIR_INDEXREIVFFASTSCAN_H

#include <cstdint>
#include <vector>
#include <faiss/utils/AlignedTable.h>
#include <faiss/rair/IndexReIVF.h>

namespace faiss {

struct NormTableScaler;
struct SIMDResultHandlerToFloat;

namespace rair {

/** Fast scan version of ReIVFPQ/ReIVFAQ. Works for 4-bit PQ/AQ for now.
 *
 * The codes in the inverted lists are not stored sequentially but
 * grouped in blocks of size bbs. This makes it possible to very quickly
 * compute distances with SIMD instructions.
 */
struct IndexReIVFFastScan : faiss::rair::IndexReIVF {

  int bbs = 32; ///< size of the kernel; set at build time

  size_t M; ///< group number of PQ
  size_t nbits; ///< bit per group
  size_t ksub; ///< cluster number for each group; (1 << nbits) by default

  size_t M2; ///< M rounded up to a multiple of 2

  /** search-time implementation
   *
   * 0: auto-select implementation (default)
   * 1: origin search, re-implemented
   * 2: origin search, re-ordered by invlist
   * 10: optimizer int16 search, collect results in heap, no qbs
   * 11: idem, collect results in reservoir
   * 12: optimizer int16 search, collect results in heap, uses qbs
   * 13: idem, collect results in reservoir
   * 14: internally multithreaded implem over nq * nprobe
   * 15: same with reservoir
   */
  int implem = 0;

  /// skip some parts(de-normalization) of the computation (for timing)
  /// If true, the returned dis would be wrong, but the ids are not influenced.
  bool skip = false;

  int qbs = 0; ///< batching factors at search time (0 = default)
  size_t qbs2 = 0;

  /// Prune in blocks before distance computing
  bool block_pruning = true;

  /// block_color[list_no][block_no] = the_other_list_no of vec in this block.
  /// = -1 if indefinite, which means the block is for sundries.
  ::std::vector<::std::vector<int32_t>> block_color;

  /// The start no. of blocks of unsafe big-cell area,
  /// safe big-cell and sundries area in sequence.
  ///
  /// Safe: list_no <= the_other_list_no,
  /// which means no deduplication is needed.
  ::std::vector<::std::vector<uint32_t>> area_bound;


  IndexReIVFFastScan(Index* quantizer, size_t d, size_t nlist,
                     size_t code_size, size_t nrep = 2,
                     MetricType metric = METRIC_L2);
  IndexReIVFFastScan() = default;

  void init_fastscan(size_t M, size_t nbits, size_t nlist, MetricType metric,
                     int bbs);

  /// initialize the CodePacker in the InvertedLists
  void init_code_packer();

  void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

  CodePacker* get_CodePacker() const override;

  /// compact way of conveying coarse quantization results
  struct CoarseQuantized {
    size_t nprobe;
    const float* dis = nullptr;
    const idx_t* ids = nullptr;
  };

  virtual void compute_LUT(size_t n, const float* x,
                           AlignedTable<float>& dis_tables) const = 0;

  void compute_LUT_uint8(size_t n, const float* x,
                         AlignedTable<uint8_t>& dis_tables,
                         float* normalizers) const;

  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void search_preassigned(idx_t n, const float* x, idx_t k, const idx_t* assign,
                          const float* centroid_dis, float* distances,
                          idx_t* labels, bool store_pairs,
                          const IVFSearchParameters* params = nullptr,
                          IndexIVFStats* stats = nullptr) const override;

  /// dispatch to implementations and parallelize
  void search_dispatch_implem(idx_t n, const float* x, idx_t k, float* distances,
                              idx_t* labels, const CoarseQuantized& cq,
                              const NormTableScaler* scaler) const;

  /// optimizer int16 search, uses qbs, collect results in heap/reservoir
  void search_implem_12_13(
      idx_t n, const float* x, SIMDResultHandlerToFloat& handler,
      const CoarseQuantized& cq, size_t* ndis_out, size_t* nlist_out,
      const NormTableScaler* scaler) const;

  /// TODO: rewrite range_search
  /// TODO: reconstruct_from_offset
  /// TODO: other search implementations

}; // IndexReIVFFastScan

} // namespace faiss::rair

} // namespace rair

#endif // FAISS_RAIR_INDEXREIVFFASTSCAN_H
