#include <algorithm>
#include <vector>
#include <omp.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/rair/IndexReIVFFastScan.h>
#include <faiss/rair/impl/dedup_simd_result_handlers.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/rair/impl/dedup_pq4_fast_scan.h>
#include <faiss/IndexIVFFastScan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/quantize_lut.h>

using namespace faiss;
using namespace faiss::rair;
using namespace faiss::simd_result_handlers::rair;

IndexReIVFFastScan::IndexReIVFFastScan(
    Index* quantizer, size_t d, size_t nlist,
    size_t code_size, size_t nrep, MetricType metric
    ): IndexReIVF(quantizer, d, nlist, code_size, nrep, metric) {
  /// unlike other indexes, we prefer no residuals for performance reasons.
  by_residual = false;
  omit_redund_visit = true;
  FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
}

void IndexReIVFFastScan::init_fastscan(size_t M_, size_t nbits_, size_t nlist_,
                                       MetricType, int bbs_) {
  M = M_; nbits = nbits_; bbs = bbs_;
  ksub = (1 << nbits);
  M2 = (M + 1) / 2 * 2;
  code_size = M2 / 2;
  is_trained = false;
  FAISS_THROW_IF_NOT(bbs == 32);
  FAISS_THROW_IF_NOT(nbits == 4);
  replace_invlists(new BlockInvertedLists(nlist_, get_CodePacker()), true);
  block_color.clear();
  block_color.resize(nlist_);
  area_bound.clear();
  area_bound.resize(nlist_);
}

void IndexReIVFFastScan::init_code_packer() {
  auto bil = dynamic_cast<BlockInvertedLists*>(invlists);
  FAISS_THROW_IF_NOT(bil);
  delete bil->packer; ///< in case there was one before
  bil->packer = get_CodePacker();
}

/**
 * Code management functions
 */

/// Cell: (list_no, the_other_list_no)
///
/// Memory layout optimization:
///   Put vectors of the same cell
///   in the same block as much as possible.
///
/// Sundries: small cells and remainder of big cells
///
/// Optimized layout:
///   <head sundries area> | <big-cell blocks area> | <tail sundries area>
///   head sundries area: old_list_sz ... empty_blocks_start_pos
///   big-cell blocks area: empty_blocks_start_pos ... end_pos of some block
///   tail sundries area: end_pos of the block ... new_list_sz
///   Attention: end_pos of the block <= empty_blocks_end_pos

void
IndexReIVFFastScan::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
  FAISS_THROW_IF_NOT(is_trained);
  /// do some blocking to avoid excessive allocs
  constexpr idx_t bs = (1<<27); // in IndexIVFFastScan: 65536
  if (n > bs) {
    for (idx_t i0 = 0; i0 < n; i0 += bs) {
      idx_t i1 = std::min(n, i0 + bs);
      add_with_ids(i1 - i0, x + i0 * d, xids ? xids + i0 : nullptr);
    }
    return;
  }
  InterruptCallback::check();

  direct_map.check_can_add(xids);
  std::unique_ptr<idx_t[]> coarse_idx(new idx_t[nrep*n]);
  cluster_assign(n, x, coarse_idx.get(), nrep);

  AlignedTable<uint8_t> flat_codes(n * code_size);
  encode_vectors(n, x, nullptr /* list_nos */, flat_codes.get(), false);
  DirectMapAdd dm_adder(direct_map, n, xids); ///< no use by default

  auto* bil = dynamic_cast<BlockInvertedLists*>(invlists);
  FAISS_THROW_IF_NOT_MSG(bil, "only block inverted lists supported");

  /// arg sort for coarse_idx
  std::vector<idx_t> order(n * nrep);
  for (size_t i = 0; i < n * nrep; i ++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&coarse_idx](idx_t a, idx_t b) {
    /// For the same list, order by the_other_list_no
    idx_t a_the_other = (coarse_idx[a^1]==-1) ? coarse_idx[a] : coarse_idx[a^1];
    idx_t b_the_other = (coarse_idx[b^1]==-1) ? coarse_idx[b] : coarse_idx[b^1];
    return (coarse_idx[a] != coarse_idx[b]) ? (coarse_idx[a] < coarse_idx[b])
    : (a_the_other < b_the_other);
  });

  std::unique_ptr<size_t[]> cnt_in_cells(new size_t[nlist * nlist]);
  memset(cnt_in_cells.get(), 0, nlist * nlist * sizeof(size_t));
  size_t sundries_cnt = 0, non_sundries_cnt = 0;

  size_t i0 = 0; ///< start order of a list
  while (i0 < n * nrep) { ///< list iteration
    idx_t list_no = coarse_idx[order[i0]];
    size_t i1 = i0 + 1; ///< will be the end order of this list
    while (i1 < n * nrep && coarse_idx[order[i1]] == list_no) i1 ++;
    /// Now for all j in i0...i1, coarse_idx[order[j]] is min of unvisited list
    if (list_no == -1) { i0 = i1; continue; }
    FAISS_ASSERT(list_no >= 0);

    /// Extend the list_no list of BlockInvertedList
    AlignedTable<uint8_t> list_codes((i1 - i0) * code_size);
    size_t old_list_sz = bil->list_size(list_no);
    size_t new_list_sz = old_list_sz + i1 - i0;
    bil->resize(list_no, new_list_sz);
    block_color[list_no].resize((new_list_sz + 31) >> 5, -1);
    size_t empty_blocks_start_pos = ((old_list_sz + bbs - 1) & ~(bbs - 1));
    size_t empty_blocks_end_pos = (new_list_sz & ~(bbs - 1));
    size_t big_cell_blocks_ptr = empty_blocks_start_pos;
    size_t sundries_ptr = (old_list_sz < empty_blocks_start_pos) ?
        old_list_sz : (new_list_sz - 1);

    size_t i00 = i0; ///< start order of a cell (list_no, the_other_list_no)
    while (i00 < i1) { ///< cell iteration
      idx_t the_other_list_no = coarse_idx[order[i00]^1];
      size_t i01 = i00 + 1; ///< will be the end order of this cell
      while (i01 < i1 && coarse_idx[order[i01]^1] == the_other_list_no) i01 ++;
      if (the_other_list_no == -1) the_other_list_no = list_no;
      cnt_in_cells[list_no * nlist + the_other_list_no] += (i01 - i00);

      for (size_t i = i00; i < i01; i ++) {
        idx_t vec_seq = order[i] / nrep;
        idx_t id = xids ? xids[vec_seq] : (ntotal+vec_seq);
        id |= (the_other_list_no << 32); ///< Put it in id for dedup

        size_t vec_pos = old_list_sz + i - i0; // w/o mem layout optimization
        // With mem layout optimization
        bool is_sundries = false;
        if ((i01-i00)/bbs == (i-i00)/bbs) is_sundries = true;
        if (big_cell_blocks_ptr >= empty_blocks_end_pos) is_sundries = true;
        if (is_sundries) {
          if (sundries_ptr < empty_blocks_start_pos) { ///< put in head area
            vec_pos = sundries_ptr;
            sundries_ptr ++;
            if (sundries_ptr == empty_blocks_start_pos) {
              sundries_ptr = new_list_sz - 1; ///< switch to tail sundries area
            }
          } else { ///< put in tail sundries area
            vec_pos = sundries_ptr;
            sundries_ptr --;
          }
          sundries_cnt ++;
        } else { ///< not sundries
          vec_pos = big_cell_blocks_ptr;
          block_color[list_no][vec_pos>>5] = the_other_list_no;
          big_cell_blocks_ptr ++;
          non_sundries_cnt ++;
        }

        bil->ids[list_no][vec_pos] = id;
        dm_adder.add(vec_seq, list_no, vec_pos);
        memcpy(list_codes.data() + (vec_pos - old_list_sz) * code_size,
               flat_codes.data() + vec_seq * code_size, code_size);
      }
      i00 = i01;
    } ///< End of the cell<list, the_other_list>

    /// Construct area_bound
    area_bound[list_no].push_back(0);
    for (size_t blk_no = 0; blk_no < block_color[list_no].size(); blk_no ++) {
      if (area_bound[list_no].size() % 3 == 1) { ///< in unsafe big-cell blocks
        if (list_no <= block_color[list_no][blk_no]) {
          /// encounter safe big-cell block
          area_bound[list_no].push_back(blk_no);
        } else if (block_color[list_no][blk_no] == -1) {
          /// directly encounter sundries block without safe blocks
          /// duplicate insertion means no safe block
          area_bound[list_no].push_back(blk_no);
          area_bound[list_no].push_back(blk_no);
        }
      } else if (area_bound[list_no].size() % 3 == 2) { ///< safe big-cell blocks
        if (block_color[list_no][blk_no] == -1) {
          /// encounter sundries block
          area_bound[list_no].push_back(blk_no);
        } else if (list_no > block_color[list_no][blk_no]) {
          /// directly encounter unsafe big-cell block without sundries blocks
          /// duplicate insertion means no sundries block
          area_bound[list_no].push_back(blk_no);
          area_bound[list_no].push_back(blk_no);
        }
      } else { ///< checking sundries blocks now
        if (block_color[list_no][blk_no] != -1
        && list_no > block_color[list_no][blk_no]) {
          /// encounter unsafe big-cell block
          area_bound[list_no].push_back(blk_no);
        } else if (list_no <= block_color[list_no][blk_no]) {
          /// directly encounter safe big-cell block without unsafe blocks
          /// duplicate insertion means no unsafe block
          area_bound[list_no].push_back(blk_no);
          area_bound[list_no].push_back(blk_no);
        }
      } ///< end if
    } ///< end block iteration

    pq4_pack_codes_range(list_codes.data(), M, old_list_sz, new_list_sz,
                         bbs, M2, bil->codes[list_no].data());
    i0 = i1;

  } ///< End of the list

  ntotal += n;

//  for (size_t i = 0; i < nlist; i ++) {
//    for (size_t j = 0; j < nlist; j ++) {
//      printf("%lu,", cnt_in_cells[i*nlist+j]);
//    }
//    printf("\n");
//  }
//  exit(0);
}

CodePacker* IndexReIVFFastScan::get_CodePacker() const {
  return new CodePackerPQ4(M, bbs);
}

/**
 * Search
 */

void
IndexReIVFFastScan::search(idx_t n, const float* x, idx_t k, float* distances,
                           idx_t* labels, const SearchParameters* params) const {
  auto paramsi = dynamic_cast<const SearchParametersIVF*>(params);
  FAISS_THROW_IF_NOT_MSG(!params || paramsi, "need IVFSearchParameters");

  if (omit_redund_visit) {
    search_preassigned(n, x, k, nullptr, nullptr, distances, labels,
                       false, paramsi);
    return;
  }

  idx_t k_ = nrep * k;
  std::unique_ptr<float[]> distances_(new float[n*k_]);
  std::unique_ptr<idx_t[]> labels_(new idx_t[n*k_]);

  search_preassigned(n, x, k_, nullptr, nullptr, distances_.get(), labels_.get(),
                     false, paramsi);

  /// deduplicate the same label
  for (idx_t i = 0; i < n; i ++) {
    idx_t cur = 0, cnt = 0;
    while (cnt < k) {
      FAISS_ASSERT(cur < k_);
      if (cnt == 0 || labels_[i*k_+cur] != labels[i*k+cnt-1]) {
        distances[i*k+cnt] = distances_[i*k_+cur];
        labels[i*k+cnt] = labels_[i*k_+cur];
        cnt ++;
      }
      cur ++;
    }
  }
}

void IndexReIVFFastScan::search_preassigned(
    idx_t n, const float* x, idx_t k, const idx_t* assign,
    const float* centroid_dis, float* distances, idx_t* labels, bool store_pairs,
    const IVFSearchParameters* params, IndexIVFStats* stats) const {
  size_t nprobe = this->nprobe;
  if (params) {
    FAISS_THROW_IF_NOT_MSG(!params->quantizer_params,
                           "quantizer params not supported");
    FAISS_THROW_IF_NOT(params->max_codes == 0);
    nprobe = params->nprobe;
  }
  FAISS_THROW_IF_NOT_MSG(!store_pairs,
                         "store_pairs not supported for this index");
  FAISS_THROW_IF_NOT_MSG(!stats, "stats not supported for this index");
  FAISS_THROW_IF_NOT(k > 0);

  /// If this is called by search(), centroid_dis==nullptr and assign==nullptr
  const CoarseQuantized cq = {nprobe, centroid_dis, assign};
  search_dispatch_implem(n, x, k, distances, labels, cq, nullptr);
}

namespace {

template <class C, bool dedup>
simd_result_handlers::ResultHandlerCompare<C, true>* make_knn_handler_fixC(
    int impl, idx_t n, idx_t k, float* distances, idx_t* labels) {
  if (k == 1) {
    return new simd_result_handlers::SingleResultHandler<C, true>(
        n, 0, distances, labels);
  } else if (impl % 2 == 0) {
    if constexpr (dedup) {
      FAISS_ASSERT_MSG(false, "We strongly recommend you to set "
                              "omit_redund_dis=false for k<20.");
    } else {
      return new simd_result_handlers::HeapHandler<C, true>(
          n, 0, k, distances, labels);
    }
  } else /* if (impl % 2 == 1) */ {
    if constexpr (dedup) {
      return new DedupReservoirHandler<C, true>(n, 0, k, 2 * k, distances,
                                                labels);
    } else {
      return new simd_result_handlers::ReservoirHandler<C, true>(
          n, 0, k, 2 * k, distances, labels);
    }
  }
}

SIMDResultHandlerToFloat* make_knn_handler(
    bool is_max, bool dedup, int impl, idx_t n, idx_t k, float* distances,
    idx_t* labels) {
  if (is_max) {
    if (dedup) {
      return make_knn_handler_fixC<CMax<uint16_t, int32_t>, true>(
          impl, n, k, distances, labels);
    } else {
      return make_knn_handler_fixC<CMax<uint16_t, int64_t>, false>(
          impl, n, k, distances, labels);
    }
  } else {
    if (dedup) {
      return make_knn_handler_fixC<CMin<uint16_t, int32_t>, true>(
          impl, n, k, distances, labels);
    } else {
      return make_knn_handler_fixC<CMin<uint16_t, int64_t>, false>(
          impl, n, k, distances, labels);
    }
  }
}

using CoarseQuantized = IndexReIVFFastScan::CoarseQuantized;

struct CoarseQuantizedWithBuffer : CoarseQuantized {
  explicit CoarseQuantizedWithBuffer(const CoarseQuantized& cq)
      : CoarseQuantized(cq) {}

  std::vector<idx_t> ids_buffer;
  std::vector<float> dis_buffer;

  void quantize(const Index* quantizer, idx_t n, const float* x) {
    dis_buffer.resize(nprobe * n);
    ids_buffer.resize(nprobe * n);
    quantizer->search(n, x, nprobe, dis_buffer.data(), ids_buffer.data());
    dis = dis_buffer.data();
    ids = ids_buffer.data();
  }
};

struct CoarseQuantizedSlice : CoarseQuantizedWithBuffer {
  size_t i0, i1;
  CoarseQuantizedSlice(const CoarseQuantized& cq, size_t i0, size_t i1)
      : CoarseQuantizedWithBuffer(cq), i0(i0), i1(i1) {
    if (ids) {
      dis += nprobe * i0;
      ids += nprobe * i0;
    }
  }

  void quantize_slice(const Index* quantizer, const float* x) {
    quantize(quantizer, i1 - i0, x + quantizer->d * i0);
  }
};

} // namespace

void IndexReIVFFastScan::search_dispatch_implem(
    idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
    const CoarseQuantized& cq_in, const NormTableScaler* scaler) const {
  bool is_max = !is_similarity_metric(metric_type);
  using RH = SIMDResultHandlerToFloat;

  /// actual implementation used
  int impl = implem;
  if (impl == 0) {
    impl = bbs==32 ? 12 : 10;
    if (k > 20) impl ++;  ///< use reservoir rather than heap
  }
  FAISS_ASSERT_MSG(impl==12 || impl==13, "other strategy not implemented");

  CoarseQuantizedWithBuffer cq(cq_in);
  size_t ndis = 0, nlist_visited = 0;
  int nslice = std::min(omp_get_max_threads(), int(n));

#pragma omp parallel for reduction(+ : ndis, nlist_visited)
  for (int slice = 0; slice < nslice; slice ++) {
    idx_t i0 = n * slice / nslice;
    idx_t i1 = n * (slice + 1) / nslice;
    float* dis_i = distances + i0 * k;
    idx_t* lab_i = labels + i0 * k;
    CoarseQuantizedSlice cq_i(cq, i0, i1);
    if (!cq_i.ids) { ///< If called from search(), we do coarse quantization.
      cq_i.quantize_slice(quantizer, x);
    }
    std::unique_ptr<RH> handler(make_knn_handler(
        is_max, omit_redund_visit, impl, i1 - i0, k, dis_i, lab_i
        ));
    search_implem_12_13(i1 - i0, x + i0 * d, *(handler.get()), cq_i,
                        &ndis, &nlist_visited, scaler);
  }

  indexIVF_stats.nq += n;
  indexIVF_stats.ndis += ndis;
  indexIVF_stats.nlist += nlist_visited;
}

void IndexReIVFFastScan::search_implem_12_13(
    idx_t n, const float* x, SIMDResultHandlerToFloat& handler,
    const CoarseQuantized& cq, size_t* ndis_out, size_t* nlist_out,
    const NormTableScaler* scaler) const {
  if (n == 0) return;
  FAISS_THROW_IF_NOT(bbs == 32);
  FAISS_ASSERT(cq.dis && cq.ids);

  AlignedTable<uint8_t> dis_tables;
  std::unique_ptr<float[]> normalizers(new float[2 * n]);
  compute_LUT_uint8(n, x, dis_tables, normalizers.get());
  handler.begin(skip ? nullptr : normalizers.get());

  struct QC {
    int qno;     ///< sequence number of the query
    int list_no; ///< list to visit
    int rank;    ///< this is the rank'th result of the coarse quantizer
  };
  std::vector<QC> qcs;
  for (int i = 0; i < n; i ++) {
    for (size_t j = 0; j < cq.nprobe; j ++) {
      if (cq.ids[i * cq.nprobe + j] >= 0) {
        qcs.push_back(QC{i, int(cq.ids[i * cq.nprobe + j]), int(j)});
      }
    }
  }
  std::sort(qcs.begin(), qcs.end(), [](const QC& a, const QC& b) {
    return a.list_no < b.list_no;
  });

  int qbs2_ = qbs2 ? qbs2 : 11; ///< nc <= qbs2_ (== 11 by default)
  size_t ndis = 0, i0 = 0;

  while (i0 < qcs.size()) {
    /// find all queries that access this inverted list
    int list_no = qcs[i0].list_no;
    size_t i1 = i0 + 1;
    while (i1 < qcs.size() && i1 < i0 + qbs2_) {
      if (qcs[i1].list_no != list_no) {
        break;
      }
      i1 ++;
    }
    size_t list_size = invlists->list_size(list_no);
    if (list_size == 0) {
      i0 = i1;
      continue;
    }

    /// re-organize LUTs and biases into the right order
    int nc = i1 - i0;
    std::vector<int> q_map(nc), lut_entries(nc);
    AlignedTable<uint8_t> LUT(nc * ksub * M2); ///< LUT packed for QBS
    memset(LUT.get(), -1, nc * ksub * M2);
    AlignedTable<uint8_t> LUT2(nc * ksub * M2); ///< LUT packed for single query
    memset(LUT2.get(), -1, nc * ksub * M2);
    int qbs_ = pq4_preferred_qbs(nc); ///< 0x2333 if nc==11
    for (size_t i = i0; i < i1; i ++) {
      const QC& qc = qcs[i];
      q_map[i - i0] = qc.qno;
      lut_entries[i - i0] = qc.qno;
    }
    pq4_pack_LUT_qbs_q_map(
        qbs_, M2, dis_tables.get(), lut_entries.data(), LUT.get());

    /// access the inverted list
    InvertedLists::ScopedCodes codes(invlists, list_no);
    InvertedLists::ScopedIds ids(invlists, list_no);
    handler.q_map = q_map.data();
    if (!omit_redund_visit || !block_pruning) { ///< treat the list as a whole
      ndis += (i1 - i0) * list_size;
      handler.ntotal = list_size;
      handler.id_map = ids.get();
    } else { ///< treat the list area by area
      for (int q = 0; q < nc; q ++) {
        pq4_pack_LUT_qbs_q_map(
            0x1, M2, dis_tables.get(), lut_entries.data() + q,
            LUT2.get() + q * ksub * M2);
      }
    }

    if (omit_redund_visit) {
      if (block_pruning) { ///< Area-by-area
        for (size_t area = 0; area < area_bound[list_no].size(); area ++) {
          size_t start = area_bound[list_no][area] * 32;
          size_t end = (area == area_bound[list_no].size() - 1) ? list_size
              : area_bound[list_no][area+1] * 32;
          size_t area_size = end - start;
          if (area_size == 0) continue;
          handler.ntotal = area_size;
          handler.id_map = ids.get() + start;
          ndis += (i1 - i0) * area_size;
          if (area % 3 == 0) { ///< Unsafe big-cell area: dedup by block pruning
            /// kernel
            size_t nprune = dedup_pq4_accumulate_loop_1_with_block_pruning(
                nc, area_size, M2, codes.get() + start * code_size,
                block_color[list_no].data() + area_bound[list_no][area],
                LUT2.get(), handler, scaler
                );
            ndis -= nprune * 32;
          } else if (area % 3 == 1) { ///< Safe big-cell area: no dedup at all
            /// kernel
            dedup_pq4_accumulate_loop_qbs<false>(
                qbs_, area_size, M2, codes.get() + start * code_size,
                LUT.get(), handler, scaler
                );
          } else { ///< Sundries area: dedup but no block pruning
            /// kernel
            dedup_pq4_accumulate_loop_qbs<true>(
                qbs_, area_size, M2, codes.get() + start * code_size,
                LUT.get(), handler, scaler
                );
          } ///< end if
        } ///< end for
      } else { ///< no block pruning at all
        /// kernel
        dedup_pq4_accumulate_loop_qbs<true>(
            qbs_, list_size, M2, codes.get(), LUT.get(), handler, scaler
            );
      } ///< end if(block_pruning)
      auto* dhi = dynamic_cast<DedupHandlerInterface*>(&handler);
      if (dhi) { dhi->end_list(q_map, list_no); }
    } else { ///< no omit_redund_visit
      pq4_accumulate_loop_qbs(
          qbs_, list_size, M2, codes.get(), LUT.get(), handler, scaler
          );
    } ///< end if(omit_redund_visit)

    i0 = i1;
  }

  handler.end();
  *ndis_out = ndis;
  *nlist_out = nlist;
}

/**
 * Look-up table
 */
void IndexReIVFFastScan::compute_LUT_uint8(
    size_t n, const float* x, AlignedTable<uint8_t>& dis_tables,
    float* normalizers) const {
  AlignedTable<float> dis_tables_float;
  compute_LUT(n, x, dis_tables_float);
  dis_tables.resize(n * ksub * M2);

#pragma omp parallel for if (n > 100)
  for (size_t i = 0; i < n; i ++) {
    const float* t_in = dis_tables_float.get() + i * ksub * M;
    uint8_t* t_out = dis_tables.get() + i * ksub * M2;
    quantize_lut::quantize_LUT_and_bias(
        nprobe, M, ksub, false, t_in, nullptr, t_out, M2, nullptr,
        normalizers + 2 * i, normalizers + 2 * i + 1
    );
  }
}

faiss::rair::DedupReservoirStat faiss::rair::dedupReservoirStat;
