#ifndef FAISS_RAIR_IMPL_DEDUPSIMDRESULTHANDLERS_H
#define FAISS_RAIR_IMPL_DEDUPSIMDRESULTHANDLERS_H

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/rair/utils/bitmap.h>

namespace faiss {

namespace rair {

struct DedupReservoirStat {
  std::atomic<int> npass_thres{0};
  std::atomic<int> npass_bloom{0};
  std::atomic<int> npass_exist_check{0};
  std::atomic<int> nshrink{0};
  std::atomic<int> nshrink_after_dedup{0};

  void reset() {
    npass_thres = 0;
    npass_bloom = 0;
    npass_exist_check = 0;
    nshrink = 0;
    nshrink_after_dedup = 0;
  }

  void print(int round = 1) {
    printf("npt: %d, npb: %d, dup: %d, ns: %d, nsad: %d\n",
           int(npass_thres)/round, int(npass_bloom)/round,
           int(npass_exist_check)/round, int(nshrink)/round,
           int(nshrink_after_dedup)/round);
  }
};

extern DedupReservoirStat dedupReservoirStat;

/**
 * Reservoir for a single query but with deduplication for id
 * See faiss::ReservoirTopN
 */
template <class C>
struct DedupReservoirTopN : ResultHandler<C> {
  using T = typename C::T;
  using TI = typename C::TI;
  using ResultHandler<C>::threshold;

  T* vals;
  TI* ids; ///< aligned with 64B

  size_t i;        ///< number of stored elements
  size_t n;        ///< number of requested elements
  size_t capacity; ///< size of storage

  size_t list_cnt{0};

  static constexpr size_t NLIST = 1024;
  bitmap<NLIST> visited;

  DedupReservoirTopN() {}

  DedupReservoirTopN(size_t n, size_t capacity, T* vals, TI* ids)
      : vals(vals), ids(ids), i(0), n(n), capacity(capacity) {
    threshold = C::neutral();
    visited.reset();
  }

  bool add_result(T val, TI id) final {
    bool update_thres = false;
    if (i == cap_) {
      shrink_fuzzy();
      update_thres = true;
    }
    vals[i] = val;
    ids[i] = id;
    i ++;
    return update_thres;
  }

  size_t n_ = n;
  size_t cap_ = capacity;
  /// reduce storage from capacity to some value
  /// between n_ and (cap_ + n_) / 2
  inline void shrink_fuzzy() {
    threshold = partition_fuzzy<C>(vals, ids, cap_, n_, (cap_+n_)/2, &i);
  }

  void final_shrink() {
    threshold = partition<C>(vals, ids, i, n);
    i = n;
    /// till now, no need to set deduplication info (set or bloom filter) here
  }

}; // DedupReservoirTopN

} // namespace faiss::rair

namespace simd_result_handlers {

namespace rair {

struct DedupHandlerInterface {
  // bool dedup; // should be declared in all derived class
  virtual bool list_visited(size_t q_idx, int list_no) = 0;
  virtual void end_list(std::vector<int>& qvec, int list_no) = 0;
};

/** Simple top-N implementation using a reservoir.
 *
 * Results are stored when they are below the threshold until the capacity is
 * reached. Then a partition sort is used to update the threshold. */

/** Handler built from several ReservoirTopN (one per query)
 * Difference with ReservoirHandler:
 * deduplicate id, since IndexReIVF puts vector in multiple buckets
 */
template <class C, bool with_id_map = false>
struct DedupReservoirHandler : ResultHandlerCompare<C, with_id_map>,
                               DedupHandlerInterface {
  using T = typename C::T;
  using TI = typename C::TI;
  using RHC = ResultHandlerCompare<C, with_id_map>;
  using RHC::normalizers;

  size_t capacity; // rounded up to multiple of 16
  bool dedup = true;

  // where the final results will be written
  float* dis;
  int64_t* ids;

  AlignedTable<TI, 64> all_ids;
  AlignedTable<T> all_vals;
  std::vector<faiss::rair::DedupReservoirTopN<C>> reservoirs;

  DedupReservoirHandler(size_t nq, size_t ntotal, size_t k, size_t cap,
      float* dis, int64_t* ids):
      RHC(nq, ntotal), DedupHandlerInterface(),
      capacity((cap + 15) & ~15), dis(dis), ids(ids) {
    all_ids.resize(nq * capacity);
    all_vals.resize(nq * capacity);
    for (size_t q = 0; q < nq; q ++) {
      reservoirs.emplace_back(k, capacity, all_vals.get() + q*capacity,
                              all_ids.data() + q*capacity);
    }
  }

  bool list_visited(size_t q_idx, int list_no) final {
    size_t q = q_idx + this->i0;
    if constexpr (with_id_map) {
      q = this->q_map[q];
    }
    return reservoirs[q].visited.get(list_no);
  }

  void end_list(std::vector<int>& qvec, int list_no) final {
    for (int q : qvec) {
      reservoirs[q].visited.set_true(list_no);
      reservoirs[q].list_cnt ++;
    }
  }

  void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
    if (this->disable) {
      return;
    }
    this->adjust_with_origin(q, d0, d1); ///< q is modified to q_map[q+i0]

    faiss::rair::DedupReservoirTopN<C>& res = reservoirs[q];
    uint32_t lt_mask = this->get_lt_mask(res.threshold, b, d0, d1);

    if (!lt_mask) {
      return;
    }
    ALIGNED(32) uint16_t d32tab[32];
    d0.store(d32tab);
    d1.store(d32tab + 16);

    /// Stage 1: list_cnt == 0: no deduplication when visiting first bucket.
    if (!dedup || res.list_cnt == 0) {
      while (lt_mask) {
        int j = __builtin_ctz(lt_mask); ///< find first non-zero
        int64_t idx_with_list_no = this->adjust_id(b, j);
        T dis_j = d32tab[j];
        lt_mask -= 1 << j;
        if (C::cmp(res.threshold, dis_j)) {
          if (res.i == res.cap_) res.shrink_fuzzy();
          res.vals[res.i] = dis_j;
          res.ids[res.i] = (TI)idx_with_list_no; ///< trick: for sizeof(TI)==4
          res.i ++;
        }
      }
      return;
    }

    /// Stage 2: duplication may happen
    /// prefetching seems no use here
    while (lt_mask) {
      int j = __builtin_ctz(lt_mask); ///< find first non-zero
      int64_t idx_with_list_no = this->adjust_id(b, j);
      T dis_j = d32tab[j];
      lt_mask -= 1 << j;
      int nodup_mask = !res.visited.get(idx_with_list_no >> 32);
      nodup_mask &= C::cmp(res.threshold, dis_j);
      if (res.i == res.cap_) res.shrink_fuzzy();
      res.vals[res.i] = dis_j;
      res.ids[res.i] = (TI)idx_with_list_no; ///< trick: for sizeof(TI)==4
      res.i += nodup_mask;
    }
  }

  void end() override {
    using Cf = typename std::conditional<
        C::is_max,
        CMax<float, int64_t>,
        CMin<float, int64_t>>::type;

    for (int q = 0; q < reservoirs.size(); q++) {
      faiss::rair::DedupReservoirTopN<C>& res = reservoirs[q];
      size_t n = res.n;

      if (res.i > res.n) {
        res.final_shrink();
      }
      int64_t* heap_ids = ids + q * n;
      float* heap_dis = dis + q * n;

      float one_a = 1.0, b = 0.0;
      if (normalizers) {
        one_a = 1 / normalizers[2 * q];
        b = normalizers[2 * q + 1];
      }
      std::vector<int> perm(res.i);
      for (int i = 0; i < res.i; i++) {
        perm[i] = i;
      }
      // indirect sort of result arrays
      std::sort(perm.begin(), perm.begin() + res.i, [&res](int i, int j) {
        return C::cmp2(res.vals[j], res.vals[i], res.ids[j], res.ids[i]);
      });
      for (int i = 0; i < res.i; i++) {
        heap_dis[i] = res.vals[perm[i]] * one_a + b;
        heap_ids[i] = res.ids[perm[i]];
      }
      // possibly add empty results
      heap_heapify<Cf>(n - res.i, heap_dis + res.i, heap_ids + res.i);
    }
  }
}; // DedeupReservoirHandler

/*******************************************************************************
 * Dynamic dispatching function. The consumer should have a templatized method f
 * that will be replaced with the actual SIMDResultHandler that is determined
 * dynamically.
 */

template <class C, bool W, class Consumer, class... Types>
void dispatch_DedupSIMDResultHanlder_fixedCW(
    SIMDResultHandler& res,
    Consumer& consumer,
    Types... args) {
  if (auto resh = dynamic_cast<DedupReservoirHandler<C, W>*>(&res)) {
    consumer.template f<DedupReservoirHandler<C, W>>(*resh, args...);
  } else { // generic handler -- will not be inlined
    FAISS_THROW_IF_NOT_FMT(
        false,
        "Handler type \"%s\" does not support deduplication.",
        typeid(res).name());
  }
}

template <class C, class Consumer, class... Types>
void dispatch_DedupSIMDResultHanlder_fixedC(
    SIMDResultHandler& res,
    Consumer& consumer,
    Types... args) {
  if (res.with_fields) {
    dispatch_DedupSIMDResultHanlder_fixedCW<C, true>(res, consumer, args...);
  } else {
    dispatch_DedupSIMDResultHanlder_fixedCW<C, false>(res, consumer, args...);
  }
}

template <class Consumer, class... Types>
void dispatch_DedupSIMDResultHanlder(
    SIMDResultHandler& res,
    Consumer& consumer,
    Types... args) {
  if (res.sizeof_ids == 0) {
    FAISS_THROW_IF_NOT_MSG(false, "res.sizeof_ids == 0");
  } else if (res.sizeof_ids == sizeof(int)) {
    if (res.is_CMax) {
      dispatch_DedupSIMDResultHanlder_fixedC<CMax<uint16_t, int>>(
          res, consumer, args...);
    } else {
      dispatch_DedupSIMDResultHanlder_fixedC<CMin<uint16_t, int>>(
          res, consumer, args...);
    }
  } else if (res.sizeof_ids == sizeof(int64_t)) {
    if (res.is_CMax) {
      dispatch_DedupSIMDResultHanlder_fixedC<CMax<uint16_t, int64_t>>(
          res, consumer, args...);
    } else {
      dispatch_DedupSIMDResultHanlder_fixedC<CMin<uint16_t, int64_t>>(
          res, consumer, args...);
    }
  } else {
    FAISS_THROW_FMT("Unknown id size %d", res.sizeof_ids);
  }
}

} // namespace faiss::simd_result_handlers::rair

} // namespace faiss::simd_result_handlers

} // namespace faiss

#endif // FAISS_RAIR_IMPL_DEDUPSIMDRESULTHANDLERS_H
