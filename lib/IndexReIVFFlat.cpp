#include <cinttypes>
#include <cstring>
#include <memory>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/rair/IndexReIVFFlat.h>
#include <omp.h>

using namespace faiss;
using namespace faiss::rair;

#ifdef __GNUC__
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x)   (x)
#define unlikely(x) (x)
#endif

IndexReIVFFlat::IndexReIVFFlat(Index* quantizer, size_t d, size_t nlist,
                               size_t nrep, MetricType metric):
                               IndexReIVF(quantizer, d, nlist, sizeof(float)*d,
                                          nrep, metric) {
  /* empty */
}

void
IndexReIVFFlat::encode_vectors(idx_t n, const float* x, const idx_t* list_nos,
                               uint8_t* codes, bool include_listnos) const {
  FAISS_ASSERT_MSG(!include_listnos, "Not implemented");
  memcpy(codes, x, code_size * n);
}

void IndexReIVFFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
  size_t coarse_size = coarse_code_size();
  for (idx_t i = 0; i < n; i++) {
    const uint8_t* code = bytes + i * (code_size + coarse_size);
    float* xi = x + i * d;
    memcpy(xi, code + coarse_size, code_size);
  }
}

namespace {

class RairBitmap {
public:
  explicit RairBitmap(size_t size): len_((size + 31) >> 5) {
    bits_ = std::make_unique<uint32_t[]>(len_);
    reset();
  }
  RairBitmap() = default;

  void set(size_t index, bool value) {
    if (value) bits_[index>>5] |= (1u << (index&31));
    else bits_[index>>5] &= ~(1u << (index&31));
  }

  bool get(size_t index) const {
    return (bits_[index>>5] >> (index&31)) & 1u;
  }

  bool get_and_set_true(size_t index) {
    size_t word_index = index >> 5;
    size_t bit_index = index & 31;
    bool ret = (bits_[word_index] >> bit_index) & 1u;
    bits_[word_index] |= (1u << bit_index);
    return ret;
  }

  const uint32_t* data() const {
    return bits_.get();
  }

  void reset() {
    memset(bits_.get(), 0, len_*sizeof(uint32_t));
  }

private:
  size_t len_{0};
  std::unique_ptr<uint32_t[]> bits_;
}; // RairBitmap

template <MetricType metric, class C, bool omit_redund_dis>
struct ReIVFFlatScanner : InvertedListScanner {
  size_t d;
  mutable RairBitmap id_checked;

  ReIVFFlatScanner(size_t d, bool store_pairs, size_t ntotal)
      : InvertedListScanner(store_pairs), d(d) {
    keep_max = is_similarity_metric(metric);
    if (omit_redund_dis) {
      FAISS_ASSERT(!store_pairs);
      id_checked = RairBitmap(ntotal);
    }
  }

  const float* xi;
  void set_query(const float* query) override {
    this->xi = query;
    if (omit_redund_dis) {
      id_checked.reset();
    }
  }

  idx_t list_no;
  void set_list(idx_t list_no_, float /* coarse_dis */) override {
    this->list_no = list_no_;
  }

  float distance_to_code(const uint8_t* code) const override {
    const float* yj = (float*)code;
    float dis = metric == METRIC_INNER_PRODUCT
                ? fvec_inner_product(xi, yj, d)
                : fvec_L2sqr(xi, yj, d);
    return dis;
  }

  size_t scan_codes(
      size_t list_size, const uint8_t* codes, const idx_t* ids,
      float* simi, idx_t* idxi, size_t k
      ) const override {
    const float* list_vecs = (const float*)codes;
    size_t nup = 0;
    size_t local_omit_dis = 0;
    for (size_t j = 0; j < list_size; j ++) {
      const float* yj = list_vecs + d * j;
      if (omit_redund_dis) {
        /// TODO: bottleneck: 10% time cost here
        /// explicit prefetch seems useless
//#ifdef __GNUC__
//        if (likely(j + 4 < list_size)) {
//          __builtin_prefetch(&(id_checked.data()[ids[j+4]>>5]), 1, 3);
//        }
//#endif
        bool checked = id_checked.get_and_set_true(ids[j]);
        if (checked) {
          local_omit_dis ++;
          continue;
        }
      }
      float dis = metric == METRIC_INNER_PRODUCT
                  ? fvec_inner_product(xi, yj, d)
                  : fvec_L2sqr(xi, yj, d);
      if (C::cmp(simi[0], dis)) {
        int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
        heap_replace_top<C>(k, simi, idxi, dis, id);
        nup ++;
      }
    }
    omit_ndis += local_omit_dis;
    return nup;
  }

  void scan_codes_range(
      size_t list_size, const uint8_t* codes, const idx_t* ids,
      float radius, RangeQueryResult& res
      ) const override {
    auto* list_vecs = (const float*)codes;
    for (size_t j = 0; j < list_size; j++) {
      const float* yj = list_vecs + d * j;
      int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
      float dis = metric == METRIC_INNER_PRODUCT
                  ? fvec_inner_product(xi, yj, d)
                  : fvec_L2sqr(xi, yj, d);
      if (C::cmp(radius, dis)) {
        res.add(dis, id);
      }
    }
  }
}; // ReIVFFlatScanner

} // anonymous namespace

InvertedListScanner* IndexReIVFFlat::get_InvertedListScanner(
    bool store_pairs, const IDSelector* sel) const {
  FAISS_ASSERT_MSG(!sel, "IDSelector not supported");
  FAISS_ASSERT(parallel_mode == 0);
  if (metric_type == METRIC_INNER_PRODUCT) {
    if (omit_redund_visit) {
      return new ReIVFFlatScanner<METRIC_INNER_PRODUCT, CMin<float, int64_t>,
          true>(d, store_pairs, ntotal);
    } else {
      return new ReIVFFlatScanner<METRIC_INNER_PRODUCT, CMin<float, int64_t>,
          false>(d, store_pairs, ntotal);
    }
  } else if (metric_type == METRIC_L2) {
    if (omit_redund_visit) {
      return new ReIVFFlatScanner<METRIC_L2, CMax<float, int64_t>, true>(
          d, store_pairs, ntotal);
    } else {
      return new ReIVFFlatScanner<METRIC_L2, CMax<float, int64_t>, false>(
          d, store_pairs, ntotal);
    }
  } else {
    FAISS_THROW_MSG("metric type not supported");
  }
}