#include <faiss/rair/impl/dedup_pq4_fast_scan.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LookupTableScaler.h>
#include <faiss/rair/impl/dedup_simd_result_handlers.h>
#include <faiss/utils/simdlib.h>

using namespace faiss;
using namespace faiss::simd_result_handlers;
using namespace faiss::rair;
using namespace faiss::simd_result_handlers::rair;

/************************************************************
 * Accumulation functions
 ************************************************************/

namespace {

/**
 * The computation kernel
 * It accumulates results for NQ queries and 2 * 16 database elements
 * writes results in a ResultHandler
 */
template <int NQ, class ResultHandler, class Scaler>
void dedup_kernel_accumulate_block(
    int nsq,
    const uint8_t* codes,
    const uint8_t* LUT,
    ResultHandler& res,
    const Scaler& scaler) {
  // dummy alloc to keep the windows compiler happy
  constexpr int NQA = NQ > 0 ? NQ : 1;
  // distance accumulators
  // layout: accu[q][b]: distance accumulator for vectors 8*b..8*b+7
  // in one accu: 1 query, odd_group_accu and even_group_accu, 8 vecs
  simd16uint16 accu[NQA][4];

  for (int q = 0; q < NQ; q ++) {
    for (int b = 0; b < 4; b ++) {
      accu[q][b].clear();
    }
  }

  // _mm_prefetch(codes + 768, 0);
  for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) { // each 2 groups
    // prefetch
    // 32 vecs, 2 vec_group, 4bit per group.
    // v1g1, v17g1, v9g1, v25g1, ..., (g2...)
    simd32uint8 c(codes);
    codes += 32;

    simd32uint8 mask(0xf);
    // shift op does not exist for int8...
    simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
    // 4 low-bit in each byte.
    // v1g1, v9g1, v2g1, v10g1,  ..., v8g1, v16g1, v1g2, v9g2, ..., v8g2, v16g2
    simd32uint8 clo = c & mask;

    for (int q = 0; q < NQ; q ++) {
      // load LUTs for 2 quantizers
      // 2 group, 16 centroid, 8bit: g1c1, g1c2, ..., g2c1, g2c2, ..., g2c16
      simd32uint8 lut(LUT);
      LUT += 32;

      // v1g1, v9g1, v2g1, v10g1,  ..., v8g1, v16g1, v1g2, v9g2, ..., v8g2, v16g2
      simd32uint8 res0 = lut.lookup_2_lanes(clo);
      simd32uint8 res1 = lut.lookup_2_lanes(chi);

      accu[q][0] += simd16uint16(res0); // v1g1, ..., v8g1, v1g2, ..., v8g2
      accu[q][1] += simd16uint16(res0) >> 8; // v9g1, ..., v16g1, v9g2, ..., v16g2

      accu[q][2] += simd16uint16(res1);
      accu[q][3] += simd16uint16(res1) >> 8;
    }
  }

  for (int sq = 0; sq < scaler.nscale; sq += 2) {
    // prefetch
    simd32uint8 c(codes);
    codes += 32;

    simd32uint8 mask(0xf);
    // shift op does not exist for int8...
    simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
    simd32uint8 clo = c & mask;

    for (int q = 0; q < NQ; q ++) {
      // load LUTs for 2 quantizers
      simd32uint8 lut(LUT);
      LUT += 32;

      simd32uint8 res0 = scaler.lookup(lut, clo);
      accu[q][0] += scaler.scale_lo(res0); // handle vectors 0..7
      accu[q][1] += scaler.scale_hi(res0); // handle vectors 8..15

      simd32uint8 res1 = scaler.lookup(lut, chi);
      accu[q][2] += scaler.scale_lo(res1); // handle vectors 16..23
      accu[q][3] += scaler.scale_hi(res1); // handle vectors 24..31
    }
  }

  for (int q = 0; q < NQ; q ++) {
    accu[q][0] -= accu[q][1] << 8;
    simd16uint16 dis0 = combine2x2(accu[q][0], accu[q][1]);
    accu[q][2] -= accu[q][3] << 8;
    simd16uint16 dis1 = combine2x2(accu[q][2], accu[q][3]);
    res.handle(q, 0, dis0, dis1);
  }
}

/// handle at most 4 blocks of queries
template <int QBS, class DedupResultHandler, class Scaler>
void dedup_accumulate_q_4step(
    size_t ntotal2,
    int nsq,
    const uint8_t* codes,
    const uint8_t* LUT0,
    DedupResultHandler& res,
    const Scaler& scaler) {
  constexpr int Q1 = QBS & 15;
  constexpr int Q2 = (QBS >> 4) & 15;
  constexpr int Q3 = (QBS >> 8) & 15;
  constexpr int Q4 = (QBS >> 12) & 15;
  constexpr int SQ = Q1 + Q2 + Q3 + Q4;

  for (size_t j0 = 0; j0 < ntotal2; j0 += 32) { ///< treat each block
    FixedStorageHandler<SQ, 2> res2;
    const uint8_t* LUT = LUT0;
    if constexpr (Q1 > 0) {
      dedup_kernel_accumulate_block<Q1>(nsq, codes, LUT, res2, scaler);
      LUT += Q1 * nsq * 16;
    }
    if constexpr (Q2 > 0) {
      res2.set_block_origin(Q1, 0);
      dedup_kernel_accumulate_block<Q2>(nsq, codes, LUT, res2, scaler);
      LUT += Q2 * nsq * 16;
    }
    if constexpr (Q3 > 0) {
      res2.set_block_origin(Q1 + Q2, 0);
      dedup_kernel_accumulate_block<Q3>(nsq, codes, LUT, res2, scaler);
      LUT += Q3 * nsq * 16;
    }
    if constexpr (Q4 > 0) {
      res2.set_block_origin(Q1 + Q2 + Q3, 0);
      dedup_kernel_accumulate_block<Q4>(nsq, codes, LUT, res2, scaler);
    }
    res.set_block_origin(0, j0);
    res2.to_other_handler(res);
    codes += 32 * nsq / 2;
  } ///< end of this block
}

template <class DedupResultHandler, class Scaler>
void dedup_pq4_accumulate_loop_qbs_fixed_scaler(
    int qbs, size_t ntotal2, int nsq, const uint8_t* codes, const uint8_t* LUT0,
    DedupResultHandler& res, const Scaler& scaler) {
  assert(nsq % 2 == 0);
  assert(is_aligned_pointer(codes));
  assert(is_aligned_pointer(LUT0));

  // try out optimized versions
  switch (qbs) {
#define DISPATCH(QBS) \
    case QBS: \
      dedup_accumulate_q_4step<QBS>( \
          ntotal2, nsq, codes, LUT0, res, scaler); \
      return;
    DISPATCH(0x3333); DISPATCH(0x2333); DISPATCH(0x2233); DISPATCH(0x333);
    DISPATCH(0x2223); DISPATCH(0x233); DISPATCH(0x1223); DISPATCH(0x223);
    DISPATCH(0x34); DISPATCH(0x133); DISPATCH(0x6); DISPATCH(0x33);
    DISPATCH(0x123); DISPATCH(0x222); DISPATCH(0x23); DISPATCH(0x5);
    DISPATCH(0x13); DISPATCH(0x22); DISPATCH(0x4); DISPATCH(0x3);
    DISPATCH(0x21); DISPATCH(0x2); DISPATCH(0x1);
#undef DISPATCH
  }

  FAISS_ASSERT_MSG(false, "illegal QBS");
}

template <bool dedup>
struct Run_dedup_pq4_accumulate_loop_qbs {
  template <class DedupResultHandler>
  void f(DedupResultHandler& res, int qbs, size_t nb, int nsq,
         const uint8_t* codes, const uint8_t* LUT,
         const NormTableScaler* scaler) {
    res.dedup = dedup;
    if (scaler) {
      dedup_pq4_accumulate_loop_qbs_fixed_scaler(
          qbs, nb, nsq, codes, LUT, res, *scaler);
    } else {
      DummyScaler dummy;
      dedup_pq4_accumulate_loop_qbs_fixed_scaler(
          qbs, nb, nsq, codes, LUT, res, dummy);
    }
  }
}; // Run_dedup_pq4_accumulate_loop_qbs

} // anonymous namespace

// Public functions

template <bool dedup>
void faiss::rair::dedup_pq4_accumulate_loop_qbs(
    int qbs,
    size_t nb,
    int nsq,
    const uint8_t* codes,
    const uint8_t* LUT,
    SIMDResultHandler& res,
    const NormTableScaler* scaler) {
  Run_dedup_pq4_accumulate_loop_qbs<dedup> consumer;
  dispatch_DedupSIMDResultHanlder(
      res, consumer, qbs, nb, nsq, codes, LUT, scaler
      );
}

template void faiss::rair::dedup_pq4_accumulate_loop_qbs<true>(
    int qbs, size_t nb, int nsq, const uint8_t* codes, const uint8_t* LUT,
    SIMDResultHandler& res, const NormTableScaler* scaler
    );

template void faiss::rair::dedup_pq4_accumulate_loop_qbs<false>(
    int qbs, size_t nb, int nsq, const uint8_t* codes, const uint8_t* LUT,
    SIMDResultHandler& res, const NormTableScaler* scaler
);