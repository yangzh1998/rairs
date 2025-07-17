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
 * It accumulates results for 1 query and 2 * 16 database elements
 * writes results in a ResultHandler
 */
template <class ResultHandler, class Scaler>
void dedup_kernel_accumulate_block_1(
    int nsq,
    const uint8_t* codes,
    const uint8_t* LUT,
    ResultHandler& res,
    const Scaler& scaler) {

  simd16uint16 accu[4];
  accu[0].clear(); accu[1].clear(); accu[2].clear(); accu[3].clear();

  for (int sq = 0; sq < nsq - scaler.nscale; sq += 2) {
    simd32uint8 c(codes);
    codes += 32;

    simd32uint8 mask(0xf);
    simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
    simd32uint8 clo = c & mask;

    simd32uint8 lut(LUT);
    LUT += 32;

    // v1g1, v9g1, v2g1, v10g1,  ..., v8g1, v16g1, v1g2, v9g2, ..., v8g2, v16g2
    simd32uint8 res0 = lut.lookup_2_lanes(clo);
    simd32uint8 res1 = lut.lookup_2_lanes(chi);

    accu[0] += simd16uint16(res0); // v1g1, ..., v8g1, v1g2, ..., v8g2
    accu[1] += simd16uint16(res0) >> 8; // v9g1, ..., v16g1, v9g2, ..., v16g2

    accu[2] += simd16uint16(res1);
    accu[3] += simd16uint16(res1) >> 8;
  }

  for (int sq = 0; sq < scaler.nscale; sq += 2) {
    simd32uint8 c(codes);
    codes += 32;

    simd32uint8 mask(0xf);
    // shift op does not exist for int8...
    simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
    simd32uint8 clo = c & mask;

    simd32uint8 lut(LUT);
    LUT += 32;

    simd32uint8 res0 = scaler.lookup(lut, clo);
    accu[0] += scaler.scale_lo(res0); // handle vectors 0..7
    accu[1] += scaler.scale_hi(res0); // handle vectors 8..15

    simd32uint8 res1 = scaler.lookup(lut, chi);
    accu[2] += scaler.scale_lo(res1); // handle vectors 16..23
    accu[3] += scaler.scale_hi(res1); // handle vectors 24..31
  }

  accu[0] -= accu[1] << 8;
  simd16uint16 dis0 = combine2x2(accu[0], accu[1]);
  accu[2] -= accu[3] << 8;
  simd16uint16 dis1 = combine2x2(accu[2], accu[3]);
  res.handle(0, 0, dis0, dis1);
}

template <int NQ, class DedupResultHandler, class Scaler>
void dedup_accumulate_q_1_bp(
    size_t ntotal2,
    int nsq,
    const uint8_t* codes,
    const int32_t* colors,
    const uint8_t* LUT0,
    DedupResultHandler& res,
    const Scaler& scaler,
    size_t& nprune) {

  for (size_t j0 = 0; j0 < ntotal2; j0 += 32) { ///< treat each block
    FixedStorageHandler<NQ, 2> res2;
    const uint8_t* LUT = LUT0;
    nprune += NQ;

    if constexpr (NQ <= 12) {
      bool visited0, visited1, visited2, visited3, visited4, visited5;

#define CHECK_VISITED(QID, VARID) \
      if constexpr (NQ > (QID)) visited##VARID = res.list_visited(QID, *colors);
#define VISIT(QID, VARID) \
      if constexpr (NQ > (QID)) { \
        if (!visited##VARID) { \
          res2.set_block_origin(QID, 0); \
          dedup_kernel_accumulate_block_1(nsq, codes, LUT, res2, scaler); \
          nprune --; \
        } \
        LUT += nsq * 16; \
      }

      CHECK_VISITED(0, 0); CHECK_VISITED(1, 1); CHECK_VISITED(2, 2);
      CHECK_VISITED(3, 3); CHECK_VISITED(4, 4); CHECK_VISITED(5, 5);
      VISIT(0, 0); VISIT(1, 1); VISIT(2, 2);
      VISIT(3, 3); VISIT(4, 4); VISIT(5, 5);
      CHECK_VISITED(6, 0); CHECK_VISITED(7, 1); CHECK_VISITED(8, 2);
      CHECK_VISITED(9, 3); CHECK_VISITED(10, 4); CHECK_VISITED(11, 5);
      VISIT(6, 0); VISIT(7, 1); VISIT(8, 2);
      VISIT(9, 3); VISIT(10, 4); VISIT(11, 5);

#undef CHECK_VISITED
#undef VISIT

    } else { ///< NQ > 12
      for (size_t q = 0; q < NQ; q ++) {
        if (!res.list_visited(q, *colors)) {
          res2.set_block_origin(q, 0);
          dedup_kernel_accumulate_block_1(nsq, codes, LUT, res2, scaler);
          nprune --;
        }
        LUT += nsq * 16;
      }
    }

    res.set_block_origin(0, j0);
    res2.to_other_handler(res);
    codes += 32 * nsq / 2;
    colors ++;
  }
}

template <class DedupResultHandler, class Scaler>
void dedup_pq4_accumulate_loop_1_with_bp_fixed_scaler(
    int nq, size_t ntotal2, int nsq, const uint8_t* codes,
    const int32_t* colors, const uint8_t* LUT0, DedupResultHandler& res,
    const Scaler& scaler, size_t& nprune) {
  assert(nsq % 2 == 0);
  assert(is_aligned_pointer(codes));
  assert(is_aligned_pointer(LUT0));
  switch (nq) {
#define DISPATCH(NQ) \
    case NQ: \
      dedup_accumulate_q_1_bp<NQ>( \
          ntotal2, nsq, codes, colors, LUT0, res, scaler, nprune); \
      return;
    DISPATCH(1); DISPATCH(2); DISPATCH(3); DISPATCH(4); DISPATCH(5);
    DISPATCH(6); DISPATCH(7); DISPATCH(8); DISPATCH(9); DISPATCH(10);
    DISPATCH(11); DISPATCH(12);
#undef DISPATCH
  }
}

struct Run_dedup_pq4_accumulate_loop_1_with_bp {
  template <class DedupResultHandler>
  void f(DedupResultHandler& res, int nq, size_t nb, int nsq,
         const uint8_t* codes, const int32_t* colors, const uint8_t* LUT,
         const NormTableScaler* scaler, size_t* nprune) {
    if (scaler) {
      dedup_pq4_accumulate_loop_1_with_bp_fixed_scaler(
          nq, nb, nsq, codes, colors, LUT, res, *scaler, *nprune);
    } else {
      DummyScaler dummy;
      dedup_pq4_accumulate_loop_1_with_bp_fixed_scaler(
          nq, nb, nsq, codes, colors, LUT, res, dummy, *nprune);
    }
  }
}; // Run_dedup_pq4_accumulate_loop_1_with_bp

} // anonymous namespace

size_t faiss::rair::dedup_pq4_accumulate_loop_1_with_block_pruning(
    int nq,
    size_t nb,
    int nsq,
    const uint8_t* codes,
    const int32_t* colors,
    const uint8_t* LUT,
    SIMDResultHandler& res,
    const NormTableScaler* scaler) {
  size_t nprune = 0;
  Run_dedup_pq4_accumulate_loop_1_with_bp consumer;
  dispatch_DedupSIMDResultHanlder(
      res, consumer, nq, nb, nsq, codes, colors, LUT, scaler, &nprune
      );
  return nprune;
}