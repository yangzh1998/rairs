#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <faiss/impl/CodePacker.h>

/** PQ4 SIMD packing and accumulation functions with deduplication
 * See faiss/impl/pq4_fast_scan.h
 */

namespace faiss {

struct NormTableScaler;
struct SIMDResultHandler;

namespace rair {

/** Run accumulation loop in qbs without block pruning.
 *
 * @param qbs     4-bit encoded number of queries
 * @param nb      number of database codes (mutliple of bbs)
 * @param nsq     number of sub-quantizers
 * @param codes   encoded database vectors (packed)
 * @param LUT     look-up table (packed)
 * @param res     call-back for the results
 * @param scaler  scaler to scale the encoded norm
 */
template <bool dedup>
void dedup_pq4_accumulate_loop_qbs(
    int qbs,
    size_t nb,
    int nsq,
    const uint8_t* codes,
    const uint8_t* LUT,
    SIMDResultHandler& res,
    const NormTableScaler* scaler = nullptr);

/** Run accumulation loop query by query with block pruning.
 *
 * @param nq      number of queries
 * @param nb      number of database codes (mutliple of bbs)
 * @param nsq     number of sub-quantizers
 * @param codes   encoded database vectors (packed)
 * @param colors  the color of blocks in this list (-1 if a sundries block)
 * @param LUT     look-up table (packed)
 * @param res     call-back for the results
 * @param scaler  scaler to scale the encoded norm
 *
 * @return the number of pruned pair(query, block)
 */
size_t dedup_pq4_accumulate_loop_1_with_block_pruning(
    int nq,
    size_t nb,
    int nsq,
    const uint8_t* codes,
    const int32_t* colors,
    const uint8_t* LUT,
    SIMDResultHandler& res,
    const NormTableScaler* scaler = nullptr);

} // namespace faiss::rair

} // namespace faiss

