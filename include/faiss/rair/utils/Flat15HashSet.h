#ifndef FAISS_RAIR_UTILS_FLAT15HASHSET_H
#define FAISS_RAIR_UTILS_FLAT15HASHSET_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace faiss::rair {

/**
 * @tparam _NC Number of Chunks, should be 2^k
 */
template <size_t _NC>
class Flat15HashSetForI32 {
public:
  struct alignas(64) chunk_t {
    uint16_t valid{0};
    uint16_t no_use;
    int32_t ele[15];
  }; // 64B

  static constexpr size_t MASK = _NC - 1;

  // Methods

  Flat15HashSetForI32() = default;

  inline void reset() {
    for (size_t i = 0; i < _NC; i ++) {
      _chunks[i].valid = 0;
    }
  }

private:
  inline uint16_t _eq_mask(int32_t* ele, int32_t key, uint16_t valid) {
    /// TODO: avx2
#ifdef __AVX512F__
    __m512i key_vec = _mm512_set1_epi32(key);
    __m512i ele_vec = _mm512_load_si512(reinterpret_cast<const void*>(ele));
    __mmask16 cmp_mask = _mm512_mask_cmpeq_epi32_mask(valid, ele_vec, key_vec);
    return cmp_mask;
#else
    uint16_t mask = 0;
    for (size_t i = 0; i < 16; i ++) {
      if (ele[i] == key) {
        mask |= (1 << i);
      }
    }
    mask &= valid;
    return mask;
#endif
  }

public:
  inline void insert(int32_t key) {
    find_and_insert(key);
  }

  /**
   * @return if key already exists
   */
  inline bool find_and_insert(int32_t key) {
    size_t idx = key & MASK;
    chunk_t& chunk = _chunks[idx];
    int32_t* ele = (int32_t*)(&chunk);
    uint16_t mask = _eq_mask(ele, key, chunk.valid);
    if (!mask) {
      //assert(chunk.valid == 0xfffe); /// TODO: deal with this situation
      int pos = __builtin_ctz(~(chunk.valid | 1));
      ele[pos] = key;
      chunk.valid |= (1 << pos);
    }
    return mask;
  }

  inline bool find(int32_t key) {
    size_t idx = key & MASK;
    chunk_t& chunk = _chunks[idx];
    int32_t* ele = (int32_t*)(&chunk);
    uint16_t mask = _eq_mask(ele, key, chunk.valid);
    return mask;
  }

  inline void erase(int32_t key) {
    /// TODO: not tested
    size_t idx = key & MASK;
    chunk_t& chunk = _chunks[idx];
    int32_t* ele = (int32_t*)(&chunk);
    uint16_t mask = _eq_mask(ele, key, chunk.valid);
    if (mask) {
      int pos = __builtin_ctz(mask);
      chunk.valid &= ~(1 << pos);
    }
  }

private:
  alignas(64) chunk_t _chunks[_NC];
}; // Flat15HashSetForI32

} // namespace faiss::rair


#endif // FAISS_RAIR_UTILS_FLAT15HASHSET_H
