#ifndef FAISS_RAIR_UTILS_BITMAP_H
#define FAISS_RAIR_UTILS_BITMAP_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace faiss::rair {

/**
 * @tparam _Nb num of bits, a multiple of 512
 */
template<uint32_t _Nb>
class alignas(64) bitmap {
public:
  bitmap() = default;

  inline void set_true(size_t index) {
    _bits[index >> 5] |= (1u << (index & 31));
  }

  inline void set_false(size_t index) {
    _bits[index >> 5] &= ~(1u << (index & 31));
  }

  inline bool get(size_t index) const {
    return (_bits[index >> 5] >> (index & 31)) & 1u;
  }

  inline void reset() {
#ifdef __AVX512F__
    __m512i zero512 = _mm512_setzero_si512();
    for (size_t i = 0; i < _Nb>>5; i += 16) {
      _mm512_store_si512(_bits+i, zero512);
    }
#else
    memset(_bits, 0, _Nb >> 3);
#endif
  }

  inline void prefetch() const {
#if defined(__GNUC__)
    if constexpr (_Nb == 1024) {
      __builtin_prefetch(_bits, 0 /* read-only */, 3 /* high locality */);
      __builtin_prefetch(_bits+16, 0 /* read-only */, 3 /* high locality */);
    }
#endif
  }

private:
  uint32_t _bits[_Nb >> 5];
}; // bitmap

} // namespace faiss::rair

#endif // FAISS_RAIR_UTILS_BITMAP_H
