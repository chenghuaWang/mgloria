#ifndef _MGLORIA___VEC_SSE_HPP_
#define _MGLORIA___VEC_SSE_HPP_

#include <emmintrin.h>
#include "./vectorization/__vec_prepare.hpp"

namespace mgloria {
namespace vectorization {

template<>
struct Vectorized<float, VecArch::SSE_Arch> {
  // float in vector
  static const index_t num = 4;
  // friends
  friend MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator+(
      const Vectorized<float, VecArch::SSE_Arch>& lhs,
      const Vectorized<float, VecArch::SSE_Arch>& rhs);

  friend MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator-(
      const Vectorized<float, VecArch::SSE_Arch>& lhs,
      const Vectorized<float, VecArch::SSE_Arch>& rhs);

  friend MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator*(
      const Vectorized<float, VecArch::SSE_Arch>& lhs,
      const Vectorized<float, VecArch::SSE_Arch>& rhs);

  friend MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator/(
      const Vectorized<float, VecArch::SSE_Arch>& lhs,
      const Vectorized<float, VecArch::SSE_Arch>& rhs);

  // constructor
  Vectorized() = default;
  explicit Vectorized(__m128 data) : m_data(data) {}

  // static create Vectorized functions
  MGLORIA_INLINE_CPU static Vectorized<float, VecArch::SSE_Arch> Fill(float s) {
    return Vectorized<float, VecArch::SSE_Arch>(_mm_set1_ps(s));
  }

  MGLORIA_INLINE_CPU static Vectorized<float, VecArch::SSE_Arch> Load(const float* s) {
    return Vectorized<float, VecArch::SSE_Arch>(_mm_load_ps(s));
  }

  MGLORIA_INLINE_CPU static Vectorized<float, VecArch::SSE_Arch> LoadUnAligned(const float* s) {
    return Vectorized<float, VecArch::SSE_Arch>(_mm_loadu_ps(s));
  }

  // operator overload.
  MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch>& operator=(float s) {
    m_data = _mm_set1_ps(s);
    return *this;
  }

  MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch>& operator=(const float* s) {
    m_data = _mm_load_ps(s);
    return *this;
  }

  MGLORIA_INLINE_CPU float Sum() const {
    __m128 ans = _mm_add_ps(m_data, _mm_movehl_ps(m_data, m_data));
    __m128 rst = _mm_add_ss(ans, _mm_shuffle_ps(ans, ans, 1));

    float rr = _mm_cvtss_f32(rst);
    return rr;
  }

  // Store vectorized data to normal data.
  MGLORIA_INLINE_CPU void Store(float* data) const { _mm_store_ps(data, m_data); }
  MGLORIA_INLINE_CPU void StoreEach(float* data) const { _mm_store1_ps(data, m_data); }

 private:
  // parameters
  __m128 m_data;
};

template<>
struct Vectorized<double, VecArch::SSE_Arch> {
  // float in vector
  static const index_t num = 2;
  // friends
  friend MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator+(
      const Vectorized<double, VecArch::SSE_Arch>& lhs,
      const Vectorized<double, VecArch::SSE_Arch>& rhs);

  friend MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator-(
      const Vectorized<double, VecArch::SSE_Arch>& lhs,
      const Vectorized<double, VecArch::SSE_Arch>& rhs);

  friend MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator*(
      const Vectorized<double, VecArch::SSE_Arch>& lhs,
      const Vectorized<double, VecArch::SSE_Arch>& rhs);

  friend MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator/(
      const Vectorized<double, VecArch::SSE_Arch>& lhs,
      const Vectorized<double, VecArch::SSE_Arch>& rhs);

  // constructor
  Vectorized() = default;
  explicit Vectorized(__m128 data) : m_data(data) {}

  // static create Vectorized functions
  MGLORIA_INLINE_CPU static Vectorized<double, VecArch::SSE_Arch> Fill(double s) {
    return Vectorized<double, VecArch::SSE_Arch>(_mm_set1_pd(s));
  }

  MGLORIA_INLINE_CPU static Vectorized<double, VecArch::SSE_Arch> Load(const double* s) {
    return Vectorized<double, VecArch::SSE_Arch>(_mm_load1_pd(s));
  }

  MGLORIA_INLINE_CPU static Vectorized<double, VecArch::SSE_Arch> LoadUnAligned(const double* s) {
    return Vectorized<double, VecArch::SSE_Arch>(_mm_loadu_pd(s));
  }

  // operator overload.
  MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch>& operator=(double s) {
    m_data = _mm_set1_pd(s);
    return *this;
  }

  MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch>& operator=(const double* s) {
    m_data = _mm_load_pd(s);
    return *this;
  }

  MGLORIA_INLINE_CPU double Sum(void) const {
    __m128d tmp = _mm_add_sd(m_data, _mm_unpackhi_pd(m_data, m_data));
    double ans = _mm_cvtsd_f64(tmp);
    return ans;
  }

  // Store vectorized data to normal data.
  MGLORIA_INLINE_CPU void Store(double* data) const { _mm_store_pd(data, m_data); }
  MGLORIA_INLINE_CPU void StoreEach(double* data) const { _mm_store1_pd(data, m_data); }

 private:
  // parameters
  __m128 m_data;
};

MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator+(
    const Vectorized<float, VecArch::SSE_Arch>& lhs,
    const Vectorized<float, VecArch::SSE_Arch>& rhs) {
  return Vectorized<float, VecArch::SSE_Arch>(_mm_add_ps(lhs.m_data, rhs.m_data));
}

MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator-(
    const Vectorized<float, VecArch::SSE_Arch>& lhs,
    const Vectorized<float, VecArch::SSE_Arch>& rhs) {
  return Vectorized<float, VecArch::SSE_Arch>(_mm_sub_ps(lhs.m_data, rhs.m_data));
}

MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator*(
    const Vectorized<float, VecArch::SSE_Arch>& lhs,
    const Vectorized<float, VecArch::SSE_Arch>& rhs) {
  return Vectorized<float, VecArch::SSE_Arch>(_mm_mul_ps(lhs.m_data, rhs.m_data));
}

MGLORIA_INLINE_CPU Vectorized<float, VecArch::SSE_Arch> operator/(
    const Vectorized<float, VecArch::SSE_Arch>& lhs,
    const Vectorized<float, VecArch::SSE_Arch>& rhs) {
  return Vectorized<float, VecArch::SSE_Arch>(_mm_div_ps(lhs.m_data, rhs.m_data));
}

MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator+(
    const Vectorized<double, VecArch::SSE_Arch>& lhs,
    const Vectorized<double, VecArch::SSE_Arch>& rhs) {
  return Vectorized<double, VecArch::SSE_Arch>(_mm_add_ps(lhs.m_data, rhs.m_data));
}

MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator-(
    const Vectorized<double, VecArch::SSE_Arch>& lhs,
    const Vectorized<double, VecArch::SSE_Arch>& rhs) {
  return Vectorized<double, VecArch::SSE_Arch>(_mm_sub_ps(lhs.m_data, rhs.m_data));
}

MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator*(
    const Vectorized<double, VecArch::SSE_Arch>& lhs,
    const Vectorized<double, VecArch::SSE_Arch>& rhs) {
  return Vectorized<double, VecArch::SSE_Arch>(_mm_mul_ps(lhs.m_data, rhs.m_data));
}

MGLORIA_INLINE_CPU Vectorized<double, VecArch::SSE_Arch> operator/(
    const Vectorized<double, VecArch::SSE_Arch>& lhs,
    const Vectorized<double, VecArch::SSE_Arch>& rhs) {
  return Vectorized<double, VecArch::SSE_Arch>(_mm_div_ps(lhs.m_data, rhs.m_data));
}

}  // namespace vectorization
}  // namespace mgloria

#endif