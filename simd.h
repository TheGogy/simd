/**
*   @file simd.h
*   @brief This file contains a simple wrapper around __m256d for SIMD operations on four-element vectors.
*
*   @info REQUIRED FLAGS: -std=c++20 -mfma -msse4.1
*
*   This class supports:
*   - arithmetic operations (+, -, *, /)
*   - bitwise operations    (&, |, ^, ~)
*   - square root, reciprocal square root, reciprocal
*   - dot product, cross product
*   - sum, hmax, hmin
*   - horizontal addition / subtraction
*   - element wise minimum / maximum
*   - element wise greater than / less than
*   - shuffling
*
*   Elements can be accessed as arrays or individual floats.
*/

#pragma once


#include <array>
#include <cassert>
#include <immintrin.h>
#include <limits>
#include <numbers>
#include <ostream>


/**
* @class Simd4
* @brief Simple wrapper class around __m128.
*/
class Simd4
{
public:
    __m128 data;


    /**
    * Constructors.
    */
    explicit Simd4() : data(_mm_setzero_ps()) {}
    explicit Simd4(__m128 v) : data(v) {}
    explicit Simd4(float a)  : data(_mm_set1_ps(a)) {}
    explicit Simd4(float x, float y, float z, float w) : data(_mm_set_ps(w, z, y, x)) {}


    /**
    * Special cases.
    */
    static Simd4 one()              { return Simd4(1.0); }
    static Simd4 zero()             { return Simd4(_mm_setzero_ps()); }
    static Simd4 minus_one()        { return Simd4(-1.0); }
    static Simd4 half()             { return Simd4(0.5f); }
    static Simd4 adj_mask()         { return Simd4(_mm_setr_ps(1.0, -1.0, -1.0, 1.0)); }
    static Simd4 sign_mask()        { return Simd4(_mm_castsi128_ps(_mm_set1_epi32(0x80000000))); }
    static Simd4 all_bits_set()     { return Simd4(_mm_castsi128_ps(_mm_set1_epi32(-1))); }
    static Simd4 abs_mask()         { return Simd4(_mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))); }
    static Simd4 infinity()         { return Simd4(std::numeric_limits<float>::infinity()); }
    static Simd4 epsilon()          { return Simd4(std::numeric_limits<float>::epsilon()); }
    static Simd4 max()              { return Simd4(std::numeric_limits<float>::max()); }
    static Simd4 min()              { return Simd4(std::numeric_limits<float>::min()); }
    static Simd4 nan()              { return Simd4(std::numeric_limits<float>::quiet_NaN()); }
    static Simd4 negative_zero()    { return Simd4(-0); }
    static Simd4 pi()               { return Simd4(std::numbers::pi); }
    static Simd4 inv_pi()           { return Simd4(std::numbers::inv_pi); }
    static Simd4 half_pi()          { return Simd4(1.57079632679489661923); }
    static Simd4 quarter_pi()       { return Simd4(0.78539816339744830962); }
    static Simd4 two_pi()           { return Simd4(6.28318530717958647692); }

    static Simd4 unit_x()           { return Simd4(1.0, 0.0, 0.0, 0.0); }
    static Simd4 unit_y()           { return Simd4(0.0, 1.0, 0.0, 0.0); }
    static Simd4 unit_z()           { return Simd4(0.0, 0.0, 1.0, 0.0); }
    static Simd4 unit_w()           { return Simd4(0.0, 0.0, 1.0, 1.0); }

    static Simd4 reflection()       { return Simd4(-1.0, 1.0, 0.0, 0.0); }


    /**
    * Computes the absolute value of the Simd4
    */
    Simd4 abs() const noexcept
    {
        return *this & abs_mask();
    }

    /**
    * Sets the item at the given index to the given value.
    *
    * @tparam Index The index
    * @param val The value
    */
    template <int Index>
    void set(float val_)
    {
        static_assert(Index >= 0 && Index < 4, "Simd4::set(): Index out of bounds");
        data = _mm_insert_ps(data, _mm_set_ss(val_), Index << 4);
    }


    /**
    * Sets the item at the given index to the given value.
    *
    * @tparam Index The index
    * @param val The value
    */
    template <int Index>
    Simd4 with_set(float val_) const noexcept
    {
        static_assert(Index >= 0 && Index < 4, "Simd4::with_set(): Index out of bounds");
        return Simd4(_mm_insert_ps(data, _mm_set_ss(val_), Index << 4));
    }


    /**
    * @brief Convert scalar single.
    *
    * Extracts the first element in the array to a single floating point value.
    * Simd4(1.0, 2.0, 3.0, 4.0).cvtss() -> 1.0
    *
    * @return The first element in the array.
    */
    float cvtss() const noexcept
    {
        return _mm_cvtss_f32(data);
    }


    /**
    * @brief Calculates the element-wise square root of the vector.
    *
    * Simd4(1.0, 4.0, 9.0, 16.0).sqrt() -> Simd4(1.0, 2.0, 3.0, 4.0)
    *
    * @return A Simd4 array containing the element wise square root.
    */
    Simd4 sqrt() const noexcept
    {
        return Simd4(_mm_sqrt_ps(data));
    }


    /**
    * @brief Calculates the element-wise reciprocal square root of the vector.
    *
    * Simd4(1.0, 4.0, 16.0, 25.0).rsqrt() -> Simd4(1.0, 0.5, 0.25, 0.2)
    *
    * @return A Simd4 array containing the element wise reciprocal square root.
    */
    Simd4 rsqrt() const noexcept
    {
        return Simd4(_mm_rsqrt_ps(data));
    }


    /**
    * @brief Calculates the element-wise reciprocal of the vector.
    *
    * Simd4(1.0, 2.0, 4.0, 5.0).reciprocal() -> Simd4(1.0, 0.5, 0.25, 0.2)
    *
    * @return A Simd4 array containing the element wise reciprocal.
    */
    Simd4 reciprocal() const noexcept
    {
        return Simd4(_mm_rcp_ps(data));
    }


    /**
    * @brief Converts the array into its equivaluent __m128i value.
    *
    * @return The elements of the Simd4 array as an __m128i value.
    */
    __m128i as_m128i() const noexcept
    {
        return _mm_cvtps_epi32(data);
    }


    /**
    * @brief Sums up all the elements in the vector.
    *
    * @return The sum of all the elements.
    */
    float sum() const noexcept
    {
        __m128 sum_1 = _mm_hadd_ps(data, data);
        __m128 sum_2 = _mm_hadd_ps(sum_1, sum_1);
        return _mm_cvtss_f32(sum_2);
    }


    /**
    * @brief Sums up the first 3 elements in the vector.
    *
    * @return The sum of the first 3 elements.
    */
    float sum3() const noexcept
    {
        __m128 v = data;
        v = _mm_add_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, 1)));
        v = _mm_add_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, 1)));
        return _mm_cvtss_f32(v);
    }


    /**
    * @brief Computes the horizontal maximum element in the vector.
    *
    * Simd4(1.0, 2.0, 3.0, 4.0).hmax() -> 4.0
    *
    * @return The horizontal maximum element.
    */
    float hmax() const noexcept
    {
        __m128 v = data;
        v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3)));
        v = _mm_max_ps(v, _mm_movehl_ps(v, v));
        return _mm_cvtss_f32(v);
    }


    /**
    * @brief Computes the horizontal minimum element in the vector.
    *
    * Simd4(1.0, 2.0, 3.0, 4.0).hmin() -> 1.0
    *
    * @return The horizontal minimum element.
    */
    float hmin() const noexcept
    {
        __m128 v = data;
        v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3)));
        v = _mm_min_ps(v, _mm_movehl_ps(v, v));
        return _mm_cvtss_f32(v);
    }


    /**
    * @brief Computes the horizontal maximum of the first 3 elements in the vector.
    *
    * Simd4(1.0, 2.0, 3.0, 4.0).hmax3() -> 3.0
    *
    * @return The horizontal max element.
    */
    float hmax3() const noexcept
    {
        __m128 v = data;
        v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, 1)));
        v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, 2)));
        return _mm_cvtss_f32(v);
    }


    /**
    * @brief Computes the horizontal minimum of the first 3 elements in the vector.
    *
    * Simd4(1.0, 2.0, 3.0, 0.0).hmin3() -> 1.0
    *
    * @return The horizontal min element.
    */
    float hmin3() const noexcept
    {
        __m128 v = data;
        v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, 1)));
        v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, 2)));
        return _mm_cvtss_f32(v);
    }


    /**
    * @brief Clamps the vector between minimum and maximum vectors.
    *
    * @param min The minimum vector.
    * @param max The maximum vector.
    * @return The clamped vector.
    */
    Simd4 clamp(const Simd4& min, const Simd4& max) const noexcept
    {
        return Simd4::min(Simd4::max(*this, min), max);
    }


    /**
    * @brief Clamps the vector to a minimum and maximum value.
    *
    * @param min The minimum parameter.
    * @param max The maximum parameter.
    * @return The clamped vector.
    */
    Simd4 clamp(float min, float max) const noexcept
    {
        return clamp(Simd4(min), Simd4(max));
    }


    /**
    * @brief Shuffles the vector according to the given shuffle.
    *
    * @note Use _MM_SHUFFLE(w, z, y, x) to shuffle the mask.
    * Simd4(1.0, 2.0, 3.0, 4.0).shuffle<_MM_SHUFFLE(3, 2, 1, 0)> -> Simd4(4.0, 3.0, 2.0, 1.0)
    *
    * @tparam Mask The shuffle mask to use.
    * @return The shuffled Simd vector.
    */
    template <int Mask>
    Simd4 shuffle() const noexcept
    {
        return Simd4(_mm_shuffle_ps(data, data, Mask));
    }


    /**
    * @brief Shuffles the vectors according to the given shuffle.
    *
    * @note Use _MM_SHUFFLE(w, z, y, x) to shuffle the mask.
    * Simd4(1.0, 2.0, 3.0, 4.0).shuffle<_MM_SHUFFLE(3, 2, 1, 0)> -> Simd4(4.0, 3.0, 2.0, 1.0)
    *
    * @tparam Mask The shuffle mask to use.
    * @param other The other vector to shuffle with this.
    * @return The shuffled Simd vector.
    */
    template <int Mask>
    Simd4 shuffle(const Simd4& other) const noexcept
    {
        return Simd4(_mm_shuffle_ps(data, other.data, Mask));
    }


    /**
    * @brief Shuffles the vector in a single register.
    *
    * @note Use _MM_SHUFFLE(w, z, y, x) to shuffle the mask.
    * Simd4(1.0, 2.0, 3.0, 4.0).shuffle<_MM_SHUFFLE(3, 2, 1, 0)> -> Simd4(4.0, 3.0, 2.0, 1.0)
    *
    * @tparam Mask The shuffle mask to use.
    * @param a The vector to shuffle.
    * @return The shuffled Simd vector.
    */
    template <int Mask>
    Simd4 cast_shuffle() const noexcept
    {
        return Simd4(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(data), Mask)));
    }


    /**
    * Calculates the dot product of the vector with itself.
    *
    * @return The output of the dot product.
    */
    float dot() const noexcept
    {
        return _mm_cvtss_f32(_mm_dp_ps(data, data, 0x71));
    }


    /**
    * Calculates the dot product of the vector with another.
    *
    * @param other The other vector to calculate the dot product with.
    * @return The output of the dot product.
    */
    float dot(const Simd4& other) const noexcept
    {
        return _mm_cvtss_f32(_mm_dp_ps(data, other.data, 0x71));
    }


    /**
    * Calculates the dot product of the vector with itself.
    *
    * @tparam Mask The mask to use for the dot product.
    * @return The output of the dot product.
    */
    template <int Mask>
    Simd4 dot() const noexcept
    {
        return Simd4(_mm_dp_ps(data, data, Mask));
    }


    /**
    * Calculates the dot product of the vector with another.
    *
    * @tparam Mask The mask to use for the dot product.
    * @param other The other vector to calculate the dot product with.
    * @return The output of the dot product.
    */
    template <int Mask>
    Simd4 dot(const Simd4& other) const noexcept
    {
        return Simd4(_mm_dp_ps(data, other.data, Mask));
    }


    /**
    * Calculates the cross product of the vector with another.
    *
    * @param other The other vector to calculate the cross product with.
    * @return The output of the cross product.
    */
    const Simd4 cross_prod(const Simd4& other) const noexcept
    {
        Simd4 tmp0 = other.shuffle<_MM_SHUFFLE(3, 0, 2, 1)>();
        Simd4 tmp1 =       shuffle<_MM_SHUFFLE(3, 0, 2, 1)>();
        tmp0 *= *this;
        tmp1 *= other;
        const Simd4 tmp2 = tmp0 - tmp1;
        return tmp2.shuffle<_MM_SHUFFLE(3, 0, 2, 1)>();
    }


    /**
    * Tests to see if all values are set to 0.
    * Note this is different to bool().
    *
    * @return Whether all elements are set to 0.
    */
    bool is_zero() const noexcept
    {
        const __m128 cmp = _mm_cmpneq_ps(data, _mm_setzero_ps());
        return _mm_movemask_ps(cmp) == 0;
    }


    /**
    * Tests to see if any value is NaN.
    *
    * @return Whether any value is NaN.
    */
    bool contains_nan() const noexcept
    {
        const __m128 cmp = _mm_cmpneq_ps(data, data);
        return _mm_movemask_ps(cmp) != 0;
    }


    /**
    * @brief Returns an array containing the elements in the vector.
    *
    * Use this if you require frequent access to the elements
    * as this is faster than accessing by index each time.
    *
    * @return An array containing the elements.
    */
    std::array<float, 4> as_arr() const noexcept
    {
        alignas(16) std::array<float, 4> elements;
        _mm_store_ps(elements.data(), data);
        return elements;
    }


    /**
    * @brief Returns an array containing the elements in the vector as integers.
    *
    * Use this if you require frequent access to the elements
    * as this is faster than accessing by index each time.
    *
    * @return An array containing the elements as integers.
    */
    std::array<int, 4> as_arr_int() const noexcept
    {
        alignas(16) std::array<int, 4> elements;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(elements.data()), as_m128i());
        return elements;
    }


    /**
    * Access elements by index.
    */
    float operator[](int index) const noexcept
    {
        assert(index < 4 && index >= 0);
        return as_arr()[index];
    }


    /**
    * Calculates the element wise maximum element between the two vectors.
    *
    * @param a The first vector.
    * @param b The second vector.
    * @return A vector containing the max elements between the input vectors.
    */
    static Simd4 max(const Simd4& a, const Simd4& b) noexcept
    {
        return Simd4(_mm_max_ps(a.data, b.data));
    }


    /**
    * Calculates the element wise minimum element between the two vectors.
    *
    * @param a The first vector.
    * @param b The second vector.
    * @return A vector containing the min elements between the input vectors.
    */
    static Simd4 min(const Simd4& a, const Simd4& b) noexcept
    {
        return Simd4(_mm_min_ps(a.data, b.data));
    }


    /**
    * Performs horizontal addition along the vector.
    *
    * @param a The first vector to horizontally add.
    * @param b The second vector to horizontally add.
    * @return The horizontal sum of the vectors.
    */
    static Simd4 hadd(const Simd4& a, const Simd4& b) noexcept
    {
        return Simd4(_mm_hadd_ps(a.data, b.data));
    }


    /**
    * Performs horizontal subtraction along the vector.
    *
    * @param a The first vector to horizontally sub.
    * @param b The second vector to horizontally sub.
    * @return The horizontal subtraction of the vectors.
    */
    static Simd4 hsub(const Simd4& a, const Simd4& b) noexcept
    {
        return Simd4(_mm_hsub_ps(a.data, b.data));
    }


    /**
    * Tests a vector for equality with a given mask.
    *
    * @note The mask is flipped around: 0b0011 compares
    * the first two numbers for example.
    *
    * @tparam Mask The mask of which numbers to check.
    * @param a     The first vector to check for equality.
    * @param b     The second vector to check for equality.
    * @return Whether the vectors are equal along a mask.
    */
    template <int Mask = 0b1111>
    static bool is_eq(const Simd4& a, const Simd4& b) noexcept
    {
        const __m128 cmp = _mm_cmpeq_ps(a.data, b.data);
        const int result = _mm_movemask_ps(cmp);
        return (result & Mask) == Mask;
    }


    /**
    * Tests a vector for approximate equality with a given mask.
    *
    * @note The mask is flipped around: 0b0011 compares
    * the first two numbers for example.
    *
    * @tparam Mask    The mask of which numbers to check.
    * @tparam Epsilon The range considered to be equal.
    * @param a        The first vector to check for equality.
    * @param b        The second vector to check for equality.
    * @return Whether the vectors are equal along a mask.
    */
    template <int Mask = 0b1111>
    static bool is_approx_eq(const Simd4& a, const Simd4& b, float epsilon) noexcept
    {
        const __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.f), _mm_sub_ps(a.data, b.data));
        const __m128 cmp      = _mm_cmplt_ps(abs_diff, _mm_set1_ps(epsilon));
        const int result      = _mm_movemask_ps(cmp);
        return (result & Mask) == Mask;
    }


    /**
    * @brief Selects between two vectors based on a mask.
    * Returns mask ? a : b for each component.
    *
    * @param mask The mask vector (typically from a comparison)
    * @param a First vector (selected when mask is true)
    * @param b Second vector (selected when mask is false)
    * @return A new vector with components selected from a or b based on mask
    */
    static Simd4 select(const Simd4& mask, const Simd4& a, const Simd4& b) noexcept
    {
        return Simd4(_mm_blendv_ps(b.data, a.data, mask.data));
    }


    /**
    * Special cases for shuffles.
    */
    friend Simd4 shuffle_0101(const Simd4& a, const Simd4& b) noexcept { return Simd4(_mm_movelh_ps(a.data, b.data)); }
    friend Simd4 shuffle_2323(const Simd4& a, const Simd4& b) noexcept { return Simd4(_mm_movehl_ps(b.data, a.data)); }
    friend Simd4 shuffle_0022(const Simd4& a) noexcept { return Simd4(_mm_moveldup_ps(a.data)); }
    friend Simd4 shuffle_1133(const Simd4& a) noexcept { return Simd4(_mm_movehdup_ps(a.data)); }

    /**
    * Overloaded operations.
    */
    Simd4 operator+(const Simd4& other) const noexcept { return Simd4(_mm_add_ps(data, other.data)); }
    Simd4 operator-(const Simd4& other) const noexcept { return Simd4(_mm_sub_ps(data, other.data)); }
    Simd4 operator*(const Simd4& other) const noexcept { return Simd4(_mm_mul_ps(data, other.data)); }
    Simd4 operator/(const Simd4& other) const noexcept { return Simd4(_mm_div_ps(data, other.data)); }

    void operator+=(const Simd4& other) noexcept { data = _mm_add_ps(data, other.data); }
    void operator-=(const Simd4& other) noexcept { data = _mm_sub_ps(data, other.data); }
    void operator*=(const Simd4& other) noexcept { data = _mm_mul_ps(data, other.data); }
    void operator/=(const Simd4& other) noexcept { data = _mm_div_ps(data, other.data); }

    Simd4 operator+(float scalar) const noexcept { return Simd4(_mm_add_ps(data, _mm_set1_ps(scalar))); }
    Simd4 operator-(float scalar) const noexcept { return Simd4(_mm_sub_ps(data, _mm_set1_ps(scalar))); }
    Simd4 operator*(float scalar) const noexcept { return Simd4(_mm_mul_ps(data, _mm_set1_ps(scalar))); }
    Simd4 operator/(float scalar) const noexcept { return Simd4(_mm_div_ps(data, _mm_set1_ps(scalar))); }

    void operator+=(float scalar) noexcept { data = _mm_add_ps(data, _mm_set1_ps(scalar)); }
    void operator-=(float scalar) noexcept { data = _mm_sub_ps(data, _mm_set1_ps(scalar)); }
    void operator*=(float scalar) noexcept { data = _mm_mul_ps(data, _mm_set1_ps(scalar)); }
    void operator/=(float scalar) noexcept { data = _mm_div_ps(data, _mm_set1_ps(scalar)); }

    Simd4 operator-() const noexcept { return *this * Simd4::minus_one(); }

    Simd4 operator&(const Simd4& other) const noexcept { return Simd4(_mm_and_ps(data, other.data)); }
    Simd4 operator|(const Simd4& other) const noexcept { return Simd4(_mm_or_ps(data, other.data)); }
    Simd4 operator^(const Simd4& other) const noexcept { return Simd4(_mm_xor_ps(data, other.data)); }
    Simd4 operator~() const noexcept { return Simd4(_mm_andnot_ps(data, _mm_set1_ps(-1.0f))); }

    /**
    * Comparison operators
    */
    Simd4 operator==(const Simd4& other) const noexcept { return Simd4(_mm_cmpeq_ps(data, other.data)); }
    Simd4 operator!=(const Simd4& other) const noexcept { return Simd4(_mm_cmpneq_ps(data, other.data)); }
    Simd4 operator< (const Simd4& other) const noexcept { return Simd4(_mm_cmplt_ps(data, other.data)); }
    Simd4 operator> (const Simd4& other) const noexcept { return Simd4(_mm_cmpgt_ps(data, other.data)); }
    Simd4 operator<=(const Simd4& other) const noexcept { return Simd4(_mm_cmple_ps(data, other.data)); }
    Simd4 operator>=(const Simd4& other) const noexcept { return Simd4(_mm_cmpge_ps(data, other.data)); }


    /**
    * Prints the Simd4 vector into an ostream.
    */
    friend std::ostream& operator<<(std::ostream &os, const Simd4 &s)
    {
        std::array<float, 4> buf = s.as_arr();
        os << "[" << buf[0] << ", " << buf[1] << ", " << buf[2] << ", " << buf[3] << "]";
        return os;
    }
};
