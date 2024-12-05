/**
*   @file matrix.h
*   @brief This file contains a simple implementation of a 4x4 matrix class.
*
*   @info REQUIRED FLAGS: -std=c++20 -mfma -msse4.1
*/

#pragma once


#include <cassert>
#include <xmmintrin.h>
#include <iostream>


#include "simd.h"


/**
* @class Matrix4x4
* @brief SIMD-optimized 4x4 matrix operations.
*/
class Matrix4x4
{
public:
    /**
    * Default constructor - identity matrix.
    */
    Matrix4x4()
    {
        rows[0] = Simd4::unit_x();
        rows[1] = Simd4::unit_y();
        rows[2] = Simd4::unit_z();
        rows[3] = Simd4::unit_w();
    }


    /**
    * Constructor with values.
    */
    Matrix4x4(float a, float b, float c, float d,
              float e, float f, float g, float h,
              float i, float j, float k, float l,
              float m, float n, float o, float p)
    {
        rows[0] = Simd4(a, b, c, d);
        rows[1] = Simd4(e, f, g, h);
        rows[2] = Simd4(i, j, k, l);
        rows[3] = Simd4(m, n, o, p);
    }


    /**
    * Constructor with direct Simd4 values.
    */
    Matrix4x4(__m128 r0, __m128 r1, __m128 r2, __m128 r3)
    {
        rows[0] = Simd4(r0);
        rows[1] = Simd4(r1);
        rows[2] = Simd4(r2);
        rows[3] = Simd4(r3);
    }


    /**
    * Constructor with direct Simd4 values.
    */
    Matrix4x4(Simd4 r0, Simd4 r1, Simd4 r2, Simd4 r3)
    {
        rows[0] = r0;
        rows[1] = r1;
        rows[2] = r2;
        rows[3] = r3;
    }


    /**
    * Calculates the inverse of the given matrix.
    *
    * @param v The vector to apply the transform to.
    * @return The vector with the transform applied.
    */
    Matrix4x4 inverse() const noexcept
    {
        const Simd4 A = shuffle_0101(rows[0], rows[1]);
        const Simd4 B = shuffle_0101(rows[2], rows[3]);
        const Simd4 C = shuffle_2323(rows[0], rows[1]);
        const Simd4 D = shuffle_2323(rows[2], rows[3]);

        const Simd4 det_X = (
            rows[0].shuffle<_MM_SHUFFLE(2, 0, 2, 0)>(rows[2])
            * rows[1].shuffle<_MM_SHUFFLE(3, 1, 3, 1)>(rows[3])
        ) - (
            rows[0].shuffle<_MM_SHUFFLE(3, 1, 3, 1)>(rows[2])
            * rows[1].shuffle<_MM_SHUFFLE(2, 0, 2, 0)>(rows[3])
        );

        const Simd4 det_A = det_X.cast_shuffle<_MM_SHUFFLE(0, 0, 0, 0)>();
        const Simd4 det_B = det_X.cast_shuffle<_MM_SHUFFLE(2, 2, 2, 2)>();
        const Simd4 det_C = det_X.cast_shuffle<_MM_SHUFFLE(1, 1, 1, 1)>();
        const Simd4 det_D = det_X.cast_shuffle<_MM_SHUFFLE(3, 3, 3, 3)>();

        const Simd4 DC = mat2x2_mul<true, false>(D, C);
        const Simd4 AB = mat2x2_mul<true, false>(A, B);

        Simd4 X = (det_D * A) - mat2x2_mul(B, DC);
        Simd4 W = (det_A * D) - mat2x2_mul(C, AB);

        // Calculate first part of det_M with det_A and det_D still in cache
        Simd4 det = det_A * det_D;

        Simd4 Y = (det_B * C) - mat2x2_mul<false, true>(D, AB);
        Simd4 Z = (det_C * B) - mat2x2_mul<false, true>(A, DC);

        det = det + (det_B * det_C);

        Simd4 tr = AB * DC.shuffle<_MM_SHUFFLE(3, 1, 2, 0)>();

        tr = Simd4::hadd(tr, tr);
        tr = Simd4::hadd(tr, tr);

        det = det - tr;

        const Simd4 r_det = Simd4::adj_mask() / det;

        X *= r_det;
        Y *= r_det;
        Z *= r_det;
        W *= r_det;

        Matrix4x4 ret;

        ret.rows[0] = X.shuffle<_MM_SHUFFLE(1, 3, 1, 3)>(Z);
        ret.rows[1] = X.shuffle<_MM_SHUFFLE(0, 2, 0, 2)>(Z);
        ret.rows[2] = Y.shuffle<_MM_SHUFFLE(1, 3, 1, 3)>(W);
        ret.rows[3] = Y.shuffle<_MM_SHUFFLE(0, 2, 0, 2)>(W);

        return ret;
    }


    /**
    * Transposes the matrix in place.
    */
    void transpose()
    {
        _MM_TRANSPOSE4_PS(rows[0].data, rows[1].data, rows[2].data, rows[3].data);
    }


    /**
    * Returns a transposed copy of the matrix.
    * @return The transposed matrix
    */
    Matrix4x4 transposed() const
    {
        __m128 r0 = rows[0].data;
        __m128 r1 = rows[1].data;
        __m128 r2 = rows[2].data;
        __m128 r3 = rows[3].data;
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
        return Matrix4x4(r0, r1, r2, r3);
    }


    /**
    * Gets a row of the matrix.
    * @param index The index of the row to get
    * @return The row of the matrix
    */
    const Simd4& get_row(int index) const
    {
        assert(index >= 0 && index < 4);
        return rows[index];
    }


    /**
    * Gets a value at a specific index of the matrix.
    * @param index The index of the value to get
    * @return The value at the index
    */
    float at(int index) const
    {
        assert(index >= 0 && index < 16);
        return rows[index / 4][index % 4];
    }


    /**
    * Gets a value at a specific index of the matrix.
    * @param row The row of the value to get
    * @param col The column of the value to get
    * @return The value at the index
    */
    float at(int row, int col) const
    {
        assert(row >= 0 && row < 4 && col >= 0 && col < 4);
        return rows[row][col];
    }


    /**
    * Arithmetic operators
    */
    Matrix4x4 operator*(float scalar) const noexcept
    { 
        return Matrix4x4(rows[0] * scalar, rows[1] * scalar, 
                         rows[2] * scalar, rows[3] * scalar); 
    }
    void operator*=(float scalar) noexcept
    { 
        rows[0] *= scalar; rows[1] *= scalar; 
        rows[2] *= scalar; rows[3] *= scalar; 
    }

    Matrix4x4 operator/(float scalar) const noexcept
    { 
        float inv = 1.0f / scalar;
        return *this * inv;
    }
    void operator/=(float scalar) noexcept
    { 
        float inv = 1.0f / scalar;
        *this *= inv;
    }

    Matrix4x4 operator+(const Matrix4x4& rhs) const noexcept
    { 
        return Matrix4x4(rows[0] + rhs.rows[0], 
                         rows[1] + rhs.rows[1],
                         rows[2] + rhs.rows[2], 
                         rows[3] + rhs.rows[3]); 
    }
    void operator+=(const Matrix4x4& rhs) noexcept
    { 
        rows[0] += rhs.rows[0]; 
        rows[1] += rhs.rows[1];
        rows[2] += rhs.rows[2]; 
        rows[3] += rhs.rows[3];
    }

    Matrix4x4 operator-(const Matrix4x4& rhs) const noexcept
    { 
        return Matrix4x4(rows[0] - rhs.rows[0], 
                         rows[1] - rhs.rows[1],
                         rows[2] - rhs.rows[2], 
                         rows[3] - rhs.rows[3]); 
    }
    void operator-=(const Matrix4x4& rhs) noexcept
    { 
        rows[0] -= rhs.rows[0];
        rows[1] -= rhs.rows[1];
        rows[2] -= rhs.rows[2];
        rows[3] -= rhs.rows[3];
    }

    // Unary operators
    Matrix4x4 operator-() const noexcept
    { 
        return Matrix4x4(-rows[0], -rows[1], -rows[2], -rows[3]); 
    }

    // Matrix multiplication
    Matrix4x4 operator*(const Matrix4x4& rhs) const noexcept
    {
        Matrix4x4 result;
        Matrix4x4 t = rhs.transposed();
        for (int i = 0; i < 4; i++) {
            result.rows[i] = rows[i].dot<0xF1>(t.rows[0])
                           + rows[i].dot<0xF2>(t.rows[1])
                           + rows[i].dot<0xF4>(t.rows[2])
                           + rows[i].dot<0xF8>(t.rows[3]);
        }
        return result;
    }

    void operator*=(const Matrix4x4& rhs) noexcept
    {
        *this = *this * rhs;
    }

    /**
    * Comparison operators - equal
    *
    * @note To see if two matrices are equal:
    *       Matrix4x4::is_eq(a, b)
    */
    Matrix4x4 operator==(const Matrix4x4& rhs) const noexcept
    {
        return Matrix4x4(
            rows[0] == rhs.rows[0],
            rows[1] == rhs.rows[1],
            rows[2] == rhs.rows[2],
            rows[3] == rhs.rows[3]
        );
    }


    /**
    * Comparison operators - not equal
    *
    * @note To see if two matrices are not equal:
    *       !Matrix4x4::is_eq(a, b)
    */
    Matrix4x4 operator!=(const Matrix4x4& rhs) const noexcept
    {
        return Matrix4x4(
            rows[0] != rhs.rows[0],
            rows[1] != rhs.rows[1],
            rows[2] != rhs.rows[2],
            rows[3] != rhs.rows[3]
        );
    }


    /**
    * See if two matrices are equal.
    *
    * @return True if the matrices are equal.
    */
    bool is_eq(const Matrix4x4& rhs) const noexcept
    {
        return Simd4::is_eq(rows[0], rhs.rows[0]) &&
               Simd4::is_eq(rows[1], rhs.rows[1]) &&
               Simd4::is_eq(rows[2], rhs.rows[2]) &&
               Simd4::is_eq(rows[3], rhs.rows[3]);
    }


    /**
    * See if two matrices are approximately equal.
    *
    * @return True if the matrices are approximately equal.
    */
    bool is_approx_eq(const Matrix4x4& rhs, float epsilon) const noexcept
    {
        return Simd4::is_approx_eq(rows[0], rhs.rows[0], epsilon) &&
               Simd4::is_approx_eq(rows[1], rhs.rows[1], epsilon) &&
               Simd4::is_approx_eq(rows[2], rhs.rows[2], epsilon) &&
               Simd4::is_approx_eq(rows[3], rhs.rows[3], epsilon);
    }


    /**
    * See if all bits are set to 1.
    *
    * @return True if all bits are set to 1.
    */
    bool all_one() const noexcept
    {
        return rows[0].all_one() && rows[1].all_one() && rows[2].all_one() && rows[3].all_one();
    }


    /**
    * See if all bits are set to 0.
    *
    * @return True if all bits are set to 0.
    */
    bool all_zero() const noexcept
    {
        return rows[0].all_zero() && rows[1].all_zero() && rows[2].all_zero() && rows[3].all_zero();
    }


    /**
    * Multiplies the matrix by a Simd4 vector.
    * Performs the multiplication M * v where M is this matrix and v is the vector.
    *
    * @param v The vector to multiply with
    * @return The resulting transformed vector
    */
    Simd4 operator*(const Simd4& v) const noexcept
    {
        return rows[0].dot<0xF1>(v) + rows[1].dot<0xF2>(v) + rows[2].dot<0xF4>(v) + rows[3].dot<0xF8>(v);
    }


    friend std::ostream& operator<<(std::ostream &os, const Matrix4x4 &m)
    {
        os << "Matrix4x4 {" << m.rows[0] << ", " << m.rows[1] << ", " << m.rows[2] << ", " << m.rows[3] << "}";
        return os;
    }


protected:
    Simd4 rows[4];


private:
    /**
    * Multiplies two 2x2 matrices together with optional adjugate operations.
    * Three multiplication modes:
    * 1. Standard:    A * B
    * 2. Adj_a:       A' * B    (where A' is adjugate of A)
    * 3. Adj_b:       A * B'    (where B' is adjugate of B)
    *
    * @tparam Adj_a If true, use adjugate of matrix A
    * @tparam Adj_b If true, use adjugate of matrix B
    * @param a First 2x2 matrix stored as [a00 a01 a10 a11]
    * @param b Second 2x2 matrix stored as [b00 b01 b10 b11]
    * @return Result of matrix multiplication
    */
    template <bool Adj_a = false, bool Adj_b = false>
    static inline Simd4 mat2x2_mul(const Simd4& a, const Simd4& b)
    {
        if constexpr (Adj_a) {
            // A' * B = [a11 -a01; -a10 a00] * B
            return (a.shuffle<_MM_SHUFFLE(0, 3, 0, 3)>() * b                                   ) -
            (a.shuffle<_MM_SHUFFLE(1, 2, 1, 2)>() * b.shuffle<_MM_SHUFFLE(2, 3, 0, 1)>());
        }
        else if constexpr (Adj_b) {
            // A * B' = A * [b11 -b01; -b10 b00]
            return (a                                    * b.shuffle<_MM_SHUFFLE(0, 0, 3, 3)>()) -
            (a.shuffle<_MM_SHUFFLE(1, 0, 3, 2)>() * b.shuffle<_MM_SHUFFLE(2, 2, 1, 1)>());
        }
        else {
            // Standard matrix multiplication A * B
            return (a                                    * b.shuffle<_MM_SHUFFLE(3, 3, 0, 0)>()) +
            (a.shuffle<_MM_SHUFFLE(1, 0, 3, 2)>() * b.shuffle<_MM_SHUFFLE(2, 2, 1, 1)>());
        }
    }
};
