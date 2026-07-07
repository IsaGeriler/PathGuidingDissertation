#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <algorithm>
#include <cassert>
#include <string.h>

#define SQ(x)(x * x)

#pragma warning(disable : 4244)  // Stop warnings about M_PI being a double

// Helper Functions for MIS (Multiple Importance Sampling)
// Balanced Heuristics
float weightBalancedHeurictics(float pA, float pB) {
	if (pA + pB <= 0.f) return 0.f;
	return pA / (pA + pB);
}

// Power Heuristics
// Beta is set to 2 as default, but can be increased further
float weightPowerHeuristics(float pA, float pB, float beta = 2.f) {
	float pABeta = std::pow(pA, beta);
	float pBBeta = std::pow(pB, beta);
	if (pABeta + pBBeta <= 0.f) return 0.f;
	return pABeta / (pABeta + pBBeta);
}

class Colour {
public:
	// Attributes: Red - Green - Blue Channels
	float r, g, b;

	// Constructors
	Colour() { r = 0.f; g = 0.f; b = 0.f; }
	Colour(float _r, float _g, float _b) { r = _r; g = _g; b = _b; }

	Colour(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a) {
		r = static_cast<float>(_r / 255.f);
		g = static_cast<float>(_g / 255.f);
		b = static_cast<float>(_b / 255.f);
	}

	// Operator Overloading
	Colour operator+(const Colour& colour) const { return Colour(r + colour.r, g + colour.g, b + colour.b); }
	Colour operator-(const Colour& colour) const { return Colour(r - colour.r, g - colour.g, b - colour.b); }
	Colour operator*(const Colour& colour) const { return Colour(r * colour.r, g * colour.g, b * colour.b); }
	Colour operator/(const Colour& colour) const { return Colour(r / colour.r, g / colour.g, b / colour.b); }

	Colour operator+(const float v) const { return Colour(r + v, g + v, b + v); }
	Colour operator*(const float v) const { return Colour(r * v, g * v, b * v); }

	Colour operator/(const float v) const {
		float invV = 1.f / v;
		return Colour(r * invV, g * invV, b * invV);
	}

	// Methods
	float Lum() const {
		return (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
	}

	void toRGB(unsigned char& cr, unsigned char& cg, unsigned char& cb) const {
		cr = (unsigned char)(r * 255);
		cg = (unsigned char)(g * 255);
		cb = (unsigned char)(b * 255);
	}
};

class Vec4 {
public:
	// Attributes - Vector Components
	union {
		struct { float x, y, z, w; };
		float coords[4];
	};

	// Constructors
	Vec4() { x = 0.f; y = 0.f; z = 0.f; w = 1.f; }
	Vec4(float _x, float _y, float _z) { x = _x; y = _y; z = _z; w = 1.f; }
	Vec4(float _x, float _y, float _z, float _w) { x = _x; y = _y; z = _z; w = _w; }

	// Operator Overloading
	Vec4 operator+(const Vec4 v) const { return Vec4(x + v.x, y + v.y, z + v.z); }
	Vec4 operator-(const Vec4 v) const { return Vec4(x - v.x, y - v.y, z - v.z); }
	Vec4 operator*(const Vec4 v) const { return Vec4(x * v.x, y * v.y, z * v.z); }

	Vec4 operator*(const float v) const { return Vec4(x * v, y * v, z * v); }

	Vec4 operator/(const float v) const {
		float invV = 1.f / v;
		return Vec4(x * invV, y * invV, z * invV, w * invV);
	}

	// Operator Overloading - Negation
	Vec4 operator-() const { return Vec4(-x, -y, -z); }

	// Operator Overloading - Indexing
	float& operator[](int index) {
		assert((index >= 0 && index < 4) && "Invalid index passed, out of bounds!");
		return coords[index];
	}
	
	// Methods
	Vec4 cross(Vec4 v) const {
		return Vec4(
			(y * v.z) - (z * v.y),
			(z * v.x) - (x * v.z),
			(x * v.y) - (y * v.x)
		);
	}

	float dot(Vec4 v) const { return (x * v.x) + (y * v.y) + (z * v.z); }

	Vec4 perspectiveDivide() const {
		float invW = 1.f / w;
		return Vec4(x * invW, y * invW, z * invW, invW);
	}

	float length() const { return sqrtf(SQ(x) + SQ(y) + SQ(z)); }
	float lengthSquare() const { return SQ(x) + SQ(y) + SQ(z); }
	
	Vec4 normalize() const {
		float l = 1.f / length();
		return Vec4(x * l, y * l, z * l);
	}
};

static Vec4 Cross(const Vec4& v1, const Vec4& v2) {
	return Vec4(
		(v1.y * v2.z) - (v1.z * v2.y),
		(v1.z * v2.x) - (v1.x * v2.z),
		(v1.x * v2.y) - (v1.y * v2.x)
	);
}

static float Dot(const Vec4 v1, const Vec4 v2) { return ((v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)); }
static Vec4 Max(Vec4 v1, Vec4 v2) { return Vec4(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z)); }
static Vec4 Min(Vec4 v1, Vec4 v2) { return Vec4(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z)); }

struct Vertex {
	Vec4 p, normal;
	float u, v;
};

class Matrix {
public:
	// Attributes - Store Matrix in 1D and 2D forms
	union {
		float a[4][4];
		float m[16];
	};

	// Constructors
	Matrix() { identity(); }
	Matrix(float m00, float m01, float m02, float m03,
		   float m10, float m11, float m12, float m13,
		   float m20, float m21, float m22, float m23,
		   float m30, float m31, float m32, float m33) {
		// First Row
		a[0][0] = m00; a[0][1] = m01; a[0][2] = m02; a[0][3] = m03;
		// Second Row
		a[1][0] = m10; a[1][1] = m11; a[1][2] = m12; a[1][3] = m13;
		// Third Row
		a[2][0] = m20; a[2][1] = m21; a[2][2] = m22; a[2][3] = m23;
		// Fourth Row
		a[3][0] = m30; a[3][1] = m31; a[3][2] = m32; a[3][3] = m33;
	}

	// Operator Overloading
	float& operator[](int index) {
		assert((index >= 0 && index < 16) && "Invalid index passed, out of bounds!");
		return m[index];
	}

	// Methods
	void identity() {
		memset(m, 0, 16 * sizeof(float));
		m[0] = 1.f; m[5] = 1.f; m[10] = 1.f; m[15] = 1.f;
	}

	Matrix transpose() {
		return Matrix(
			a[0][0], a[1][0], a[2][0], a[3][0],
			a[0][1], a[1][1], a[2][1], a[3][1],
			a[0][2], a[1][2], a[2][2], a[3][2],
			a[0][3], a[1][3], a[2][3], a[3][3]
		);
	}
	
	static Matrix translation(const Vec4& v) {
		Matrix mat;
		mat.a[0][3] = v.x;
		mat.a[1][3] = v.y;
		mat.a[2][3] = v.z;
		return mat;
	}

	static Matrix scaling(const Vec4& v) {
		Matrix mat;
		mat.m[0] = v.x;
		mat.m[5] = v.y;
		mat.m[10] = v.z;
		return mat;
	}

	Matrix mul(const Matrix& matrix) const {
		Matrix ret;
		ret.m[0] = m[0] * matrix.m[0] + m[1] * matrix.m[4] + m[2] * matrix.m[8] + m[3] * matrix.m[12];
		ret.m[1] = m[0] * matrix.m[1] + m[1] * matrix.m[5] + m[2] * matrix.m[9] + m[3] * matrix.m[13];
		ret.m[2] = m[0] * matrix.m[2] + m[1] * matrix.m[6] + m[2] * matrix.m[10] + m[3] * matrix.m[14];
		ret.m[3] = m[0] * matrix.m[3] + m[1] * matrix.m[7] + m[2] * matrix.m[11] + m[3] * matrix.m[15];
		ret.m[4] = m[4] * matrix.m[0] + m[5] * matrix.m[4] + m[6] * matrix.m[8] + m[7] * matrix.m[12];
		ret.m[5] = m[4] * matrix.m[1] + m[5] * matrix.m[5] + m[6] * matrix.m[9] + m[7] * matrix.m[13];
		ret.m[6] = m[4] * matrix.m[2] + m[5] * matrix.m[6] + m[6] * matrix.m[10] + m[7] * matrix.m[14];
		ret.m[7] = m[4] * matrix.m[3] + m[5] * matrix.m[7] + m[6] * matrix.m[11] + m[7] * matrix.m[15];
		ret.m[8] = m[8] * matrix.m[0] + m[9] * matrix.m[4] + m[10] * matrix.m[8] + m[11] * matrix.m[12];
		ret.m[9] = m[8] * matrix.m[1] + m[9] * matrix.m[5] + m[10] * matrix.m[9] + m[11] * matrix.m[13];
		ret.m[10] = m[8] * matrix.m[2] + m[9] * matrix.m[6] + m[10] * matrix.m[10] + m[11] * matrix.m[14];
		ret.m[11] = m[8] * matrix.m[3] + m[9] * matrix.m[7] + m[10] * matrix.m[11] + m[11] * matrix.m[15];
		ret.m[12] = m[12] * matrix.m[0] + m[13] * matrix.m[4] + m[14] * matrix.m[8] + m[15] * matrix.m[12];
		ret.m[13] = m[12] * matrix.m[1] + m[13] * matrix.m[5] + m[14] * matrix.m[9] + m[15] * matrix.m[13];
		ret.m[14] = m[12] * matrix.m[2] + m[13] * matrix.m[6] + m[14] * matrix.m[10] + m[15] * matrix.m[14];
		ret.m[15] = m[12] * matrix.m[3] + m[13] * matrix.m[7] + m[14] * matrix.m[11] + m[15] * matrix.m[15];
		return ret;

		// TO:DO - 4x4 Matrix Multiplication with SIMD (SSE/AVX/AVX2)
		//__m128 first_row = _mm_loadu_ps(&matrix.a[0][0]);
		//__m128 second_row = _mm_loadu_ps(&matrix.a[1][0]);
		//__m128 third_row = _mm_loadu_ps(&matrix.a[2][0]);
		//__m128 fourth_row = _mm_loadu_ps(&matrix.a[3][0]);

		//for (unsigned int i = 0; i < 4; ++i) {
		//	__m128 x = _mm_loadu_ps(&a[i][0]);
		//	__m128 y = _mm_loadu_ps(&a[i][1]);
		//	__m128 z = _mm_loadu_ps(&a[i][2]);
		//	__m128 w = _mm_loadu_ps(&a[i][3]);

		//	__m128 A = _mm_fmadd_ps(x, first_row, _mm_mul_ps(y, second_row));
		//	__m128 B = _mm_fmadd_ps(z, third_row, _mm_mul_ps(w, fourth_row));
		//	__m128 result = _mm_add_ps(A, B);
		//	_mm_storeu_ps(&ret.a[i][0], result);
		//}
	}

	Matrix operator*(const Matrix& matrix) { return mul(matrix); }

	Vec4 mulVec(const Vec4& v) {
		return Vec4(
			(v.x * m[0] + v.y * m[1] + v.z * m[2]),
			(v.x * m[4] + v.y * m[5] + v.z * m[6]),
			(v.x * m[8] + v.y * m[9] + v.z * m[10])
		);
	}

	Vec4 mulPoint(const Vec4& v) {
		return Vec4(
			(v.x * m[0] + v.y * m[1] + v.z * m[2]) + m[3],
			(v.x * m[4] + v.y * m[5] + v.z * m[6]) + m[7],
			(v.x * m[8] + v.y * m[9] + v.z * m[10]) + m[11]
		);
	}

	Vec4 mulPointAndPerspectiveDivide(const Vec4& v) {
		Vec4 v1 = Vec4(
			(v.x * m[0] + v.y * m[1] + v.z * m[2]) + m[3],
			(v.x * m[4] + v.y * m[5] + v.z * m[6]) + m[7],
			(v.x * m[8] + v.y * m[9] + v.z * m[10]) + m[11]
		);
		float w;
		w = (m[12] * v.x) + (m[13] * v.y) + (m[14] * v.z) + m[15];
		w = 1.f / w;
		return (v1 * w);
	}

	Matrix operator=(const Matrix& matrix) {
		memcpy(m, matrix.m, sizeof(float) * 16);
		return (*this);
	}
	
	Matrix invert() {
		Matrix inv;
		// Unrolled inverse from MESA library
		// First Column
		inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
		inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
		inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
		inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
		// Second Column
		inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
		inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
		inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
		inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
		// Third Column
		inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
		inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
		inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
		inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
		// Fourth Column
		inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
		inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
		inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
		inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
		
		float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];	
		// This code should never be called. Add error handling if it is
		if (det == 0.f) {
			identity();
			det = 1.f;
		}

		// Take the inverse determinant
		det = 1.f / det;
		for (int i = 0; i < 16; i++) {
			inv[i] = inv[i] * det;
		}
		return inv;
	}

	static Matrix lookAt(const Vec4& from, const Vec4& to, const Vec4& up) {
		Vec4 dir = (from - to).normalize();
		Vec4 left = up.cross(dir).normalize();
		Vec4 newUp = dir.cross(left);

		Matrix mat;
		mat.a[0][0] = left.x;
		mat.a[0][1] = left.y;
		mat.a[0][2] = left.z;

		mat.a[1][0] = newUp.x;
		mat.a[1][1] = newUp.y;
		mat.a[1][2] = newUp.z;

		mat.a[2][0] = dir.x;
		mat.a[2][1] = dir.y;
		mat.a[2][2] = dir.z;

		mat.a[0][3] = -from.dot(left);
		mat.a[1][3] = -from.dot(newUp);
		mat.a[2][3] = -from.dot(dir);
		mat.a[3][3] = 1.f;
		return mat;
	}

	static Matrix perspective(const float n, const float f, float aspect, const float fov) {
		Matrix pers;
		memset(pers.m, 0, 16 * sizeof(float));

		// FOV (Field of View) in degrees, outputs transposed Matrix for DX
		float t = 1.f / (tanf(fov * 0.5f * 3.141592654f / 180.f));
		float invFarMinusNear = 1.f / (f - n);

		pers.a[0][0] = t / aspect;
		pers.a[1][1] = t;
		pers.a[2][2] = -f * invFarMinusNear;
		pers.a[2][3] = -(f * n) * invFarMinusNear;
		pers.a[3][2] = -1.f;
		return pers;
	}

	static Matrix rotateX(float theta) {
		// Store cos(theta) and sin(theta) to avoid multiple calculations
		float ct = cosf(theta);
		float st = sinf(theta);

		Matrix mat;
		mat.m[5] = ct;
		mat.m[6] = st;
		mat.m[9] = -st;
		mat.m[10] = ct;
		return mat;
	}

	static Matrix rotateY(float theta) {
		// Store cos(theta) and sin(theta) to avoid multiple calculations
		float ct = cosf(theta);
		float st = sinf(theta);

		Matrix mat;
		mat.m[0] = ct;
		mat.m[2] = -st;
		mat.m[8] = st;
		mat.m[10] = ct;
		return mat;
	}

	static Matrix rotateZ(float theta) {
		// Store cos(theta) and sin(theta) to avoid multiple calculations
		float ct = cosf(theta);
		float st = sinf(theta);

		Matrix mat;
		mat.m[0] = ct;
		mat.m[1] = st;
		mat.m[4] = -st;
		mat.m[5] = ct;
		return mat;
	}
};

class Frame {
public:
	// Attributes - Normalized Vectors
	Vec4 u, v, w;

	// Methods
	void fromVector(const Vec4& n) {
		// Gram-Schmidt Process
		w = n.normalize();
		if (fabsf(w.x) > fabsf(w.y)) {
			float l = 1.f / sqrtf(w.x * w.x + w.z * w.z);
			u = Vec4(w.z * l, 0.f, -w.x * l);
		} else {
			float l = 1.f / sqrtf(w.y * w.y + w.z * w.z);
			u = Vec4(0.f, w.z * l, -w.y * l);
		}
		v = Cross(w, u);
	}

	void fromVectorTangent(const Vec4& n, const Vec4& t) {
		w = n.normalize();
		u = t.normalize();
		v = Cross(w, u);
	}

	Vec4 toLocal(const Vec4& vec) const { return Vec4(Dot(vec, u), Dot(vec, v), Dot(vec, w)); }
	Vec4 toWorld(const Vec4& vec) const { return ((u * vec.x) + (v * vec.y) + (w * vec.z)); }
};

class SphericalCoordinates {
public:
	static Vec4 sphericalToWorld(float theta, float phi) {
		return Vec4(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
	}

	static float sphericalTheta(const Vec4& wi) {
		return acosf(wi.z);
	}

	static float sphericalPhi(const Vec4& wi) {
		float p = atan2f(wi.y, wi.x);
		return (p < 0.f) ? (p + (2.f * 3.14159265358979323846)) : p;
	}
};

template<typename T>
T& use() {
	static T t;
	return t;
}