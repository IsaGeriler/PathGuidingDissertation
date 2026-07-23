#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <random>

#include "MyMath.h"
#include "Imaging.h"

class Sampler {
public:
	virtual float next() = 0;
};

class MTRandom : public Sampler {
public:
	std::mt19937 generator;
	std::uniform_real_distribution<float> dist;

	MTRandom(unsigned int seed = 1) : dist(0.f, 1.f) { generator.seed(seed); }
	float next() { return dist(generator); }
};

// TestSampler Class for Testing BSDF Inversion
class TestSampler : public Sampler {
private:
	float u, v;
	int called;
public:
	TestSampler() : u(0.f) , v(0.f), called(0) {}

	void set(float _u, float _v) {
		u = _u;
		v = _v;
		called = 0;
	}

	float next() {
		float randomNumber = 0.f;
		if (called == 0) randomNumber = u;
		else if (called == 1) randomNumber = v;
		else randomNumber = 0.5f;
		
		called++;
		return randomNumber;
	}
};

// PSS (Primary Sample Space) will be implemented as a sampler similar to MTRandom
struct PrimarySample {
private:
	float valueBackup = 0.f;
	int modifyBackup = 0, lastModifyIter = 0;
public:
	float value = 0.f;

	void backup() {
		valueBackup = value;
		modifyBackup = lastModifyIter;
	}

	void restore() {
		value = valueBackup;
		lastModifyIter = modifyBackup;
	}
};

// Note all of these distributions assume z-up coordinate system
class SamplingDistributions {
public:
	static Vec4 uniformSampleHemisphere(float r1, float r2) {
		float theta = acos(r1);
		float phi = 2.f * 3.14159265358979323846 * r2;
		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float uniformHemispherePDF(const Vec4 wi) {
		return 1.f / (2.f * 3.14159265358979323846);
	}

	static Vec4 uniformSampleSphere(float r1, float r2) {
		float theta = acos(1.f - (2.f * r1));
		float phi = 2.f * 3.14159265358979323846 * r2;
		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float uniformSpherePDF(const Vec4& wi) {
		return 1.f / (4.f * 3.14159265358979323846);
	}

	static Vec4 cosineSampleHemisphere(float r1, float r2) {
		float theta = acos(sqrt(r1));
		float phi = 2.f * 3.14159265358979323846 * r2;
		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float cosineHemispherePDF(const Vec4 wi) {
		float theta = SphericalCoordinates::sphericalTheta(wi);
		return (cos(theta) > 0.f) ? cos(theta) * 0.318309886183790671538 : 0.f;
	}
};

// Tabulated Sampling for Environment Map
class Distribution1D {
private:
	int binarySearch(float u) {
		int first = 0;
		int size = cdf.size();
		int iter = size;

		while (iter > 0) {
			int half = iter / 2;
			int middle = first + half;
			if (cdf[middle] <= u) {
				first = middle + 1;
				iter -= (half + 1);
			} else {
				iter = half;
			}
		}
		return std::max(std::min(size - 2, first - 1), 0);
	}
public:
	// Attributes
	// Store the piecewise-constant functions and CDFs
	std::vector<float> function, cdf;
	float functionIntegral;

	Distribution1D(const std::vector<float>& f, int n) : function(f), cdf(n + 1) {
		// Compute the integral of the piecewise-constant function at xi
		cdf[0] = 0.f;
		for (int i = 1; i < n + 1; i++) {
			cdf[i] = cdf[i - 1] + (function[i - 1] / n);
		}

		// Transform piecewise-constant function integral into CDF
		functionIntegral = cdf[n];
		if (functionIntegral == 0.f) {
			for (int i = 1; i < n + 1; ++i) {
				cdf[i] = (float)i / (float)n;
			}
		} else {
			for (int i = 1; i < n + 1; ++i) {
				cdf[i] /= functionIntegral;
			}
		}
	}

	// Continuous Sampling
	float sampleContinuous(float u, float& pdf, int& offset) {
		// Sample v from binary searching CDF(v)
		offset = binarySearch(u);
		float du = u - cdf[offset];

		if ((cdf[offset + 1] - cdf[offset]) > 0.f) {
			du /= (cdf[offset + 1] - cdf[offset]);
		}

		// Then, sample u from CDF (u|v)
		pdf = (functionIntegral > 0.f) ? (function[offset] / functionIntegral) : 0.f;
		return (offset + du) / function.size();
	}
};

class Distribution2D {
public:
	std::vector<std::unique_ptr<Distribution1D>> pConditionalV;
	std::unique_ptr<Distribution1D> pMarginal;

	Distribution2D(Texture* env) {
		int nu = env->width;
		int nv = env->height;

		std::vector<float> marginalFunction(nv);
		pConditionalV.resize(nv);

		for (int v = 0; v < nv; v++) {
			std::vector<float> function;
			for (int u = 0; u < nu; u++) {
				float luminance = env->texels[(v * nu) + u].Lum();
				float sinTheta = sin(3.14159265358979323846 * ((v + 0.5f) / nv));
				function.push_back(luminance * sinTheta);
			}
			pConditionalV[v].reset(new Distribution1D(function, nu));
			marginalFunction[v] = pConditionalV[v]->functionIntegral;
		}
		pMarginal.reset(new Distribution1D(marginalFunction, nv));
	}

	void sample(float ru, float rv, float& u, float& v, float& pdf) {
		float pdfu = 0.f, pdfv = 0.f;
		int du = 0, dv = 0;

		// Continuous Sampling calculates u and v co-ordinates
		v = pMarginal->sampleContinuous(rv, pdfv, dv);
		u = pConditionalV[dv]->sampleContinuous(ru, pdfu, du);

		// Combined PDF is multiplication of the computed PDFs
		pdf = pdfu * pdfv;
	}

	float PDF(float u, float v) const {
		if (pMarginal->functionIntegral == 0.f) return 1.f;
		int iu = std::min(std::max((int)(u * pConditionalV[0]->function.size()), 0), (int)(pConditionalV[0]->function.size() - 1));
		int iv = std::min(std::max((int)(v * pMarginal->function.size()), 0), (int)(pMarginal->function.size() - 1));
		return pConditionalV[iv]->function[iu] / pMarginal->functionIntegral;
	}
};