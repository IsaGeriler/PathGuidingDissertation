#pragma once

#include "MyMath.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"

#pragma warning(disable : 4244)  // Stop warnings about M_PI being a double

class SceneBounds {
public:
	Vec4 sceneCentre;
	float sceneRadius;
};

class Light {
public:
	virtual Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const Vec4& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec4& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec4 normal(const ShadingData& shadingData, const Vec4& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec4 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec4 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
};

class AreaLight : public Light {
public:
	// Attributes
	Triangle* triangle = nullptr;
	Colour emission;

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) {
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}

	Colour evaluate(const Vec4& wi) {
		if (Dot(wi, triangle->gNormal()) < 0.f) return emission;
		return Colour(0.f, 0.f, 0.f);
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		return 1.f / triangle->area;
	}

	bool isArea() {
		return true;
	}

	Vec4 normal(const ShadingData& shadingData, const Vec4& wi) {
		return triangle->gNormal();
	}

	float totalIntegratedPower() {
		return (triangle->area * emission.Lum());
	}
	
	Vec4 samplePositionFromLight(Sampler* sampler, float& pdf) {
		return triangle->sample(sampler, pdf);
	}

	Vec4 sampleDirectionFromLight(Sampler* sampler, float& pdf) {
		// Samples direction from the area light
		Vec4 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wi);
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
	}
};

class BackgroundColour : public Light {
public:
	// Attributes
	Colour emission;

	// Constructor
	BackgroundColour(Colour _emission) {
		emission = _emission;
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		Vec4 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}

	Colour evaluate(const Vec4& wi) {
		return emission;
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		return SamplingDistributions::uniformSpherePDF(wi);
	}

	bool isArea() {
		return false;
	}

	Vec4 normal(const ShadingData& shadingData, const Vec4& wi) {
		return -wi;
	}

	float totalIntegratedPower() {
		return emission.Lum() * 4.f * M_PI;
	}

	Vec4 samplePositionFromLight(Sampler* sampler, float& pdf) {
		Vec4 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}

	Vec4 sampleDirectionFromLight(Sampler* sampler, float& pdf) {
		Vec4 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light {
public:
	// Attributes
	Texture* env;
	Distribution2D* tabulatedSampling;

	// Constructor
	EnvironmentMap(Texture* _env) {
		// Construct tabulated sampling for Environment Map
		env = _env;
		tabulatedSampling = new Distribution2D(env);
	}

	// Destructor
	~EnvironmentMap() {
		if (env != nullptr) delete env;
		if (tabulatedSampling != nullptr) delete tabulatedSampling;
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Sample u, and v
		float u = 0.f, v = 0.f;
		float sampledPdf = 0.f;

		float ru = sampler->next();
		float rv = sampler->next();
		tabulatedSampling->sample(ru, rv, u, v, sampledPdf);

		// Convert to Spherical Coordinates
		float theta = 3.14159265358979323846 * v;
		float phi = 2 * 3.14159265358979323846 * u;
		
		// Convert to direction (y-up) - (cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta))
		float cosTheta = cos(theta); float cosPhi = cos(phi);
		float sinTheta = sin(theta); float sinPhi = sin(phi);
		Vec4 wi(cosPhi * sinTheta, cosTheta, sinPhi * sinTheta);
		
		// Get PDF and reflected colour, return direction
		pdf = (sinTheta <= 0.f) ? 0.f : sampledPdf / (2 * 3.14159265358979323846 * 3.14159265358979323846 * sinTheta);
		reflectedColour = (pdf <= 0.f) ? Colour(0.f, 0.f, 0.f) : evaluate(wi);
		return wi;
	}

	Colour evaluate(const Vec4& wi) {
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.f) ? u + (2.f * M_PI) : u;
		u = u / (2.f * M_PI);
		float v = acosf(wi.y) / M_PI;
		return env->sample(u, v);
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Obtain u and v
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.f) ? u + (2.f * M_PI) : u;
		u = u / (2.f * M_PI);
		float v = acosf(wi.y) / M_PI;

		// Calculate sinTheta = sin(PI * v) and return PDF
		float sinTheta = sin(3.14159265358979323846 * v);
		return (sinTheta <= 0.f) ? 0.f : tabulatedSampling->PDF(u, v) / (2 * 3.14159265358979323846 * 3.14159265358979323846 * sinTheta);
	}

	bool isArea() {
		return false;
	}

	Vec4 normal(const ShadingData& shadingData, const Vec4& wi) {
		return -wi;
	}

	float totalIntegratedPower() {
		float total = 0.f;
		for (int i = 0; i < env->height; i++) {
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++) {
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		return total * 4.f * M_PI;
	}

	Vec4 samplePositionFromLight(Sampler* sampler, float& pdf) {
		// Samples a point on the bounding sphere of the scene. Feel free to improve this.
		Vec4 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 1.f / (4.f * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}

	Vec4 sampleDirectionFromLight(Sampler* sampler, float& pdf) {
		// Tabulated sampling of environment maps
		ShadingData shadingData;
		Colour reflectedColour;
		Vec4 wi = sample(shadingData, sampler, reflectedColour, pdf);
		return -wi;
	}
};