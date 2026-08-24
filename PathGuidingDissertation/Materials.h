#pragma once

#include "MyMath.h"
#include "Imaging.h"
#include "Sampling.h"

#pragma warning(disable : 4244)  // Stop warnings about M_PI being a double
#pragma warning(disable : 4305)  // Double to float

class BSDF;

class ShadingData {
public:
	// Attributes
	Vec4 x, wo;
	Vec4 sNormal, gNormal;

	float tu, tv;
	float t;

	Frame frame;
	BSDF* bsdf;
	
	// Constructors
	ShadingData() {}
	ShadingData(Vec4 _x, Vec4 n) {
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = nullptr;
	}
};

class ShadingHelper {
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt) {
		// Calculate eta based on incoming cosine
		float cosThetaI = std::max(std::min(cosTheta, 1.f), -1.f);
		float eta = (cosThetaI < 0.f) ? (iorInt / iorExt) : (iorExt / iorInt);
		if (cosThetaI < 0.f) cosThetaI = fabs(cosThetaI);
		
		// Calculate transmission cosine from incoming cosine
		float etaSq = eta * eta;
		float cosThetaISq = std::max(cosThetaI * cosThetaI, 0.f);
		float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);
		float sinThetaTSq = etaSq * sinThetaISq;

		// Total Internal Reflection
		if (sinThetaTSq >= 1.f) return 1.f;
		float cosThetaT = sqrtf(std::max(1.f - sinThetaTSq, 0.f));

		// Fresnel Parallel and Perpendicular
		float parallel = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
		float perpendicular = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);

		float parallelSq = parallel * parallel;
		float perpendicularSq = perpendicular * perpendicular;

		// Return Fresnel Average
		return (parallelSq + perpendicularSq) * 0.5f;
	}

	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k) {
		// Calculate sinThetaI from cosThetaI
		float cosThetaI = std::max(std::min(cosTheta, 1.f), -1.f);
		float cosThetaISq = std::max(cosThetaI * cosThetaI, 0.f);
		float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);

		// Get cosThetaI, cosThetaISq, and sinThetaISq Colours
		Colour colCosThetaI(cosThetaI, cosThetaI, cosThetaI);
		Colour colCosThetaISq(cosThetaISq, cosThetaISq, cosThetaISq);
		Colour colSinThetaISq(sinThetaISq, sinThetaISq, sinThetaISq);

		// Fresnel Parallel and Perpendicular
		Colour iorSq = ior * ior;
		Colour kSq = k * k;

		Colour parallelSq = (
			(((iorSq + kSq) * colCosThetaISq) - (ior * colCosThetaI * 2) + colSinThetaISq) / 
			(((iorSq + kSq) * colCosThetaISq) + (ior * colCosThetaI * 2) + colSinThetaISq)
		);

		Colour perpendicularSq = (
			(iorSq + kSq - (ior * colCosThetaI * 2) + colCosThetaISq) /
			(iorSq + kSq + (ior * colCosThetaI * 2) + colCosThetaISq)
		);

		// Return Fresnel Average
		return (parallelSq + perpendicularSq) * 0.5f;
	}

	// Microfacet Model: Isotropic Trowbridge-Reitz (GGX)
	static float lambdaGGX(Vec4 wi, float alpha) {
		if (wi.z <= 0.f) return 0.f;
		float alphaSq = alpha * alpha;
		float cosThetaSq = wi.z * wi.z;
		float sinThetaSq = std::max(1.f - cosThetaSq, 0.f);
		float tanThetaSq = std::fabs(sinThetaSq / cosThetaSq);
		return (sqrtf(1.f + alphaSq * tanThetaSq) - 1.f) * 0.5f;
	}

	// Microfacet Model: Isotropic Trowbridge-Reitz (GGX)
	static float Gggx(Vec4 wi, Vec4 wo, float alpha) {
		// G(wo,wi) = G1(wo,wm) * G1(wi,wm)
		float G1wo = 1.f / (1.f + lambdaGGX(wo, alpha));
		float G1wi = 1.f / (1.f + lambdaGGX(wi, alpha));
		return G1wo * G1wi;
	}

	// Microfacet Model: Isotropic Trowbridge-Reitz (GGX)
	static float Dggx(Vec4 h, float alpha) {
		if (h.z <= 0.f) return 0.f;
		float alphaSq = alpha * alpha;
		float cosThetaMSq = h.z * h.z;
		float denominator = M_PI * (cosThetaMSq * (alphaSq - 1.f) + 1.f) * (cosThetaMSq * (alphaSq - 1.f) + 1.f);
		if (denominator <= 0.f) return 0.f;
		return alphaSq / denominator;
	}
};

class BSDF {
public:
	Colour emission;
	// virtual Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) = 0;
	virtual void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec4& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec4& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isSmooth() = 0;
	virtual bool canLearnFresnel() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight() { return emission.Lum() > 0.f; }
	void addLight(Colour _emission) { emission = _emission; }
	Colour emit(const ShadingData& shadingData, const Vec4& wi) { return emission; }
	virtual float mask(const ShadingData& shadingData) = 0;
};

class DiffuseBSDF : public BSDF {
public:
	// Attributes
	Texture* albedo;

	// Constructors
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo) { albedo = _albedo; }

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Sample incoming direction (z-up coordinate system, local space)
		float r1 = sampler->next();
		float r2 = sampler->next();
		// Discard r3, just for GuidedSampler consistency only (u,v,selectProbability)
		float r3 = sampler->next();
		Vec4 wi = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		// Convert wi to world space before passing to evaluate and pdf
		wi = shadingData.frame.toWorld(wi);
		
		// Evaluate BSDF and PDF
		reflectedColour = evaluate(shadingData, wi);  // BSDF = albedo / PI
		pdf = PDF(shadingData, wi);					  // PDF = cos(theta) / PI

		// Return incoming direction
		return wi;
	}

	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Convert wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z <= 0.f) { u = -1.f; v = -1.f; selectProbability = 0.f; return; }

		// Retrieve phi from wiLocal
		float phi = SphericalCoordinates::sphericalPhi(wiLocal);

		// Invert Cosine Sample Hemisphere to find u and v
		// theta = acos(sqrt(r1)), phi = 2 * PI * r2
		u = wiLocal.z * wiLocal.z;
		v = phi / (2.f * M_PI);

		// Clamp u and v to [0, 1)
		u = std::max(0.f, std::min(u, 0.99999f));
		v = std::max(0.f, std::min(v, 0.99999f));
		selectProbability = std::max(EPSILON, std::min(sampler->next(), 0.99999f));
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z <= 0.f) return Colour(0.f, 0.f, 0.f);
		return albedo->sample(shadingData.tu, shadingData.tv) * 0.318309886183790671538;
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}

	bool isPureSpecular() { return false; }
	bool isSmooth() { return false; }
	bool canLearnFresnel() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

class MirrorBSDF : public BSDF {
public:
	// Attributes
	Texture* albedo;

	// Constructors
	MirrorBSDF() = default;
	MirrorBSDF(Texture* _albedo) { albedo = _albedo; }

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Convert shadingData.wo to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Reflect x and y of woLocal
		Vec4 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);

		// Guard case to prevent division by zero
		if (fabsf(wrLocal.z) <= EPSILON) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }

		// Generate random numbers (this is for BSDF Inversion and testing it)
		float dummy_r1 = sampler->next();
		float dummy_r2 = sampler->next();
		float dummy_selectProbability = sampler->next();
		
		// Convert back to world space and return
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / wrLocal.z;  // BSDF = albedo / Dot(wr, n)
		pdf = 1.f;																	   // PDF = 1 (for perfect specular reflection)
		return shadingData.frame.toWorld(wrLocal);
	}

	// MirrorBSDF will not be inverted as this is due to Dirac Delta distribution
	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		u = 0.f;
		v = 0.f;
		selectProbability = std::max(EPSILON, std::min(sampler->next(), 0.99999f));
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		return Colour(0.f, 0.f, 0.f);
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		return 0.f;
	}

	bool isPureSpecular() { return true; }
	bool isSmooth() { return true; }
	bool canLearnFresnel() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

class ConductorBSDF : public BSDF {
public:
	// Attributes
	Texture* albedo;
	Colour eta, k;
	float alpha;

	// Constructors
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness) {
		albedo = _albedo;
		eta = _eta;
		k = _k;
		// Swapping alpha from engine default to Disney mapping
		// alpha = 1.62142f * sqrtf(roughness);
		alpha = std::max(SQ(roughness), 0.001f);
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Convert shadingData.wo to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		
		// Can sample only visible normals
		if (woLocal.z <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }

		// If alpha is less than epsilon, treat it as a mirror with fresnel
		if (alpha < EPSILON) {
			Vec4 wr(-woLocal.x, -woLocal.y, woLocal.z);
			Colour fresnel = ShadingHelper::fresnelConductor(wr.z, eta, k);
			reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wr.z)) * fresnel;
			pdf = 1.f;
			return shadingData.frame.toWorld(wr);
		}

		// Sample phi and theta for sampling the half vector
		// Discard r3, just for GuidedSampler consistency only (u, v, selectProbability)
		float alphaSq = alpha * alpha;
		float r1 = sampler->next();
		float r2 = sampler->next();
		float r3 = sampler->next();

		float cosThetaSq = (1.f - r1) / (r1 * (alphaSq - 1.f) + 1.f);
		float cosTheta = std::max(0.f, std::min(sqrtf(cosThetaSq), 1.f));
		float thetaM = acosf(cosTheta);
		float phiM = 2.f * M_PI * r2;

		// Find wi from wo and wm (light reflected across microfacet model)
		Vec4 wmLocal = SphericalCoordinates::sphericalToWorld(thetaM, phiM);
		Vec4 wiLocal = -woLocal + wmLocal * 2.f * Dot(wmLocal, woLocal);

		if (wiLocal.z <= 0.f || Dot(wmLocal, woLocal) <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }

		Vec4 wi = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wi);
		pdf = PDF(shadingData, wi);
		if (sampledGlossy != nullptr) *sampledGlossy = alpha < 0.12f;
		return wi;
	}

	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Convert wo and wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Can sample only visible normals
		if (wiLocal.z <= 0.f || woLocal.z <= 0.f) { u = -1.f; v = -1.f; selectProbability = 0.f; return; }

		// If alpha is less than epsilon, treat it as a MirrorBSDF
		if (alpha < EPSILON) { u = 0.f; v = 0.f; selectProbability = 0.f; return; }

		// Obtain the half vector from wi and wo
		Vec4 wmLocal = (wiLocal + woLocal).normalize();

		// Can only sample from visible normals
		if (wmLocal.lengthSquare() < EPSILON) { u = -1.f; v = -1.f; selectProbability = 0.f; return; }
		wmLocal = wmLocal.normalize();

		// Get phi from the half vector
		float phi = SphericalCoordinates::sphericalPhi(wmLocal);
		
		// v is the same as phi = 2 * PI * v
		// However for u, use it's CDF as we used CDF inversion for sampling!
		// u = (alphaSq / (SQ(cos(theta)) * (alphaSqMinusOne * alphaSqMinusOne) + alphaSqMinusOne)) - (1.f / alphaSqMinusOne);
		float alphaSq = alpha * alpha;
		float alphaSqMinusOne = (alphaSq - 1.f);
		float cosThetaSq = wmLocal.z * wmLocal.z;

		u = (1.f - cosThetaSq) / (cosThetaSq * alphaSqMinusOne + 1.f);
		v = phi / (2 * M_PI);

		// Clamp u and v to [0, 1)
		u = std::max(0.f, std::min(u, 0.99999f));
		v = std::max(0.f, std::min(v, 0.99999f));
		selectProbability = std::max(EPSILON, std::min(sampler->next(), 0.99999f));
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Convert shadingData.wo and wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Can sample only visible normals from wo
		if (wiLocal.z <= 0.f || woLocal.z <= 0.f) return Colour(0.f, 0.f, 0.f);

		// If alpha is less than epsilon, treat as a mirror
		if (alpha < EPSILON) return Colour(0.f, 0.f, 0.f);
		
		// Half Vector
		Vec4 wmLocal = wiLocal + woLocal;
		if (wmLocal.lengthSquare() < EPSILON) return Colour(0.f, 0.f, 0.f);
		wmLocal = wmLocal.normalize();
		if (Dot(woLocal, wmLocal) <= 0.f) return Colour(0.f, 0.f, 0.f);
		if (wiLocal.z * woLocal.z <= 0.f) return Colour(0.f, 0.f, 0.f);

		// Cook-Torrance BRDF
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float D = ShadingHelper::Dggx(wmLocal, alpha);
		Colour F = ShadingHelper::fresnelConductor(Dot(woLocal, wmLocal), eta, k);
		return albedo->sample(shadingData.tu, shadingData.tv) * ((F * G * D) / (4.f * wiLocal.z * woLocal.z));
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Convert shadingData.wo and wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Can sample only visible normals from wo
		if (wiLocal.z <= 0.f || woLocal.z <= 0.f) return 0.f;

		// If alpha is less than epsilon, treat as a mirror
		if (alpha < EPSILON) return 0.f;

		// Half Vector
		Vec4 wmLocal = wiLocal + woLocal;
		if (wmLocal.lengthSquare() < EPSILON) return 0.f;
		wmLocal = wmLocal.normalize();
		if (Dot(woLocal, wmLocal) <= 0.f) return 0.f;
		if (wiLocal.z * woLocal.z <= 0.f) return 0.f;

		float D = ShadingHelper::Dggx(wmLocal, alpha);
		return (D * wmLocal.z) / (4.f * Dot(woLocal, wmLocal));
	}

	bool isPureSpecular() { return alpha < EPSILON; }
	bool isSmooth() { return alpha < 0.12f; }
	bool canLearnFresnel() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

class GlassBSDF : public BSDF {
public:
	// Attributes
	Texture* albedo;
	float intIOR, extIOR;

	// Constructors
	GlassBSDF() = default;
	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR) {
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Convert shadingData.wo to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		
		// Cosine term guard case
		if (fabsf(woLocal.z) < EPSILON) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }

		// Fresnel is the probability of the material reflecting
		float fresnel = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		float cosThetaI = woLocal.z;
		float eta = (cosThetaI < 0.f) ? (intIOR / extIOR) : (extIOR / intIOR);
		if (cosThetaI < 0.f) cosThetaI = fabsf(cosThetaI);

		// Generate random numbers (this is for BSDF Inversion and testing it)
		float dummy_r1 = sampler->next();
		float dummy_r2 = sampler->next();
		float selectProbability = sampler->next();

		// Generate a random number to decide reflect of refract
		if (selectProbability < fresnel) {
			// Reflect
			Vec4 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
			pdf = fresnel;
			reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wrLocal.z)) * fresnel;
			return shadingData.frame.toWorld(wrLocal);
		} else {
			// Transmit
			// Calculate cosThetaT
			float cosThetaISq = std::max(cosThetaI * cosThetaI, 0.f);
			float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);
			float etaSq = eta * eta;
			float sinThetaTSq = etaSq * sinThetaISq;

			// Total Internal Reflection
			if (sinThetaTSq >= 1.f) {
				// Perfect Specular Reflection - Treat as mirror with fresnel
				Vec4 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
				pdf = fresnel;
				reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wrLocal.z)) * fresnel;
				return shadingData.frame.toWorld(wrLocal);
			}

			// Refract
			float cosThetaT = sqrtf(std::max(1.f - sinThetaTSq, 0.f));
			float wtZ = (woLocal.z < 0.f) ? cosThetaT : -cosThetaT;
			Vec4 wtLocal(-eta * woLocal.x, -eta * woLocal.y, wtZ);
			
			pdf = 1.f - fresnel;
			reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wtLocal.z)) * (1.f - fresnel) * etaSq;
			return shadingData.frame.toWorld(wtLocal);
		}
	}

	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Convert wo and wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Calculate Fresnel and determine if we are reflecting or refracting
		float F = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		bool reflect = woLocal.z * wiLocal.z > 0.f;

		// GlassBSDF will not be inverted as this is due to Dirac Delta distribution
		// However, we can save the Fresnel probability for the next bounce
		u = 0.f;
		v = 0.f;
		float r3 = std::max(EPSILON, std::min(sampler->next(), 0.99999f));
		selectProbability = reflect ? (F * r3) : (F + ((1.f - F) * r3));
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		return Colour(0.f, 0.f, 0.f);
	}
	
	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		return 0.f;
	}

	bool isPureSpecular() { return true; }
	bool isSmooth() { return true; }
	bool canLearnFresnel() { return true; }
	bool isTwoSided() { return false; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

class DielectricBSDF : public BSDF {
public:
	// Attributes
	Texture* albedo;
	float intIOR, extIOR;
	float alpha;

	// Constructors
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness) {
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		// Swapping alpha from engine default to Disney mapping
		// alpha = 1.62142f * sqrtf(roughness);
		alpha = std::max(SQ(roughness), 0.001f);
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Convert shadingData.wo to the local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Random numbers for sampling
		float r1 = sampler->next();
		float r2 = sampler->next();
		float selectProbability = sampler->next();
		
		// When alpha < EPSILON, we treat this as GlassBSDF
		if (alpha < EPSILON) {
			// Calculate fresnel to determine if we should reflect or refract
			float fresnel = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
			if (selectProbability < fresnel) {
				// Reflect
				Vec4 wiLocal(-woLocal.x, -woLocal.y, woLocal.z);
				pdf = fresnel;
				reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wiLocal.z)) * fresnel;
				return shadingData.frame.toWorld(wiLocal);
			}
			else {
				// Refract
				float eta = woLocal.z > 0.f ? extIOR / intIOR : intIOR / extIOR;
				float etaSq = eta * eta;
				float sign = woLocal.z > 0.f ? -1.f : 1.f;

				float cosThetaISq = std::max(woLocal.z * woLocal.z, 0.f);
				float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);
				float sinThetaTSq = etaSq * sinThetaISq;

				// Total Internal Reflection
				if (sinThetaTSq >= 1.f) {
					// Perfect Specular Reflection
					// Treat as mirror with fresnel
					Vec4 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
					pdf = 1.f;
					reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wrLocal.z)) * fresnel;
					return shadingData.frame.toWorld(wrLocal);
				}

				float cosThetaT = sqrt(std::max(1.f - sinThetaTSq, 0.f));
				Vec4 wiLocal(-eta * woLocal.x, -eta * woLocal.y, cosThetaT * sign);
				pdf = 1.f - fresnel;
				reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wiLocal.z)) * (1.f - fresnel);
				return shadingData.frame.toWorld(wiLocal);
			}
		}

		// Sample half vector
		float alphaSq = alpha * alpha;
		float cosThetaSq = (1.f - r1) / (r1 * (alphaSq - 1.f) + 1.f);
		float cosTheta = std::max(0.f, std::min(sqrtf(cosThetaSq), 1.f));
		float theta = acosf(cosTheta);
		float phi = 2.f * M_PI * r2;

		Vec4 wmLocal = SphericalCoordinates::sphericalToWorld(theta, phi);

		// Calculate Fresnel Term
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		if (selectProbability < F) {
			// Reflect
			Vec4 wiLocal = -woLocal + (wmLocal * 2 * Dot(wmLocal, woLocal));
			if (wiLocal.z * woLocal.z <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
			Vec4 wi = shadingData.frame.toWorld(wiLocal);
			reflectedColour = evaluate(shadingData, wi);
			pdf = PDF(shadingData, wi);
			return wi;
		} else {
			// Refract
			bool isInside = woLocal.z < 0;
			float eta = !isInside ? (extIOR / intIOR) : (intIOR / extIOR);
			if (isInside) wmLocal = -wmLocal;
			
			float cosThetaI = std::min(std::max(Dot(wmLocal, woLocal), -1.f), 1.f);
			float etaSq = eta * eta;

			float cosThetaISq = std::max(cosThetaI * cosThetaI, 0.f);
			float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);
			float sinThetaTSq = etaSq * sinThetaISq;

			if (sinThetaTSq >= 1.f) {
				// Total Internal Reflection
				// Vec4 wr = shadingData.frame.toWorld(Vec4(-woLocal.x, -woLocal.y, woLocal.z));
				Vec4 wrLocal = -woLocal + (wmLocal * 2.f * Dot(wmLocal, woLocal));
				Vec4 wr = shadingData.frame.toWorld(wrLocal);
				pdf = F;
				reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * F / fabs(woLocal.z);
				return wr;
			}

			float cosThetaT = sqrt(std::max(1.f - sinThetaTSq, 0.f));
			Vec4 wiLocal = -woLocal * eta + wmLocal * (eta * cosThetaI - cosThetaT);
			if (wiLocal.lengthSquare() < EPSILON) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
			wiLocal = wiLocal.normalize();
			Vec4 wi = shadingData.frame.toWorld(wiLocal);
			reflectedColour = evaluate(shadingData, wi);
			pdf = PDF(shadingData, wi);
			if (sampledGlossy != nullptr) *sampledGlossy = alpha < 0.12f;
			return wi;
		}
	}

	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Convert wi and wo to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Save cosine terms
		float cosThetaO = woLocal.z;
		float cosThetaI = wiLocal.z;

		// Determine reflect according to PBRT
		// https://pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF
		bool reflect = cosThetaI * cosThetaO > 0.f;

		// When alpha < EPSILON, we treated this as GlassBSDF
		float r3 = std::max(EPSILON, std::min(sampler->next(), 0.99999f));
		if (alpha < EPSILON) {
			float fresnel = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
			u = 0.f;
			v = 0.f;
			selectProbability = reflect ? (fresnel * r3) : (fresnel + ((1.f - fresnel) * r3));
			return;
		}

		// Calculate eta
		float eta_o = cosThetaO > 0.f ? extIOR : intIOR;
		float eta_i = cosThetaI > 0.f ? extIOR : intIOR;
		float eta = reflect ? 1.f : eta_i / eta_o;

		// Retrieve the half-vector
		Vec4 wmLocal = wiLocal * eta + woLocal;;

		// Can only sample from visible normals
		if (cosThetaO == 0.f || cosThetaI == 0.f || wmLocal.lengthSquare() == 0.f) { u = -1.f; v = -1.f; selectProbability = 0.f; return; }
		wmLocal = wmLocal.normalize();

		// Face forward
		if (wmLocal.z < 0.f) wmLocal = -wmLocal;

		// thetaM and phiM are obtained from the half vector
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		float theta = SphericalCoordinates::sphericalTheta(wmLocal);
		float phi = SphericalCoordinates::sphericalPhi(wmLocal);

		// v is the same as phi = 2 * PI * v
		// However for u, use it's CDF as we used CDF inversion for sampling!
		// u = (alphaSq / (SQ(cos(theta)) * (alphaSqMinusOne * alphaSqMinusOne) + alphaSqMinusOne)) - (1.f / alphaSqMinusOne);
		float alphaSq = alpha * alpha;
		float alphaSqMinusOne = (alphaSq - 1.f);
		float cosThetaSq = wmLocal.z * wmLocal.z;
		u = (1.f - cosThetaSq) / (cosThetaSq * alphaSqMinusOne + 1.f);
		v = phi / (2 * M_PI);

		// Clamp u and v to [0, 1)
		u = std::max(0.f, std::min(u, 0.99999f));
		v = std::max(0.f, std::min(v, 0.99999f));
		selectProbability = reflect ? (F * r3) : (F + ((1.f - F) * r3));
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wo and wi to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wiLocal = shadingData.frame.toLocal(wi);

		// If alpha is less than epsilon, treat as a mirror
		if (alpha < EPSILON) return Colour(0.f, 0.f, 0.f);

		// Save cosine terms
		float cosThetaO = woLocal.z;
		float cosThetaI = wiLocal.z;

		// Determine reflect according to PBRT
		// https://pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF
		bool reflect = cosThetaI * cosThetaO > 0.f;
		float eta_o = cosThetaO > 0.f ? extIOR : intIOR;
		float eta_i = cosThetaI > 0.f ? extIOR : intIOR;
		float eta = reflect ? 1.f : eta_i / eta_o;
		
		// Generalized half-vector
		// Can only sample from visible normals
		Vec4 wmLocal = (wiLocal * eta) + woLocal;
		if (cosThetaO == 0.f || cosThetaI == 0.f || wmLocal.lengthSquare() == 0.f) return Colour(0.f, 0.f, 0.f);
		wmLocal = wmLocal.normalize();
		
		// Face forward
		if (wmLocal.z < 0.f) wmLocal = -wmLocal;

		// Discard backfacing microfacets
		if (Dot(wmLocal, wiLocal) * cosThetaI <= 0.f || Dot(wmLocal, woLocal) * cosThetaO <= 0.f) return Colour(0.f, 0.f, 0.f);
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float D = ShadingHelper::Dggx(wmLocal, alpha);

		// Cook-Torrance BRDF
		if (reflect) {
			// BSDF evaluation from ConductorBSDF
			return albedo->sample(shadingData.tu, shadingData.tv) * ((G * D * F) / (fabs(4.f * cosThetaI * cosThetaO)));
		} else {
			float denominator = (Dot(wiLocal, wmLocal) + Dot(woLocal, wmLocal) / eta);
			float denominatorSq = denominator * denominator;
			if (denominatorSq == 0.f) return Colour(0.f, 0.f, 0.f);

			// BSDF evaluation of Microfacet Refraction
			float partOne = (G * D * (1.f - F)) / denominatorSq;
			float partTwo = fabs(Dot(wiLocal, wmLocal)) / fabs(cosThetaI);
			float partThree = fabs(Dot(woLocal, wmLocal)) / fabs(cosThetaO);
			return albedo->sample(shadingData.tu, shadingData.tv) * (partOne * partTwo * partThree);
		}
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wo and wi to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wiLocal = shadingData.frame.toLocal(wi);

		// If alpha is less than epsilon, treat as a mirror
		if (alpha < EPSILON) return 0.f;

		// Save cosine terms
		float cosThetaO = woLocal.z;
		float cosThetaI = wiLocal.z;

		// Determine reflect according to PBRT
		// https://pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF
		bool reflect = cosThetaI * cosThetaO > 0.f;
		float eta_o = cosThetaO > 0.f ? extIOR : intIOR;
		float eta_i = cosThetaI > 0.f ? extIOR : intIOR;
		float eta = reflect ? 1.f : eta_i / eta_o;

		// Generalized half-vector
		// Can only sample from visible normals
		Vec4 wmLocal = (wiLocal * eta) + woLocal;
		if (cosThetaO == 0.f || cosThetaI == 0.f || wmLocal.lengthSquare() == 0.f) return 0.f;
		wmLocal = wmLocal.normalize();

		// Face forward
		if (wmLocal.z < 0.f) wmLocal = -wmLocal;

		// Discard backfacing microfacets
		if (Dot(wmLocal, wiLocal) * cosThetaI <= 0.f || Dot(wmLocal, woLocal) * cosThetaO <= 0.f) return 0.f;
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		float D = ShadingHelper::Dggx(wmLocal, alpha);
		if (reflect) {
			if (Dot(woLocal, wmLocal) == 0.f) return 0.f;
			// PDF evaluation from ConductorBSDF
			return F * ((D * wmLocal.z) / (4.f * fabs(Dot(woLocal, wmLocal))));
		} else {
			float denominator = (Dot(wiLocal, wmLocal) + Dot(woLocal, wmLocal) / eta);
			float denominatorSq = denominator * denominator;
			if (denominatorSq == 0.f) return 0.f;
			// PDF evaluation of Microfacet Refraction
			float p_wi = (D * fabs(Dot(woLocal, wmLocal))) / denominatorSq;
			return (1.f - F) * p_wi;
		}
	}

	bool isPureSpecular() { return alpha < EPSILON; }
	bool isSmooth() { return alpha < 0.12f; }
	bool canLearnFresnel() { return true; }
	bool isTwoSided() { return false; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

class OrenNayarBSDF : public BSDF {
public:
	// Attributes
	Texture* albedo;
	float sigma;

	// Constructors
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma) {
		albedo = _albedo;
		sigma = _sigma;
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Sample incoming direction (z-up coordinate system, local space)
		// Using DiffuseBSDF's cosine weighted sampling
		// Discard r3, just for GuidedSampler consistency only (u, v, selectProbability)
		float r1 = sampler->next();
		float r2 = sampler->next();
		float r3 = sampler->next();
		Vec4 wi = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		// Convert wi to world space before passing to evaluate and pdf
		wi = shadingData.frame.toWorld(wi);

		// Evaluate BSDF and PDF
		reflectedColour = evaluate(shadingData, wi);  // BSDF = Diffuse * OrenNayar
		pdf = PDF(shadingData, wi);					  // PDF = cos(theta) / PI

		// Return incoming direction
		return wi;
	}

	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Invert Cosine Sample Hemisphere where,
		// theta = acos(sqrt(r1)), phi = 2 * PI * r2
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z <= 0.f) { u = -1.f; v = -1.f; selectProbability = 0.f; return; }
		
		float phi = SphericalCoordinates::sphericalPhi(wiLocal);
		
		// u = SQ(cosf(theta));
		u = wiLocal.z * wiLocal.z;
		v = phi / (2.f * M_PI);

		// Clamp u and v to [0, 1)
		u = std::max(0.f, std::min(u, 0.99999f));
		v = std::max(0.f, std::min(v, 0.99999f));
		selectProbability = std::max(EPSILON, std::min(sampler->next(), 0.99999f));
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wiLocal = shadingData.frame.toLocal(wi);

		// Oren-Nayar Constants
		const float sigmaSq = sigma * sigma;
		const float A = 1.f - (sigmaSq / (2 * (sigmaSq + 0.33f)));
		const float B = (0.45f * sigmaSq) / (sigmaSq + 0.09f);

		// Calculate theta and phi for wi and wo
		float phiI = SphericalCoordinates::sphericalPhi(wiLocal);
		float phiO = SphericalCoordinates::sphericalPhi(woLocal);

		float thetaI = SphericalCoordinates::sphericalTheta(wiLocal);
		float thetaO = SphericalCoordinates::sphericalTheta(woLocal);

		// Store trigonometric terms
		float cosine = std::cosf(phiI - phiO);
		float sinAlpha = std::sinf(std::max(thetaI, thetaO));
		float tanBeta = std::tanf(std::min(thetaI, thetaO));

		// OrenNayar Approximation
		const float OrenNayarApproximation = A + (B * std::max(0.f, cosine) * sinAlpha * tanBeta);

		// BSDF = (albedo / PI) * OrenNayarApprox
		// Note - (albedo / PI) is BSDF evaluation of Diffuse
		return albedo->sample(shadingData.tu, shadingData.tv) * M_1_PI * OrenNayarApproximation;
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Using DiffuseBSDF's cosine hemisphere PDF
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}

	bool isPureSpecular() { return false; }
	bool isSmooth() { return false; }
	bool canLearnFresnel() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

// Using Torrance-Sparrow Microfacet Model with GGX Distribution for Plastic Material
class PlasticBSDF : public BSDF {
public:
	// Attributes
	Texture* albedo;
	float intIOR, extIOR;
	float alpha;

	// Constructors
	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness) {
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		// Swapping alpha from engine default to Disney mapping
		// alpha = 1.62142f * sqrtf(roughness);
		alpha = std::max(SQ(roughness), 0.001f);
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Get outcoming direction
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wiLocal(0.f, 0.f, 0.f);

		// Random numbers for sampling
		float r1 = sampler->next();
		float r2 = sampler->next();
		float selectProbability = sampler->next();

		// Fresnel to compute diffuse or reflect surface
		float fresnel = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		if (selectProbability < fresnel) {
			// Glossy Part - Sample theta and phi from random variables for half vector
			float alphaSq = alpha * alpha;
			float alphaSqMinusOne = alphaSq - 1.f;
			float cosThetaSq = (1.f - r1) / (r1 * (alphaSq - 1.f) + 1.f);
			float cosTheta = std::max(0.f, std::min(sqrtf(cosThetaSq), 1.f));
			float theta = acosf(cosTheta);
			float phi = 2.f * M_PI * r2;

			// Converting this to GGX sampling for half vector
			Vec4 hLocal = SphericalCoordinates::sphericalToWorld(theta, phi);
			if (hLocal.lengthSquare() < EPSILON) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
			hLocal = hLocal.normalize();

			// Flip if it's below the surface
			if (woLocal.z <= 0.f) hLocal = -hLocal;
			wiLocal = hLocal * 2.f * Dot(woLocal, hLocal) - woLocal;
			if (wiLocal.z <= 0.f || Dot(woLocal, hLocal) <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
			if (sampledGlossy != nullptr) *sampledGlossy = alpha < 0.12f;
		} else {
			// Diffuse Part - Sample wi with cosine hemisphere
			wiLocal = SamplingDistributions::cosineSampleHemisphere(r1, r2);
			// Flip if it's below the surface
			if (woLocal.z <= 0.f) wiLocal = -wiLocal;
			if (sampledGlossy != nullptr) *sampledGlossy = false;
		}
		Vec4 wi = shadingData.frame.toWorld(wiLocal);
		pdf = PDF(shadingData, wi);
		if (pdf < EPSILON) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
		reflectedColour = evaluate(shadingData, wi);
		return wi;
	}

	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Convert wi and wo to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Calculate the half vector and face it forward
		Vec4 hLocal = wiLocal + woLocal;
		if (hLocal.lengthSquare() < EPSILON) { u = -1.f; v = -1.f; selectProbability = 0.f; return; }
		hLocal = hLocal.normalize();
		if (hLocal.z < 0.f) hLocal = -hLocal;

		// Calculate PDFs and the weights to determine if we gloss or diffuse
		float F = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		float pdf_diffuse = SamplingDistributions::cosineHemispherePDF(wiLocal);

		float D = ShadingHelper::Dggx(hLocal, alpha);
		float pdf_glossy = D * (hLocal.z) / (4.f * std::max(Dot(woLocal, hLocal), EPSILON));

		float w_glossy = F * pdf_glossy;
		float w_diffuse = (1.f - F) * pdf_diffuse;
		if (w_glossy + w_diffuse < EPSILON) { u = -1.f; v = -1.f; selectProbability = 0.f; }

		// Invert the half vector sampling for GGX distribution
		bool isGlossy = w_glossy > w_diffuse;
		
		float r3 = std::max(EPSILON, std::min(sampler->next(), 0.99999f));

		if (isGlossy) {
			float alphaSq = alpha * alpha;
			float alphaSqMinusOne = (alphaSq - 1.f);

			// Retrieve phi from hLocal
			float phi = SphericalCoordinates::sphericalPhi(hLocal);
			float cosThetaHSq = hLocal.z * hLocal.z;

			u = (1.f - cosThetaHSq) / (cosThetaHSq * alphaSqMinusOne + 1.f);
			v = phi / (2 * M_PI);

			// Clamp u and v to [0, 1)
			u = std::max(0.f, std::min(u, 0.99999f));
			v = std::max(0.f, std::min(v, 0.99999f));
			selectProbability = F * r3;
			return;
		}
		// Invert the diffuse part using cosine hemisphere sampling
		// Retrieve phi from wiLocal
		float phi = SphericalCoordinates::sphericalPhi(wiLocal);
		float cosThetaSq = wiLocal.z * wiLocal.z;

		u = cosThetaSq;
		v = phi / (2.f * M_PI);

		// Clamp u and v to [0, 1)
		u = std::max(0.f, std::min(u, 0.99999f));
		v = std::max(0.f, std::min(v, 0.99999f));
		selectProbability = F + ((1.f - F) * r3);
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Calculate half vector from wi and wo
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z * woLocal.z <= 0.f) return Colour(0.f, 0.f, 0.f);

		// Get half vector and lenght check for half veector
		Vec4 hLocal = wiLocal + woLocal;
		if (hLocal.lengthSquare() < EPSILON) return Colour(0.f, 0.f, 0.f);
		hLocal = hLocal.normalize();
		if (Dot(woLocal, hLocal) <= 0.f) return Colour(0.f, 0.f, 0.f);

		// Using GGX Microfacet Model
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float D = ShadingHelper::Dggx(hLocal, alpha);
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, hLocal), intIOR, extIOR);

		float denom = (4.f * wiLocal.z * woLocal.z);
		if (denom < EPSILON) return Colour(0.f, 0.f, 0.f);
		float glossyBSDFComponent = (G * D * F) / denom;
		Colour glossyBSDF = Colour(1.f, 1.f, 1.f) * glossyBSDFComponent;
		Colour diffuseBSDF = albedo->sample(shadingData.tu, shadingData.tv) * M_1_PI * (1.f - F);
		return glossyBSDF + diffuseBSDF;
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Calculate half vector from wi and wo
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (wiLocal.z * woLocal.z <= 0.f) return 0.f;
		
		// Get half vector and lenght check for half veector
		Vec4 hLocal = wiLocal + woLocal;
		if (hLocal.lengthSquare() < EPSILON) return 0.f;
		hLocal = hLocal.normalize();
		if (Dot(woLocal, hLocal) <= 0.f) return 0.f;
		
		// Using GGX Microfacet Model
		float D = ShadingHelper::Dggx(hLocal, alpha);
		float F = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);

		float glossyPDF = F * ((D * (hLocal.z)) / (4.f * std::fabs(Dot(woLocal, hLocal))));
		float diffusePDF = (1.f - F) * SamplingDistributions::cosineHemispherePDF(wiLocal);
		return glossyPDF + diffusePDF;
	}

	bool isPureSpecular() { return false; }
	bool isSmooth() { return false; }
	bool canLearnFresnel() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

class LayeredBSDF : public BSDF {
public:
	// Attributes
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR, extIOR;

	// Constructors
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR) {
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf, bool* sampledGlossy = nullptr) {
		// Add code to include layered sampling
		// return base->sample(shadingData, sampler, reflectedColour, pdf);
		// Convert shaingData.wo and base->sample to local space
		Vec4 wo = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wi = shadingData.frame.toLocal(base->sample(shadingData, sampler, reflectedColour, pdf));
		if (pdf <= 0.f || wi.z == 0.f || wo.z == 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }

		// Handle absorbing medium
		float distWo = thickness / fabs(wo.z);
		float distWi = thickness / fabs(wi.z);

		// Beer's Law for Absorption
		Colour TrWo(powf(M_E, -distWo * sigmaa.r), powf(M_E, -distWo * sigmaa.g), powf(M_E, -distWo * sigmaa.b));
		Colour TrWi(powf(M_E, -distWi * sigmaa.r), powf(M_E, -distWi * sigmaa.g), powf(M_E, -distWi * sigmaa.b));

		// Evaluate Fresnel
		float fresnelWo = ShadingHelper::fresnelDielectric(wo.z, intIOR, extIOR);
		float fresnelWi = ShadingHelper::fresnelDielectric(wi.z, intIOR, extIOR);
		float etaWo = (wo.z > 0.f) ? (extIOR / intIOR) : (intIOR / extIOR);
		float etaWi = (wi.z > 0.f) ? (extIOR / intIOR) : (intIOR / extIOR);

		// Refract
		float refractWo = 1.f - fresnelWo;
		float refractWi = 1.f - fresnelWi;
		wi = shadingData.frame.toWorld(wi);
		reflectedColour = evaluate(shadingData, wi);
		return wi;
	}

	void invert(const ShadingData& shadingData, Sampler* sampler, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Add code to invert layered sampling
		return base->invert(shadingData, sampler, wi, u, v, selectProbability);
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Add code for evaluation of layer
		// return base->evaluate(shadingData, wi);
		// Convert shaingData.wo and wi to local space
		Vec4 wo = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z == 0.f || wo.z == 0.f) return Colour(0.f, 0.f, 0.f);

		// Handle absorbing medium
		float distWo = thickness / fabs(wo.z);
		float distWi = thickness / fabs(wiLocal.z);

		// Beer's Law for Absorption
		Colour TrWo(powf(M_E, -distWo * sigmaa.r), powf(M_E, -distWo * sigmaa.g), powf(M_E, -distWo * sigmaa.b));
		Colour TrWi(powf(M_E, -distWi * sigmaa.r), powf(M_E, -distWi * sigmaa.g), powf(M_E, -distWi * sigmaa.b));

		// Evaluate Fresnel
		float fresnelWo = ShadingHelper::fresnelDielectric(wo.z, intIOR, extIOR);
		float fresnelWi = ShadingHelper::fresnelDielectric(wiLocal.z, intIOR, extIOR);
		float etaWo = (wo.z > 0.f) ? (extIOR / intIOR) : (intIOR / extIOR);
		float etaWi = (wiLocal.z > 0.f) ? (extIOR / intIOR) : (intIOR / extIOR);

		// Refraction
		float refractWo = 1.f - fresnelWo;
		float refractWi = 1.f - fresnelWi;
		return base->evaluate(shadingData, shadingData.frame.toWorld(wiLocal)) * ((TrWo * etaWo * refractWo) + (TrWi * etaWi * refractWi));
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}

	bool isPureSpecular() { return base->isPureSpecular(); }
	bool isSmooth() { return base->isSmooth(); }
	bool canLearnFresnel() { return base->canLearnFresnel(); }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return base->mask(shadingData); }
};