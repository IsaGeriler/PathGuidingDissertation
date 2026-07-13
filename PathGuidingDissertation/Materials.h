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
		if (sinThetaTSq > 1.f) return 1.f;
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
	virtual Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec4& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec4& wi) = 0;
	virtual bool isPureSpecular() = 0;
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
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Sample incoming direction (z-up coordinate system, local space)
		float r1 = sampler->next();
		float r2 = sampler->next();
		Vec4 wi = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		// Convert wi to world space before passing to evaluate and pdf
		wi = shadingData.frame.toWorld(wi);
		
		// Evaluate BSDF and PDF
		reflectedColour = evaluate(shadingData, wi);  // BSDF = albedo / PI
		pdf = PDF(shadingData, wi);					  // PDF = cos(theta) / PI

		// Return incoming direction
		return wi;
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
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Convert shadingData.wo to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Reflect x and y of woLocal
		Vec4 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);

		// Guard case to prevent division by zero
		if (fabsf(wrLocal.z) <= EPSILON) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
		
		// Convert back to world space and return
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / wrLocal.z;  // BSDF = albedo / Dot(wr, n)
		pdf = 1.f;																	   // PDF = 1 (for perfect specular reflection)
		return shadingData.frame.toWorld(wrLocal);
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		return Colour(0.f, 0.f, 0.f);
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		return 0.f;
	}

	bool isPureSpecular() { return true; }
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
		alpha = 1.62142f * sqrtf(roughness);
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
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
		float alphaSq = alpha * alpha;
		float r1 = sampler->next(), r2 = sampler->next();

		float thetaM = acosf(sqrtf((1.f - r1) / (r1 * (alphaSq - 1.f) + 1.f)));
		float phiM = 2.f * M_PI * r2;

		// Find wi from wo and wm (light reflected across microfacet model)
		Vec4 wmLocal = SphericalCoordinates::sphericalToWorld(thetaM, phiM);
		Vec4 wiLocal = -woLocal + wmLocal * 2.f * Dot(wmLocal, woLocal);

		Vec4 wi = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wi);
		pdf = PDF(shadingData, wi);
		return wi;
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
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Convert shadingData.wo to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		
		// Cosine term guard case
		if (fabsf(woLocal.z) < EPSILON) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }

		// Fresnel is the probability of the material reflecting
		float fresnel = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		float cosThetaI = woLocal.z;
		float eta = (cosThetaI < 0.f) ? (intIOR / extIOR) : (extIOR / intIOR);
		if (cosThetaI < 0.f) cosThetaI = fabsf(cosThetaI);

		// Generate a random number to decide reflect of refract
		if (sampler->next() < fresnel) {
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
			if (sinThetaTSq > 1.f) {
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

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		return Colour(0.f, 0.f, 0.f);
	}
	
	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		return 0.f;
	}

	bool isPureSpecular() { return true; }
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
		alpha = 1.62142f * sqrtf(roughness);
	}

	// Methods
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Convert wo to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		
		// Alpha guard case for perfect specular Dielectric BSDF
		if (alpha < EPSILON) {
			// Calculate fresnel to determine if we should reflect or refract
			float fresnel = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
			if (sampler->next() < fresnel) {
				// Reflect
				Vec4 wiLocal(-woLocal.x, -woLocal.y, woLocal.z);
				pdf = fresnel;
				reflectedColour = (albedo->sample(shadingData.tu, shadingData.tv) / fabs(wiLocal.z)) * fresnel;
				return shadingData.frame.toWorld(wiLocal);
			} else {
				// Refract
				float eta = woLocal.z > 0.f ? extIOR / intIOR : intIOR / extIOR;
				float etaSq = eta * eta;
				float sign = woLocal.z > 0.f ? -1.f : 1.f;

				float cosThetaISq = std::max(woLocal.z * woLocal.z, 0.f);
				float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);
				float sinThetaTSq = etaSq * sinThetaISq;

				// Total Internal Reflection
				if (sinThetaTSq > 1.f) {
					// Perfect Specular Reflection
					// Treat as mirror with fresnel
					Vec4 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
					pdf = fresnel;
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
		float r1 = sampler->next(), r2 = sampler->next();

		float thetaM = acosf(sqrtf((1.f - r1) / (r1 * (alphaSq - 1.f) + 1.f)));
		float phiM = 2.f * M_PI * r2;
		Vec4 wmLocal = SphericalCoordinates::sphericalToWorld(thetaM, phiM);

		// Calculate fresnel to determine if we should reflect or refract
		float fresnel = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		if (sampler->next() < fresnel) {
			// Reflect (just like ConductorBSDF)
			Vec4 wiLocal = -woLocal + wmLocal * 2.f * Dot(wmLocal, woLocal);
			Vec4 wi = shadingData.frame.toWorld(wiLocal);
			reflectedColour = evaluate(shadingData, wi);
			pdf = PDF(shadingData, wi);
			return wi;
		} else {
			// Refract
			Frame wmFrame;
			wmFrame.fromVector(wmLocal);
			Vec4 woRefractedFromWm = wmFrame.toLocal(woLocal);
			float cosThetaI = woRefractedFromWm.z;

			float eta = cosThetaI > 0.f ? extIOR / intIOR : intIOR / extIOR;
			float etaSq = eta * eta;
			float sign = cosThetaI > 0.f ? -1.f : 1.f;

			float cosThetaISq = std::max(cosThetaI * cosThetaI, 0.f);
			float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);
			float sinThetaTSq = etaSq * sinThetaISq;
			float cosThetaT = sqrt(std::max(1.f - sinThetaTSq, 0.f));

			Vec4 wiLocal(-eta * woRefractedFromWm.x, -eta * woRefractedFromWm.y, cosThetaT * sign);
			Vec4 wi = shadingData.frame.toWorld(wmFrame.toWorld(wiLocal));
			reflectedColour = evaluate(shadingData, wi);
			pdf = PDF(shadingData, wi);
			return wi;
		}
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wo and wi to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wiLocal = shadingData.frame.toLocal(wi);

		// Can only sample from visible normals
		if (woLocal.z == 0.f || wiLocal.z == 0.f) return Colour(0.f, 0.f, 0.f);

		// Handling the perfect specular DielectricBSDF just like GlassBSDF
		float eta = woLocal.z > 0.f ? extIOR / intIOR : intIOR / extIOR;
		if (eta == 1.f || alpha < EPSILON) return Colour(0.f, 0.f, 0.f);

		// Determine reflect according to PBRT
		// https://pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF
		bool reflect = woLocal.z * wiLocal.z > 0.f;
		float etaProbability = 1.f;
		if (!reflect) etaProbability = woLocal.z > 0.f ? eta : 1.f / eta;
		Vec4 wmLocal = wiLocal * etaProbability + woLocal;
		if (wmLocal.lengthSquare() < EPSILON) return Colour(0.f, 0.f, 0.f);
		wmLocal = wmLocal.normalize();

		// Discard backfacing microfacets
		if (Dot(wmLocal, wiLocal) * wiLocal.z < 0.f || Dot(wmLocal, woLocal) * woLocal.z < 0.f) return Colour(0.f, 0.f, 0.f);

		// BSDF evaluation from ConductorBSDF
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		if (reflect) {
			if (Dot(woLocal, wmLocal) <= 0.f) return Colour(0.f, 0.f, 0.f);
			if (wiLocal.z * woLocal.z == 0.f) return Colour(0.f, 0.f, 0.f);

			// Cook-Torrance BRDF
			float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
			float D = ShadingHelper::Dggx(wmLocal, alpha);
			return albedo->sample(shadingData.tu, shadingData.tv) * ((F * G * D) / (4.f * wiLocal.z * woLocal.z));
		}
		// BSDF evaluation of Microfacet Refraction
		else {
			float etaSq = etaProbability * etaProbability;
			float denominator = (Dot(wiLocal, wmLocal) + Dot(woLocal, wmLocal) / etaProbability) * (Dot(wiLocal, wmLocal) + Dot(woLocal, wmLocal) / etaProbability);
			if (denominator <= 0.f) return Colour(0.f, 0.f, 0.f);

			// Cook-Torrance BRDF
			float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
			float D = ShadingHelper::Dggx(wmLocal, alpha);
			return albedo->sample(shadingData.tu, shadingData.tv) * (((1.f - F) * G * D) / denominator) * (fabs(Dot(wiLocal, wmLocal)) / fabs(wiLocal.z)) * (fabs(Dot(woLocal, wmLocal)) / fabs(woLocal.z));
		}
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wo and wi to local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec4 wiLocal = shadingData.frame.toLocal(wi);

		// Can only sample from visible normals
		if (woLocal.z == 0.f || wiLocal.z == 0.f) return 0.f;

		// Handling the perfect specular DielectricBSDF just like GlassBSDF
		float eta = woLocal.z > 0.f ? extIOR / intIOR : intIOR / extIOR;
		if (eta == 1.f || alpha < EPSILON) return 0.f;

		// Determine reflect according to PBRT
		// https://pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF
		bool reflect = woLocal.z * wiLocal.z > 0.f;
		float etaProbability = 1.f;
		if (!reflect) etaProbability = woLocal.z > 0.f ? eta : 1.f / eta;
		Vec4 wmLocal = wiLocal * etaProbability + woLocal;
		if (wmLocal.lengthSquare() < EPSILON) return 0.f;
		wmLocal = wmLocal.normalize();

		// Discard backfacing microfacets
		if (Dot(wmLocal, wiLocal) * wiLocal.z < 0.f || Dot(wmLocal, woLocal) * woLocal.z < 0.f) return 0.f;

		// PDF evaluation from ConductorBSDF
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		if (reflect) {
			if (Dot(woLocal, wmLocal) <= 0.f) return 0.f;
			if (wiLocal.z * woLocal.z == 0.f) return 0.f;

			float D = ShadingHelper::Dggx(wmLocal, alpha);
			return F * ((D * wmLocal.z) / (4.f * Dot(woLocal, wmLocal)));
		}
		// PDF evaluation of Microfacet Refraction
		else {
			float denominator = (Dot(wiLocal, wmLocal) + Dot(woLocal, wmLocal) / etaProbability) * (Dot(wiLocal, wmLocal) + Dot(woLocal, wmLocal) / etaProbability);
			if (denominator <= 0.f) return 0.f;
			
			float D = ShadingHelper::Dggx(wmLocal, alpha);
			return (1.f - F) * ((D * fabs(Dot(woLocal, wmLocal))) / denominator);
		}
	}

	bool isPureSpecular() { return alpha < EPSILON; }
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
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Sample incoming direction (z-up coordinate system, local space)
		// Using DiffuseBSDF's cosine weighted sampling 
		Vec4 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());

		// Convert wi to world space before passing to evaluate and pdf
		wi = shadingData.frame.toWorld(wi);

		// Evaluate BSDF and PDF
		reflectedColour = evaluate(shadingData, wi);  // BSDF = Diffuse * OrenNayar
		pdf = PDF(shadingData, wi);					  // PDF = cos(theta) / PI

		// Return incoming direction
		return wi;
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Replace this with OrenNayar evaluation code
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
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};

// Plastic Model: BlinnBSDF
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
		alpha = 1.62142f * sqrtf(roughness);
	}

	// Methods
	float alphaToPhongExponent() {
		return (2.f / SQ(std::max(alpha, 0.001f))) - 2.f;
	}

	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Get outcoming direction
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		//if (woLocal.z <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }

		// Fresnel to compute diffuse or reflect surface
		Vec4 wiLocal;
		float fresnel = ShadingHelper::fresnelDielectric(fabs(woLocal.z), intIOR, extIOR);
		if (sampler->next() < fresnel) {
			// Glossy Part - Sample theta and phi from random variables for half vector
			float e = alphaToPhongExponent();
			float base = 1.f - sampler->next();
			float power = 1.f / (e + 1.f);

			float thetaH = acosf(pow(base, power));
			float phiH = 2.f * M_PI * sampler->next();

			// Get half vector
			Vec4 hLocal = SphericalCoordinates::sphericalToWorld(thetaH, phiH).normalize();
			// if (Dot(woLocal, hLocal) <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
			
			// Then reflect over half vector
			wiLocal = hLocal * 2.f * Dot(woLocal, hLocal) - woLocal;
			// if (wiLocal.z <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
		} else {
			// Diffuse Part - Sample wi with cosine hemisphere
			wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		}

		Vec4 wi = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wi);
		pdf = PDF(shadingData, wi);
		return wi;
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Calculate half vector from wi and wo
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Guard case for cosine terms
		if (wiLocal.z <= 0.f || woLocal.z <= 0.f) return Colour(0.f, 0.f, 0.f);

		// Get half vector and lenght check for half veector
		Vec4 hLocal = wiLocal + woLocal;
		if (hLocal.lengthSquare() < EPSILON) return Colour(0.f, 0.f, 0.f);
		hLocal = hLocal.normalize();

		// if (Dot(woLocal, hLocal) <= 0.f) return Colour(0.f, 0.f, 0.f);

		// Calculate ks and kd
		float e = alphaToPhongExponent();
		float ks = ShadingHelper::fresnelDielectric(Dot(woLocal, hLocal), intIOR, extIOR);
		//float ks = ShadingHelper::fresnelDielectric(fabs(woLocal.z), intIOR, extIOR);
		float kd = 1.f - ks;
		
		// Blinn Normalization Factor
		// https://renderwonk.com/publications/s2010-shading-course/gotanda/course_note_practical_implementation_at_triace.pdf
		float norm = (e + 2.f) / (4.f * M_PI * (2.f - std::powf(2.f, (-e * 0.5f))));
		float normLowerBound = (e + 2.f) / (8.f * M_PI);
		float normUpperBound = (e + 4.f) / (8.f * M_PI);

		// Clamp normalization factor
		norm = std::min(std::max(normLowerBound, norm), normUpperBound);

		// BSDF = kd * DiffuseBSDF + ks * GlossyBSDF
		float glossyComponent = std::powf(std::max(hLocal.z, 0.f), e);
		Colour diffuseBSDF = albedo->sample(shadingData.tu, shadingData.tv) * M_1_PI;
		Colour glossyBSDF = Dot(woLocal, hLocal) <= 0.f ? Colour(0.f, 0.f, 0.f) : Colour(1.f, 1.f, 1.f) * glossyComponent;
		return (diffuseBSDF * kd) + (glossyBSDF * norm * ks);
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Calculate half vector from wi and wo
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Guard case for cosine terms
		if (wiLocal.z <= 0.f || woLocal.z <= 0.f) return 0.f;
		
		// Get half vector and lenght check for half veector
		Vec4 hLocal = wiLocal + woLocal;
		if (hLocal.lengthSquare() < EPSILON) return 0.f;
		hLocal = hLocal.normalize();

		// Calculate ks and kd
		float e = alphaToPhongExponent();
		float ks = ShadingHelper::fresnelDielectric(Dot(woLocal, hLocal), intIOR, extIOR);
		//float ks = ShadingHelper::fresnelDielectric(fabs(woLocal.z), intIOR, extIOR);
		float kd = 1.f - ks;

		// PDF = kd * DiffusePDF + ks * GlossyPDF
		// if (Dot(woLocal, hLocal) <= 0.f) return 0.f;
		float ph = ((e + 1.f) / (2.f * M_PI)) * std::powf(std::max(hLocal.z, 0.f), e);
		float glossyPDF = Dot(woLocal, hLocal) <= 0.f ? 0.f : (ph / (4.f * Dot(woLocal, hLocal)));
		float diffusePDF = SamplingDistributions::cosineHemispherePDF(wiLocal);
		return (kd * diffusePDF) + (ks * glossyPDF);
	}

	bool isPureSpecular() { return false; }
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
	Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) {
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}

	bool isPureSpecular() { return base->isPureSpecular(); }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return base->mask(shadingData); }
};