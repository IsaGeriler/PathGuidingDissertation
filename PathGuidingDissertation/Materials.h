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
	virtual Vec4 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) = 0;
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

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Invert Cosine Sample Hemisphere where,
		// theta = acos(sqrt(r1)), phi = 2 * PI * r2
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		float theta = SphericalCoordinates::sphericalTheta(wiLocal);
		float phi = SphericalCoordinates::sphericalPhi(wiLocal);
		u = std::min(std::max(SQ(cosf(theta)), 0.f), 0.99999f);
		v = std::min(std::max((float)(phi / (2.f * M_PI)), 0.f), 0.99999f);
		selectProbability = 0.f;
	}

	Colour evaluate(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z * wiLocal.z <= 0.f) return Colour(0.f, 0.f, 0.f);
		return albedo->sample(shadingData.tu, shadingData.tv) * 0.318309886183790671538;
	}

	float PDF(const ShadingData& shadingData, const Vec4& wi) {
		// Convert wi to local space
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z * wiLocal.z <= 0.f) return 0.f;
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

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// MirrorBSDF will not be inverted as this is due to Dirac Delta distribution
		// Invert will return 0 for u, v, and selectProbability
		u = 0.f;
		v = 0.f;
		selectProbability = 0.f;
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
		float r1 = sampler->next();
		float r2 = sampler->next();

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

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		if (alpha < EPSILON) {
			// Treat this case as MirrorBSDF
			u = 0.f;
			v = 0.f;
			selectProbability = 0.f;  // TO:DO - Save Fresnel Probability here...
		} else {
			Vec4 wiLocal = shadingData.frame.toLocal(wi);
			Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);
			// thetaM and phiM are obtained from the half vector
			Vec4 wmLocal = (wiLocal + woLocal).normalize();
			float theta = SphericalCoordinates::sphericalTheta(wmLocal);
			float phi = SphericalCoordinates::sphericalPhi(wmLocal);
			// for u, use it's CDF as we used CDF inversion for sampling!
			// phi = 2 * PI * v
			float alphaSq = alpha * alpha;
			float alphaSqMinusOne = (alphaSq - 1.f);
			u = (alphaSq / (SQ(cos(theta)) * (alphaSqMinusOne * alphaSqMinusOne) + alphaSqMinusOne)) - (1.f / alphaSqMinusOne);
			v = phi / (2 * M_PI);
			selectProbability = 0.f;  // TO:DO - Save Fresnel Probability here...
		}
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

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// GlassBSDF will not be inverted as this is due to Dirac Delta distribution
		// But we can save the Fresnel probability for the next bounce
		u = 0.f;
		v = 0.f;
		selectProbability = 1.f;  // TO:DO - Save Fresnel Probability here...
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
		// Convert shadingData.wo to the local space
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// If alpha is less than epsilon, treat it as a mirror with fresnel
		if (alpha < EPSILON) {
			// Calculate fresnel to determine if we should reflect or refract
			float fresnel = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
			if (sampler->next() < fresnel) {
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
		float r1 = sampler->next();
		float r2 = sampler->next();

		float theta = acosf(sqrtf((1.f - r1) / (r1 * (alphaSq - 1.f) + 1.f)));
		float phi = 2.f * M_PI * r2;

		Vec4 wmLocal = SphericalCoordinates::sphericalToWorld(theta, phi);

		// Calculate Fresnel Term
		float F = ShadingHelper::fresnelDielectric(Dot(woLocal, wmLocal), intIOR, extIOR);
		float selectProbability = sampler->next();

		if (selectProbability < F) {
			// Reflect
			Vec4 wiLocal = -woLocal + (wmLocal * 2 * Dot(wmLocal, woLocal));
			if (wiLocal.z * woLocal.z <= 0.f) return {};
			Vec4 wi = shadingData.frame.toWorld(wiLocal);
			reflectedColour = evaluate(shadingData, wi);
			pdf = PDF(shadingData, wi);
			return wi;
		} else {
			// Refract
			if (woLocal.z < 0) wmLocal = -wmLocal;
			
			float cosThetaI = std::min(std::max(Dot(wmLocal, woLocal), -1.f), 1.f);
			float eta = (cosThetaI > 0.f) ? (extIOR / intIOR) : (intIOR / extIOR);
			float etaSq = eta * eta;

			float cosThetaISq = std::max(cosThetaI * cosThetaI, 1.f);
			float sinThetaISq = std::max(1.f - cosThetaISq, 0.f);
			float sinThetaTSq = etaSq * sinThetaISq;

			if (sinThetaTSq >= 1.f) {
				// Total Internal Reflection
				Vec4 wr = shadingData.frame.toWorld(Vec4(-woLocal.x, -woLocal.y, woLocal.z));
				pdf = F;
				reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * F / fabs(woLocal.z);
				return wr;
			}

			float cosThetaT = sqrt(std::max(1.f - sinThetaTSq, 0.f));
			Vec4 wiLocal = -woLocal * eta + wmLocal * (eta * cosThetaI - cosThetaT);
			wiLocal = wiLocal.normalize();
			Vec4 wi = shadingData.frame.toWorld(wiLocal);
			reflectedColour = evaluate(shadingData, wi);
			pdf = PDF(shadingData, wi);
			return wi;
		}
	}

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Treat this case as MirrorBSDF
		if (alpha < EPSILON) {
			u = 0.f;
			v = 0.f;
			selectProbability = 0.f;  // TO:DO - Save Fresnel Probability here...
		}
		// Else invert the half vector sampling like ConductorBSDF
		else {
			Vec4 wiLocal = shadingData.frame.toLocal(wi);
			Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

			// thetaM and phiM are obtained from the half vector
			Vec4 wmLocal = (wiLocal + woLocal).normalize();
			float theta = SphericalCoordinates::sphericalTheta(wmLocal);
			float phi = SphericalCoordinates::sphericalPhi(wmLocal);

			// for u, use it's CDF as we used CDF inversion for sampling!
			// phi = 2 * PI * v
			float alphaSq = alpha * alpha;
			float alphaSqMinusOne = (alphaSq - 1.f);
			u = (alphaSq / (SQ(cos(theta)) * (alphaSqMinusOne * alphaSqMinusOne) + alphaSqMinusOne)) - (1.f / alphaSqMinusOne);
			v = phi / (2 * M_PI);

			float F = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
			bool reflect = woLocal.z * wiLocal.z > 0.f;
			selectProbability = reflect ? (F * 0.5f) : (F + ((1.f - F) * 0.5f));
		}
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
		float eta = eta_i / eta_o;
		
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
		float eta = eta_i / eta_o;

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
		float r1 = sampler->next();
		float r2 = sampler->next();
		Vec4 wi = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		// Convert wi to world space before passing to evaluate and pdf
		wi = shadingData.frame.toWorld(wi);

		// Evaluate BSDF and PDF
		reflectedColour = evaluate(shadingData, wi);  // BSDF = Diffuse * OrenNayar
		pdf = PDF(shadingData, wi);					  // PDF = cos(theta) / PI

		// Return incoming direction
		return wi;
	}

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Invert Cosine Sample Hemisphere where,
		// theta = acos(sqrt(r1)), phi = 2 * PI * r2
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		float theta = SphericalCoordinates::sphericalTheta(wiLocal);
		float phi = SphericalCoordinates::sphericalPhi(wiLocal);
		u = std::min(std::max(SQ(cosf(theta)), 0.f), 0.99999f);
		v = std::min(std::max((float)(phi / (2.f * M_PI)), 0.f), 0.99999f);
		selectProbability = 0.f;
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

		// Fresnel to compute diffuse or reflect surface
		Vec4 wiLocal;
		float r1 = sampler->next();
		float r2 = sampler->next();
		float selectProbability = sampler->next();
		float fresnel = ShadingHelper::fresnelDielectric(fabs(woLocal.z), intIOR, extIOR);
		if (selectProbability < fresnel) {
			// Glossy Part - Sample theta and phi from random variables for half vector
			float e = alphaToPhongExponent();
			float base = 1.f - r1;
			float power = 1.f / (e + 1.f);

			float thetaH = acosf(pow(base, power));
			float phiH = 2.f * M_PI * r2;

			// Get half vector
			Vec4 hLocal = SphericalCoordinates::sphericalToWorld(thetaH, phiH).normalize();
			
			// Then reflect over half vector
			wiLocal = hLocal * 2.f * Dot(woLocal, hLocal) - woLocal;
			if (wiLocal.z <= 0.f) { pdf = 0.f; reflectedColour = Colour(0.f, 0.f, 0.f); return Vec4(0.f, 0.f, 1.f); }
		} else {
			// Diffuse Part - Sample wi with cosine hemisphere
			wiLocal = SamplingDistributions::cosineSampleHemisphere(r1, r2);
		}

		Vec4 wi = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wi);
		pdf = PDF(shadingData, wi);
		return wi;
	}

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// TO:DO - Also invert the diffuse part if selectProbability < fresnel, but for now we will only invert the glossy part
		Vec4 wiLocal = shadingData.frame.toLocal(wi);
		Vec4 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Calculate half vector from wi and wo
		Vec4 hLocal = (wiLocal + woLocal).normalize();

		// Calculate theta and phi for half vector, and save phong exponent to a variable
		float thetaH = SphericalCoordinates::sphericalTheta(hLocal);
		float phiH = SphericalCoordinates::sphericalPhi(hLocal);
		float e = alphaToPhongExponent();

		// Invert theta and phi to get u and v
		u = 1.f - powf(cos(thetaH), (e + 1.f));
		v = phiH / (2.f * M_PI);
		selectProbability = 1.f;  // TO:DO - Save Fresnel or Glossy Lobe Probability here...
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

		// Calculate ks and kd
		float e = alphaToPhongExponent();
		float ks = ShadingHelper::fresnelDielectric(fabs(woLocal.z), intIOR, extIOR);
		float kd = 1.f - ks;
		
		// Blinn Normalization Factor (modified)
		// https://renderwonk.com/publications/s2010-shading-course/gotanda/course_note_practical_implementation_at_triace.pdf
		float norm = ((e + 2.f) * (e + 4.f)) / (8.f * M_PI * (std::powf(2.f, (-e * 0.5f)) + e));
		float normLowerBound = (e + 6.f) / (8.f * M_PI);
		float normUpperBound = (e + 8.f) / (8.f * M_PI);

		// Clamp normalization factor
		norm = std::min(std::max(normLowerBound, norm), normUpperBound);

		// BSDF = kd * DiffuseBSDF + ks * GlossyBSDF
		float glossyComponent = std::powf(std::max(hLocal.z, 0.f), e);
		Colour diffuseBSDF = albedo->sample(shadingData.tu, shadingData.tv) * M_1_PI;
		Colour glossyBSDF = Colour(1.f, 1.f, 1.f) * glossyComponent;
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
		float ks = ShadingHelper::fresnelDielectric(fabs(woLocal.z), intIOR, extIOR);
		float kd = 1.f - ks;

		// PDF = kd * DiffusePDF + ks * GlossyPDF
		float ph = ((e + 1.f) / (2.f * M_PI)) * std::powf(std::max(hLocal.z, 0.f), e);
		float glossyPDF = (Dot(woLocal, hLocal) <= 0.f) ? 0.f : (ph / (4.f * Dot(woLocal, hLocal)));
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

	void invert(const ShadingData& shadingData, const Vec4& wi, float& u, float& v, float& selectProbability) {
		// Add code to invert layered sampling
		return base->invert(shadingData, wi, u, v, selectProbability);
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
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) { return base->mask(shadingData); }
};