#pragma once

#include <vector>

#include "MyMath.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Materials.h"
#include "Lights.h"

class Camera {
public:
	Matrix projectionMatrix;		 // Camera -> Clip Space (P)
	Matrix inverseProjectionMatrix;  // Clip Space -> Camera (Inverse P)
	Matrix camera;					 // Camera -> World (Inverse V)
	Matrix cameraToView;			 // World -> Camera (V)

	float width = 0.f;
	float height = 0.f;
	float Afilm;

	Vec4 origin;  // Ray Origin
	Vec4 viewDirection;
	
	void init(Matrix ProjectionMatrix, int screenwidth, int screenheight) {
		projectionMatrix = ProjectionMatrix;
		inverseProjectionMatrix = ProjectionMatrix.invert();
		width = static_cast<float>(screenwidth);
		height = static_cast<float>(screenheight);
		float Wlens = (2.f / ProjectionMatrix.a[1][1]);
		float aspect = ProjectionMatrix.a[0][0] / ProjectionMatrix.a[1][1];
		float Hlens = Wlens * aspect;
		Afilm = Wlens * Hlens;
	}

	void updateView(Matrix V) {
		camera = V;
		cameraToView = V.invert();
		origin = camera.mulPoint(Vec4(0.f, 0.f, 0.f));
		viewDirection = inverseProjectionMatrix.mulPointAndPerspectiveDivide(Vec4(0.f, 0.f, 1.f));
		viewDirection = camera.mulVec(viewDirection);
		viewDirection = viewDirection.normalize();
	}

	Ray generateRay(float x, float y) {
		// Normalize to NDC
		float xc = (2.f * x / width) - 1.f;
		float yc = (2.f * (1.f - (y / height))) - 1.f;

		// NDC to Clip Space
		Vec4 pClip(xc, yc, 0.f);

		// Clip Space to Camera Space
		Vec4 dCamera = inverseProjectionMatrix.mulPointAndPerspectiveDivide(pClip);
		Vec4 dir = camera.mulVec(dCamera).normalize();
		return Ray(origin, dir);
	}

	bool projectOntoCamera(const Vec4& p, float& x, float& y) {
		Vec4 pview = cameraToView.mulPoint(p);
		Vec4 pproj = projectionMatrix.mulPointAndPerspectiveDivide(pview);

		x = (pproj.x + 1.f) * 0.5f;
		y = (pproj.y + 1.f) * 0.5f;
		
		if (x < 0.f || x > 1.f || y < 0.f || y > 1.f) return false;
		x = x * width;
		y = 1.f - y;
		y = y * height;
		return true;
	}
};

class Scene {
public:
	std::vector<Triangle> triangles;
	std::vector<BSDF*> materials;
	std::vector<Light*> lights;
	Light* background = NULL;
	BVHNode* bvh = NULL;
	Camera camera;
	AABB bounds;

	void build() {
		// Add BVH building code here
		// BVH Build
		// if (bvh != nullptr) delete bvh;
		// bvh = new BVHNode();
		// bvh->build(triangles);

		// Do not touch the code below this line!
		// Build light list
		for (int i = 0; i < triangles.size(); i++) {
			if (materials[triangles[i].materialIndex]->isLight()) {
				AreaLight* light = new AreaLight();
				light->triangle = &triangles[i];
				light->emission = materials[triangles[i].materialIndex]->emission;
				lights.push_back(light);
			}
		}
	}

	IntersectionData traverse(const Ray& ray) {
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		for (int i = 0; i < triangles.size(); i++) {
			float t, u, v;
			if (triangles[i].rayIntersect(ray, t, u, v)) {
				if (t < intersection.t) {
					intersection.t = t;
					intersection.ID = i;
					intersection.alpha = u;
					intersection.beta = v;
					intersection.gamma = 1.0f - (u + v);
				}
			}
		}
		return intersection;
		//return bvh->traverse(ray, triangles);
	}

	// Uniform Light Sampling
	Light* sampleLight(Sampler* sampler, float& pmf) {
		// Handle no lights case
		if (lights.empty()) { pmf = 0.f; return nullptr; }

		// PMF for i'th light
		pmf = 1.f / lights.size();

		// Sampling
		unsigned int nLights = lights.size();
		unsigned int index = static_cast<unsigned int>(std::min<unsigned int>(std::max<unsigned int>(0u, floorf(sampler->next() * nLights)), nLights - 1u));
		return lights[index];
	}
	
	void init(std::vector<Triangle> meshTriangles, std::vector<BSDF*> meshMaterials, Light* _background) {
		for (int i = 0; i < meshTriangles.size(); i++) {
			triangles.push_back(meshTriangles[i]);
			bounds.extend(meshTriangles[i].vertices[0].p);
			bounds.extend(meshTriangles[i].vertices[1].p);
			bounds.extend(meshTriangles[i].vertices[2].p);
		}

		for (int i = 0; i < meshMaterials.size(); i++) {
			materials.push_back(meshMaterials[i]);
		}

		background = _background;
		if (background->totalIntegratedPower() > 0.f) {
			lights.push_back(background);
		}
	}

	bool visible(const Vec4& p1, const Vec4& p2) {
		Ray ray;
		Vec4 dir = p2 - p1;
		float maxT = dir.length() - (2.f * EPSILON);
		dir = dir.normalize();
		ray.init(p1 + (dir * EPSILON), dir);
		return bvh->traverseVisible(ray, triangles, maxT);
	}

	Colour emit(Triangle* light, ShadingData shadingData, Vec4 wi) {
		return materials[light->materialIndex]->emit(shadingData, wi);
	}

	ShadingData calculateShadingData(IntersectionData intersection, Ray& ray) {
		ShadingData shadingData = {};
		if (intersection.t < FLT_MAX) {
			shadingData.x = ray.at(intersection.t);
			shadingData.gNormal = triangles[intersection.ID].gNormal();
			triangles[intersection.ID].interpolateAttributes(intersection.alpha, intersection.beta, intersection.gamma, shadingData.sNormal, shadingData.tu, shadingData.tv);
			shadingData.bsdf = materials[triangles[intersection.ID].materialIndex];
			shadingData.wo = -ray.dir;

			if (shadingData.bsdf->isTwoSided()) {
				if (Dot(shadingData.wo, shadingData.sNormal) < 0.f) {
					shadingData.sNormal = -shadingData.sNormal;
				}
				if (Dot(shadingData.wo, shadingData.gNormal) < 0.f) {
					shadingData.gNormal = -shadingData.gNormal;
				}
			}
			shadingData.frame.fromVector(shadingData.sNormal);
			shadingData.t = intersection.t;
		} else {
			shadingData.wo = -ray.dir;
			shadingData.t = intersection.t;
		}
		return shadingData;
	}
};