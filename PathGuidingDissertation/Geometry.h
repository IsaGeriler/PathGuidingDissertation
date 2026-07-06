#pragma once

#include <algorithm>
#include <vector>

#include "MyMath.h"
#include "Sampling.h"

// Epsilon Definitions
// Decreasing Moller epsilon from 1e-6f to 1e-8f to fix coffee scene
#define EPSILON 1e-4f
#define MOLLER_TRUMBORE_EPSILON 1e-8f

// Definitions for BVH Accelleration Structure
#define MAXNODE_TRIANGLES 8
#define BOUNDS_COST 1.f
#define INTERSECT_COST 1.5f
#define BUILD_BINS 32

class Ray {
public:
	// Attributes - Origin, Direction, and Inverse Direction Vectors
	Vec4 o, dir, invDir;

	// Constructors
	Ray() {}
	Ray(Vec4 _o, Vec4 _d) { init(_o, _d); }

	// Methods
	void init(Vec4 _o, Vec4 _d) {
		o = _o;
		dir = _d;
		invDir = Vec4(1.f / dir.x, 1.f / dir.y, 1.f / dir.z);
	}

	Vec4 at(const float t) const { return (o + (dir * t)); }
};

class Plane {
public:
	// Attributes
	Vec4 n;
	float d;

	// Methods
	void init(Vec4& _n, float _d) { n = _n; d = _d; }
	
	// Ray-Plane Intersection Equation
	bool rayIntersect(Ray& r, float& t) const {
		// Calculate Denominator
		float denominator = Dot(n, r.dir);
		if (denominator < 1e-5f) return false;
		// Solve for t
		t = -(Dot(n, r.o) + d) / denominator;
		return (t > 1e-5f);
	}
};

class Triangle {
public:
	// Attributes
	Vertex vertices[3];
	Vec4 e0;  // Edge 0
	Vec4 e1;  // Edge 1
	Vec4 e2;  // Edge 2
	Vec4 n;   // Geometric Normal

	float area;  // Triangle area
	float d;     // For Ray-Triangle Intersection

	unsigned int materialIndex;  // For accessing BSDF materials

	// Methods
	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex) {
		// Initialize Material
		materialIndex = _materialIndex;

		// Initialize Vertices
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;

		// Calculate Edge Functions
		e0 = vertices[1].p - vertices[0].p;
		e1 = vertices[2].p - vertices[1].p;
		e2 = vertices[0].p - vertices[2].p;

		// Calculate Geometric Normal and Area
		n = e1.cross(e2).normalize();
		area = e1.cross(e2).length() * 0.5f;
		d = Dot(n, vertices[0].p);
	}

	Vec4 centre() const {
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.f;
	}

	// Möller-Trumbore Ray Triangle Intersection
	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const {
		// Recalculate edge coordinates
		Vec4 _e0 = vertices[1].p - vertices[0].p;
		Vec4 _e1 = vertices[2].p - vertices[0].p;

		// Calculate p, and determinant
		Vec4 p = Cross(r.dir, _e1);
		float det = Dot(_e0, p);
		
		// If det is less than epsilon, ray is parallel to the triangle
		if (fabs(det) < MOLLER_TRUMBORE_EPSILON) return false;

		// Apply Crammer's Rule
		// Calculate inverse determinant, and T
		float invDet = 1.f / det;
		Vec4 T = r.o - vertices[0].p;
		
		// Get Barycentric Beta
		float beta = Dot(T, p) * invDet;
		if (beta < 0.f || beta > 1.f) return false;

		// Calculate q, and Get Barycentric Gamma
		Vec4 q = Cross(T, _e0);
		float gamma = Dot(r.dir, q) * invDet;
		if (gamma < 0.f || gamma > 1.f || (beta + gamma > 1.f)) return false;

		// Calculate t
		t = Dot(_e1, q) * invDet;
		if (t < MOLLER_TRUMBORE_EPSILON) return false;

		// Assign alpha and beta from computed beta and gamma
		u = 1.f - beta - gamma;
		v = beta;
		return true;
	}

	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec4& interpolatedNormal, float& interpolatedU, float& interpolatedV) const {
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}
	
	Vec4 sample(Sampler* sampler, float& pdf) {
		float r1 = sampler->next();
		float r2 = sampler->next();
		float sqrtR1 = sqrtf(r1);

		float alpha = 1.f - sqrtR1;
		float beta = r2 * sqrtR1;
		float gamma = 1.f - alpha - beta;

		pdf = 1.f / area;
		return (vertices[0].p * alpha) + (vertices[1].p * beta) + (vertices[2].p * gamma);
	}

	// Geometry Normal (different than Shading Normal)
	Vec4 gNormal() const { return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f)); }
};

class AABB {
public:
	// Attributes
	Vec4 max, min;

	// Constructor
	AABB() { reset(); }

	// Methods
	void reset() {
		max = Vec4(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec4(FLT_MAX, FLT_MAX, FLT_MAX);
	}

	void extend(const Vec4 p) {
		max = Max(max, p);
		min = Min(min, p);
	}
	
	bool rayAABB(const Ray& r, float& t) const {
		// Check all six planes simultaneously
		Vec4 tmin = (min - r.o) * r.invDir;
		Vec4 tmax = (max - r.o) * r.invDir;
		
		// Values represent entry and exit
		Vec4 tentry = Min(tmin, tmax);
		Vec4 texit = Max(tmin, tmax);

		// Compute values
		float ftentry = std::max(tentry.x, std::max(tentry.y, tentry.z));
		float ftexit = std::min(texit.x, std::min(texit.y, texit.z));

		// Handle the case if entry is less than 0
		ftentry = std::max(0.f, ftentry);
		t = std::min(ftentry, ftexit);
		return (ftentry <= ftexit && ftexit >= 0.f);
	}

	bool rayAABB(const Ray& r) {
		// Check all six planes simultaneously
		Vec4 tmin = (min - r.o) * r.invDir;
		Vec4 tmax = (max - r.o) * r.invDir;

		// Values represent entry and exit
		Vec4 tentry = Min(tmin, tmax);
		Vec4 texit = Max(tmin, tmax);

		// Compute values
		float ftentry = std::max(tentry.x, std::max(tentry.y, tentry.z));
		float ftexit = std::min(texit.x, std::min(texit.y, texit.z));

		// Handle the case if entry is less than 0
		ftentry = std::max(0.f, ftentry);
		return (ftentry <= ftexit && ftexit >= 0.f);
	}

	float area() const {
		Vec4 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.f;
	}
};

class Sphere {
public:
	Vec4 centre;
	float radius;

	void init(Vec4& _centre, float _radius) {
		centre = _centre;
		radius = _radius;
	}

	// Add code here
	bool rayIntersect(Ray& r, float& t) {
		return false;
	}
};

// TO:DO - Store Each Path Vertex In Accelleration Structure (i.e. BVH)
//		   This Includes Position, Incoming Direction (wi), and Incoming Radiance
struct IntersectionData {
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

class BVHNode {
public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;
	// This can store an offset and number of triangles in a global triangle list for example
	// But you can store this however you want!
	// unsigned int offset = 0;
	// unsigned int num = 0;

	BVHNode() {
		r = NULL;
		l = NULL;
	}

	// Note there are several options for how to implement the build method. Update this as required
	void build(std::vector<Triangle>& inputTriangles) {
		// Add BVH building code here
	}

	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles) {
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}

	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection) {
		// Add BVH Traversal code here
	}

	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, const float maxT) {
		// Add visibility code here
		return true;
	}
};