#pragma once

#include <algorithm>
#include <climits>
#include <vector>
#include <utility>

#include "MyMath.h"
#include "Sampling.h"

// Epsilon Definitions
// Decreasing Moller epsilon from 1e-6f to 1e-8f to fix coffee scene
#define EPSILON 1e-4f
#define MOLLER_TRUMBORE_EPSILON 1e-8f

// Definitions for BVH Accelleration Structure
#define MAXNODE_TRIANGLES 4
#define BOUNDS_COST 1.f
#define INTERSECT_COST 1.5f
#define BUILD_BINS 16

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
		if (fabs(denominator) < EPSILON) return false;
		// Solve for t
		t = (-Dot(n, r.o) - d) / denominator;
		return (t >= 0.f);
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
		return ((vertices[0].p + vertices[1].p + vertices[2].p) * 0.3333f);
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
		
		// Get Barycentric Beta (u)
		u = Dot(T, p) * invDet;
		if (u < 0.f || u > 1.f) return false;

		// Calculate q, and Get Barycentric Gamma (v)
		Vec4 q = Cross(T, _e0);
		v = Dot(r.dir, q) * invDet;
		if (v < 0.f || v > 1.f || (u + v > 1.f)) return false;

		// Calculate t
		t = Dot(_e1, q) * invDet;
		return (t >= 0.f);
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
		return ((vertices[0].p * alpha) + (vertices[1].p * beta) + (vertices[2].p * gamma));
	}

	// Geometry Normal (different than Shading Normal)
	Vec4 gNormal() const { return (n * (Dot(vertices[0].normal, n) > 0.f ? 1.f : -1.f)); }
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
		if (ftentry > ftexit || ftexit < 0.f) return false;
		t = std::max(0.f, ftentry);
		return true;
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
		if (ftentry > ftexit || ftexit < 0.f) return false;
		return true;
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
	unsigned int ID = UINT_MAX;
	float t = FLT_MAX;
	float alpha = 0.f;
	float beta = 0.f;
	float gamma = 0.f;
};

struct BVHBuildNode {
	AABB bounds;
	// rightChild = leftChild + 1, so no need to store it
	// leftFirst = 0 is root node, no need to reserve offset I guess...
	int leftFirst = 0, used = 0;
	bool isLeaf() const { return used > 0; }
};

struct SAHBin {
	AABB bounds;
	int triangleCount = 0;
};

class BVHNode {
private:
	std::vector<BVHBuildNode> bvhBuildNode;
	std::vector<int> triangleIndexes;
	int rootIndex = 0, nodesUsed = 1;
private:
	void updateBounds(std::vector<Triangle>& triangles, int index) {
		// Retrieve corresponding BVHBuildNode and reset it's AABB bounds
		BVHBuildNode& node = bvhBuildNode[index];
		node.bounds.reset();

		// Traverse the triangles associated with BVHBuildNode and update it's bounds
		for (int first = node.leftFirst, i = 0; i < node.used; i++) {
			int triangleIndex = triangleIndexes[first + i];
			Triangle& triangle = triangles[triangleIndex];
			node.bounds.extend(triangle.vertices[0].p);
			node.bounds.extend(triangle.vertices[1].p);
			node.bounds.extend(triangle.vertices[2].p);
		}
	}

	float calculateCost(BVHBuildNode& node) {
		// return node.used * node.bounds.area();
		return node.used * INTERSECT_COST;
	}

	float findBestSplitPlane(BVHBuildNode& node, std::vector<Triangle>& triangles, int& axis, float& splitPosition) {
		float parentArea = node.bounds.area();
		float bestCost = FLT_MAX;
		for (int ax = 0; ax < 3; ax++) {
			float minBounds = FLT_MAX, maxBounds = -FLT_MAX;
			for (int first = node.leftFirst, i = 0; i < node.used; i++) {
				int triangleIndex = triangleIndexes[first + i];
				Triangle& triangle = triangles[triangleIndex];
				minBounds = std::min(minBounds, triangle.centre()[ax]);
				maxBounds = std::max(maxBounds, triangle.centre()[ax]);
			}
			if (minBounds == maxBounds) continue;

			// Populate Bins
			SAHBin bin[BUILD_BINS];
			for (int i = 0; i < BUILD_BINS; i++) {
				bin[i].bounds.reset();
				bin[i].triangleCount = 0;
			}

			float scale = BUILD_BINS / (maxBounds - minBounds);
			for (int first = node.leftFirst, i = 0; i < node.used; i++) {
				int triangleIndex = triangleIndexes[first + i];
				Triangle& triangle = triangles[triangleIndex];
				int binIndex = std::min(BUILD_BINS - 1, static_cast<int>((triangle.centre()[ax] - minBounds) * scale));
				
				bin[binIndex].triangleCount++;
				bin[binIndex].bounds.extend(triangle.vertices[0].p);
				bin[binIndex].bounds.extend(triangle.vertices[1].p);
				bin[binIndex].bounds.extend(triangle.vertices[2].p);
			}

			// Gather data for the rest of the planes
			float leftArea[BUILD_BINS - 1]{}, rightArea[BUILD_BINS - 1]{};
			int leftCount[BUILD_BINS - 1]{}, rightCount[BUILD_BINS - 1]{};
			int leftCountSum = 0, rightCountSum = 0;
			AABB leftBounds, rightBounds;
			leftBounds.reset(); rightBounds.reset();

			for (int i = 0; i < BUILD_BINS - 1; i++) {
				// Evaluate left side of the split
				leftCountSum += bin[i].triangleCount;
				leftCount[i] = leftCountSum;

				if (bin[i].triangleCount > 0) {
					leftBounds.extend(bin[i].bounds.min);
					leftBounds.extend(bin[i].bounds.max);
				}
				leftArea[i] = (leftCountSum > 0) ? leftBounds.area() : 0.f;

				// Evaluate right side of the split
				rightCountSum += bin[BUILD_BINS - 1 - i].triangleCount;
				rightCount[BUILD_BINS - 2 - i] = rightCountSum;

				if (bin[BUILD_BINS - 1 - i].triangleCount > 0) {
					rightBounds.extend(bin[BUILD_BINS - 1 - i].bounds.min);
					rightBounds.extend(bin[BUILD_BINS - 1 - i].bounds.max);
				}
				rightArea[BUILD_BINS - 2 - i] = (rightCountSum > 0) ? rightBounds.area() : 0.f;
			}

			// Calculate split cost via Surface Area Heuristic (SAH) formula
			scale = (maxBounds - minBounds) / BUILD_BINS;
			for (int i = 0; i < BUILD_BINS - 1; i++) {
				float leftDividedParentArea = leftArea[i] / parentArea;
				float rightDividedParentArea = rightArea[i] / parentArea;
				float cost = BOUNDS_COST + (leftDividedParentArea * leftCount[i] * INTERSECT_COST) + (rightDividedParentArea * rightCount[i] * INTERSECT_COST);
				float candidatePosition = minBounds + scale * (i + 1);
				if (cost < bestCost) { splitPosition = candidatePosition, axis = ax, bestCost = cost; }
			}
		}
		return bestCost;
	}

	void subdivide(std::vector<Triangle>& triangles, int index) {
		// Terminate recursion
		BVHBuildNode& node = bvhBuildNode[index];
		if (node.used <= MAXNODE_TRIANGLES) return;

		// Split plane axis and position
		// Determine split axis using Surface Area Heuristic (SAH)
		// Calculate cost of splitting and not splitting
		int axis = -1;
		float splitPosition = -FLT_MAX;
		float splitCost = findBestSplitPlane(node, triangles, axis, splitPosition);
		float noSplitCost = calculateCost(node);
		
		// Terminate if split is more expensive than not splitting
		if (splitCost >= noSplitCost || axis == -1) return;

		// Split the group in half
		int i = node.leftFirst;
		int j = i + node.used - 1;
		while (i <= j) {
			int triangleIndex = triangleIndexes[i];
			Vec4 triangleCentroid = triangles[triangleIndex].centre();
			if (triangleCentroid[axis] < splitPosition) i++;
			else std::swap(triangleIndexes[i], triangleIndexes[j--]);
		}

		// Return if split failed
		int leftCount = i - node.leftFirst;
		if (leftCount == 0 || leftCount == node.used) return;

		// Create the child nodes
		int leftChildIndex = nodesUsed++;
		int rightChildIndex = nodesUsed++;
		
		bvhBuildNode[leftChildIndex].leftFirst = node.leftFirst;
		bvhBuildNode[leftChildIndex].used = leftCount;
		bvhBuildNode[rightChildIndex].leftFirst = i;
		bvhBuildNode[rightChildIndex].used = node.used - leftCount;
		
		// Update the parent node
		node.leftFirst = leftChildIndex;
		node.used = 0;

		// Update the bounds of the child nodes
		updateBounds(triangles, leftChildIndex);
		updateBounds(triangles, rightChildIndex);

		// Subdivide recursively
		subdivide(triangles, leftChildIndex);
		subdivide(triangles, rightChildIndex);
	}
public:
	void build(std::vector<Triangle>& inputTriangles) {
		// Early return if triangle list is empty
		if (inputTriangles.empty()) return;

		// Reset these in case of multiple builds
		rootIndex = 0; nodesUsed = 1;

		// Reserve space for the maximum number of BVHBuildNode
		bvhBuildNode.resize(inputTriangles.size() * 2 - 1);
		triangleIndexes.resize(inputTriangles.size());

		// Initialize triangleIndexes
		for (int i = 0; i < inputTriangles.size(); i++) {
			triangleIndexes[i] = i;
		}

		// Initialize root node
		BVHBuildNode& root = bvhBuildNode[rootIndex];
		root.leftFirst = rootIndex;
		root.used = inputTriangles.size();

		// Update the root's bounds
		updateBounds(inputTriangles, rootIndex);

		// Subdivide the root node
		subdivide(inputTriangles, rootIndex);
		bvhBuildNode.resize(nodesUsed);
	}

	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles) {
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}

	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection) {
		// Start from the root node
		int currentIndex = rootIndex;
		BVHBuildNode* node = &bvhBuildNode[currentIndex];

		// Stack array for storing BVHBuildNode indexes
		int stack[64]{};
		int stackPointer = 0;

		float tBounds;
		if (!node->bounds.rayAABB(ray, tBounds)) return;
		if (tBounds > intersection.t) return;

		while (true) {
			// Check if the node is child node
			if (node->isLeaf()) {
				for (int i = 0; i < node->used; i++) {
					int triangleIndex = triangleIndexes[node->leftFirst + i];
					float t, u, v;
					if (triangles[triangleIndex].rayIntersect(ray, t, u, v)) {
						if (t > EPSILON && t < intersection.t) {
							intersection.t = t;
							intersection.ID = triangleIndex;
							intersection.alpha = 1.f - (u + v);
							intersection.beta = u;
							intersection.gamma = v;
						}
					}
				}
				if (stackPointer == 0) break;
				else { currentIndex = stack[--stackPointer]; node = &bvhBuildNode[currentIndex]; }
				continue;
			}
			// Node is either left or right child, traverse the least costing child
			int leftChildIndex = node->leftFirst;
			int rightChildIndex = node->leftFirst + 1;

			BVHBuildNode* leftChild = &bvhBuildNode[leftChildIndex];
			BVHBuildNode* rightChild = &bvhBuildNode[rightChildIndex];

			float distanceLeft = 0.f, distanceRight = 0.f;
			if (!leftChild->bounds.rayAABB(ray, distanceLeft) || distanceLeft >= intersection.t) distanceLeft = FLT_MAX;
			if (!rightChild->bounds.rayAABB(ray, distanceRight) || distanceRight >= intersection.t) distanceRight = FLT_MAX;

			if (distanceLeft > distanceRight) {
				std::swap(distanceLeft, distanceRight);
				std::swap(leftChild, rightChild);
				std::swap(leftChildIndex, rightChildIndex);
			}
			if (distanceLeft == FLT_MAX) {
				if (stackPointer == 0) break;
				else { currentIndex = stack[--stackPointer]; node = &bvhBuildNode[currentIndex]; }
			} else {
				node = leftChild;
				if (distanceRight != FLT_MAX) stack[stackPointer++] = rightChildIndex;
			}
		}
	}

	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, const float maxT) {
		// Start from the root node
		int currentIndex = rootIndex;
		BVHBuildNode* node = &bvhBuildNode[currentIndex];

		// Stack array for storing BVHBuildNode indexes
		int stack[64]{};
		int stackPointer = 0;

		float tBounds;
		if (!node->bounds.rayAABB(ray, tBounds)) return true;
		if (tBounds > maxT) return true;

		while (true) {
			// Check if the node is child node
			if (node->isLeaf()) {
				for (int i = 0; i < node->used; i++) {
					int triangleIndex = triangleIndexes[node->leftFirst + i];
					float t, u, v;
					if (triangles[triangleIndex].rayIntersect(ray, t, u, v)) {
						if (t > EPSILON && t < maxT) return false;
					}
				}
				if (stackPointer == 0) break;
				else { currentIndex = stack[--stackPointer]; node = &bvhBuildNode[currentIndex]; }
				continue;
			}
			// Node is either left or right child, traverse the least costing child
			int leftChildIndex = node->leftFirst;
			int rightChildIndex = node->leftFirst + 1;

			BVHBuildNode* leftChild = &bvhBuildNode[leftChildIndex];
			BVHBuildNode* rightChild = &bvhBuildNode[rightChildIndex];

			float distanceLeft = 0.f, distanceRight = 0.f;
			if (!leftChild->bounds.rayAABB(ray, distanceLeft) || distanceLeft >= maxT) distanceLeft = FLT_MAX;
			if (!rightChild->bounds.rayAABB(ray, distanceRight) || distanceRight >= maxT) distanceRight = FLT_MAX;

			if (distanceLeft > distanceRight) {
				std::swap(distanceLeft, distanceRight);
				std::swap(leftChild, rightChild);
				std::swap(leftChildIndex, rightChildIndex);
			}
			if (distanceLeft == FLT_MAX) {
				if (stackPointer == 0) break;
				else { currentIndex = stack[--stackPointer]; node = &bvhBuildNode[currentIndex]; }
			}
			else {
				node = leftChild;
				if (distanceRight != FLT_MAX) stack[stackPointer++] = rightChildIndex;
			}
		}
		return true;
	}
};