#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <queue>
#include <ratio>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "Geometry.h"
#include "Imaging.h"
#include "Lights.h"
#include "Materials.h"
#include "MyMath.h"
#include "QTree.h"
#include "Sampling.h"
#include "Scene.h"

#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

// --- Constants for Path Guiding Algortihm ---
#define PHOTON_MAPPING true
#define GUIDED_PATH false
#define SEARCH_KNN true
#define DEBUG_GUIDED_PATH false
#define DEBUG_BVH true

// Enable NEE or not for Incoming Radiance (Li)
constexpr bool enableNEE = true;

// Max Bounces
static const int MAX_DEPTH = 8;

// Number of Photons to shoot
static const int NUM_OF_PHOTONS_TO_SHOOT = 500000;
static const int N_PHOTONS_GLOBAL = 275;
static const int N_PHOTONS_CAUSTIC = 70;

// Max Child Nodes That PointBVH Can Have
static const int MAX_CHILDNODE_RECORDS = 16;

// Defensive Sampling & Mixture
static const int MAX_NEARBY_VERTICES = 200;     // Aim between 200-800
static const int MIN_ACCEPTED_INSERTIONS = 32;  // Aim between 16-64
static const float BSDF_FRACTION = 0.5f;
static const float QTREE_FRACTION = 0.5f;
// --- Constants for Path Guiding Algortihm End ---

struct ScreenTile {
	// Default values for x, and y tiles, and tile size
	unsigned int x = 0, y = 0;
	unsigned int tile_size = 32;

	// Get start and end indexes accros tiles in x-coordinate
	unsigned int start_tile_x() const { return x; }
	unsigned int end_tile_x(Film* film) const { return std::min(x + tile_size - 1u, film->width - 1u); }

	// Get start and end indexes accros tiles in y-coordinate
	unsigned int start_tile_y() const { return y; }
	unsigned int end_tile_y(Film* film) const { return std::min(y + tile_size - 1u, film->height - 1u); }
};

// --- Struct definition for Photon Mapping [Jensen 1995, 1996] ---
struct Photon {
	Vec4 position;  // Hit Point (shadingData.x)
	Vec4 normal;    // Shading/Geometric Normal
	Vec4 wi;        // Incoming Direction
	Colour flux;    // Photon Flux (This is Energy, NOT Incoming Radiance)
	int key = 0;    // Axis/key for the KD-Tree
};

// Max-Heap: Top element becomes the one with larger distance
struct NearestPhoton {
	float distanceSq;
	int key;
	bool operator<(const NearestPhoton& other) const { return this->distanceSq < other.distanceSq; }
};
// --- Struct definition for Photon Mapping [Jensen 1995, 1996] End ---

// --- Struct definitions for Path Guiding Work ---
struct PathVertex {
	Vec4 position;  // Hit Point (shadingData.x)
	Vec4 normal;    // Shading/Geometric Normal
	Vec4 wi;	    // Incoming Direction
	Colour Li;		// Incoming Radiance
};

struct ForwardPassRecord {
	Vec4 position;			   // Hit Point (shadingData.x)
	Vec4 normal;			   // Shading/Geometric Normal
	Vec4 wi;				   // Incoming Direction

	Colour bsdfWeight;		   // = (fBsdf * cosTheta) / (pdfBsdf * rrp)
	Colour directLighting;	   // Direct Lighting w/NEE (Next Event Estimation)
	Colour emission;		   // Unweighted emitted colour
	Colour misEmission;		   // MIS weight applied to emitted colour

	bool storeRecord = false;  // Do not store if previous surface is pure specular
};

// TO:DO - Modify this according to my NearestPhoton struct so that I can speed up Path Guiding
// Max-Heap: Top element becomes the one with larger distance
struct NearestPathVertex {
	float distanceSq;
	const PathVertex* vertex;
	bool operator<(const NearestPathVertex& otherVertex) const { return this->distanceSq < otherVertex.distanceSq; }
};
// --- Struct definitions for Path Guiding Work End ---

struct PointBVHNodeStats {
	int nodeCount = 0, leafNodeCount = 0;
	int minLeafDepth = INT_MAX, maxLeafDepth = -INT_MAX;
	long long sumLeafDepth = 0;
	size_t memoryInBytes = 0;
	double buildTimeMs = 0.0;
};
// --- Struct definitions for Path Guiding Work End ---

// --- Profiling Structs ---
struct ProfilerStats {
	double bsdfInvertTimeMs = 0.0;
	double bvhSearchTimeMs = 0.0;
	double qTreeBuildTimeMs = 0.0;
	long long guidedPathBounceCount = 0;
};

struct Timer {
	std::chrono::high_resolution_clock::time_point start;
	double& accumulatedTime;

	Timer(double& _accumulatedTime) : accumulatedTime(_accumulatedTime) {
		start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		auto end = std::chrono::high_resolution_clock::now();
		accumulatedTime += std::chrono::duration<double, std::milli>(end - start).count();
	}
};
// --- Profiling Structs End ---

// --- Spatial-Tree Component to Store PathVertex Caches ---
struct PointBVHNode {
	AABB bounds;
	int leftFirst = 0;
	int used = 0;
	bool isLeaf() const { return used > 0; }
};

class PointBVH {
private:
	std::vector<PointBVHNode> nodes;
	std::vector<PathVertex> pathVertices;
	double buildTime = 0.0;

	void updateBounds(int nodeIdx) {
		// Reset the bounds
		nodes[nodeIdx].bounds.reset();
		// Extend the bounds according to the position vector
		for (int first = nodes[nodeIdx].leftFirst, i = 0; i < nodes[nodeIdx].used; i++) {
			nodes[nodeIdx].bounds.extend(pathVertices[first + i].position);
		}
	}
	
	void subdivide(int nodeIdx) {
		// Return if the used node count exceeds max child node count
		if (nodes[nodeIdx].used <= MAX_CHILDNODE_RECORDS) return;

		// Get the extend vector from the AABB bounds
		Vec4 extendVector = nodes[nodeIdx].bounds.max - nodes[nodeIdx].bounds.min;

		// Find the split axis
		int ax = 0;
		if (extendVector.y > extendVector.x) ax = 1;
		if (extendVector.z > extendVector[ax]) ax = 2;

		// Get the first, last, and nth element indexes
		auto first = pathVertices.begin() + nodes[nodeIdx].leftFirst;
		auto nth = pathVertices.begin() + nodes[nodeIdx].leftFirst + nodes[nodeIdx].used / 2;
		auto last = pathVertices.begin() + nodes[nodeIdx].leftFirst + nodes[nodeIdx].used;

		// From those indexes, sort elements via a comparator
		std::nth_element(first, nth, last, 
			// Lambda function as a comparator, capture the split axis by value
			// Unlike scene triangle BVH we do not use centroids
			// Instead, we compare the values of corresponding axis value of position vectors
			[ax](PathVertex& vertex1, PathVertex& vertex2) { return vertex1.position[ax] < vertex2.position[ax]; }
		);

		// Get the middle index
		int currentIndex = nodes[nodeIdx].leftFirst;
		int currentUsed = nodes[nodeIdx].used;
		int middleIndex = currentIndex + currentUsed / 2;

		// Reset the node count for the parent
		nodes[nodeIdx].used = 0;

		// Get left child index
		int leftChildIndex = nodes.size();
		nodes[nodeIdx].leftFirst = leftChildIndex;

		// Create left and right childs
		PointBVHNode leftChild;
		leftChild.leftFirst = currentIndex;
		leftChild.used = currentUsed / 2;

		PointBVHNode rightChild;
		rightChild.leftFirst = middleIndex;
		rightChild.used = currentUsed - leftChild.used;

		// Push back into the list
		nodes.push_back(leftChild);
		nodes.push_back(rightChild);

		// Update AABB bounds
		updateBounds(leftChildIndex);
		updateBounds(leftChildIndex + 1);

		// Subdivide the tree
		subdivide(leftChildIndex);
		subdivide(leftChildIndex + 1);
	}

	int validateNode(int nodeIdx, int depth) const {
		// Retrieve the current node
		const PointBVHNode& node = nodes[nodeIdx];

		// Check for leaf node case
		if (node.isLeaf()) {
			// Check if we have more used records than the max amount
			assert(node.used <= MAX_CHILDNODE_RECORDS && "Leaf carries records over capacity");
			// Check if the record positions are inside the bounding box or not
			for (int first = node.leftFirst, i = 0; i < node.used; i++) {
				assert(node.bounds.containsPoint(pathVertices[i].position, EPSILON) && "Record outside of its leaf's bounding box.");
			}
			return node.used;
		}
		// Get left and right child indexes
		int leftChildIndex = node.leftFirst;
		int rightChildIndex = node.leftFirst + 1;

		// Handle not leaf node case
		assert(node.used == 0 && "Parent node is abruptly carrying records");
		// assert(node->l != nullptr && node->r != nullptr && "Parent node is missing a child node");
		assert(node.bounds.containsAABB(nodes[leftChildIndex].bounds, EPSILON) && "Left child's bounding box is not in parent bounding box.");
		assert(node.bounds.containsAABB(nodes[rightChildIndex].bounds, EPSILON) && "Right child's bounding box is not in parent bounding box.");
		return validateNode(leftChildIndex, depth + 1) + validateNode(rightChildIndex, depth + 1);
	}

	void statsNode(int nodeIdx, PointBVHNodeStats& bvhStats, int depth) {
		// Retrieve the current node
		const PointBVHNode& node = nodes[nodeIdx];

		// Update the node count and memory used
		bvhStats.nodeCount++;
		bvhStats.memoryInBytes += sizeof(PointBVHNode);

		// Check if leaf node
		if (node.isLeaf()) {
			// Update rest of the profiler stats
			bvhStats.leafNodeCount++;
			bvhStats.minLeafDepth = std::min(bvhStats.minLeafDepth, depth);
			bvhStats.maxLeafDepth = std::max(bvhStats.maxLeafDepth, depth);
			bvhStats.sumLeafDepth += depth;
			return;
		}
		// Get left and right child indexes
		int leftChildIndex = node.leftFirst;
		int rightChildIndex = node.leftFirst + 1;

		// Recurse through the child nodes
		statsNode(leftChildIndex, bvhStats, depth + 1);
		statsNode(rightChildIndex, bvhStats, depth + 1);
	}

	void radiusSearchNode(int nodeIdx, Vec4 hitPosition, float radiusSq, std::vector<const PathVertex*>& nearbyVertices) {
		// Retrieve the current node
		const PointBVHNode& node = nodes[nodeIdx];

		// Find the closest point in the AABB
		Vec4 closestPoint;
		closestPoint.x = std::max(std::min(hitPosition.x, node.bounds.max.x), node.bounds.min.x);
		closestPoint.y = std::max(std::min(hitPosition.y, node.bounds.max.y), node.bounds.min.y);
		closestPoint.z = std::max(std::min(hitPosition.z, node.bounds.max.z), node.bounds.min.z);

		// Calculate distance square to prevent sqrt and omit the ones less than radiusSq
		float distanceSq = (SQ(closestPoint.x - hitPosition.x) + SQ(closestPoint.y - hitPosition.y) + SQ(closestPoint.z - hitPosition.z));
		if (distanceSq > radiusSq) return;

		if (node.isLeaf()) {
			for (int first = node.leftFirst, i = 0; i < node.used; i++) {
				const PathVertex& vertex = pathVertices[first + i];
				float distanceSq2 = (SQ(vertex.position.x - hitPosition.x) + SQ(vertex.position.y - hitPosition.y) + SQ(vertex.position.z - hitPosition.z));
				if (distanceSq2 <= radiusSq) nearbyVertices.push_back(&vertex);
			}
			return;
		}

		// Get left and right child indexes
		int leftChildIndex = node.leftFirst;
		int rightChildIndex = node.leftFirst + 1;
		
		// Recurse
		radiusSearchNode(leftChildIndex, hitPosition, radiusSq, nearbyVertices);
		radiusSearchNode(rightChildIndex, hitPosition, radiusSq, nearbyVertices);
	}

	void kNNSearchNode(int nodeIdx, Vec4 hitPosition, float& dynamicRadiusSq, std::priority_queue<NearestPathVertex>& maxHeap, int k) {
		const PointBVHNode& node = nodes[nodeIdx];
		// We are in the leaf node of the list, can search now
		if (node.isLeaf()) {
			for (int first = node.leftFirst, i = 0; i < node.used; i++) {
				const PathVertex& vertex = pathVertices[first + i];
				float distanceSq = (SQ(vertex.position.x - hitPosition.x) + SQ(vertex.position.y - hitPosition.y) + SQ(vertex.position.z - hitPosition.z));
				if (distanceSq < dynamicRadiusSq || maxHeap.size() < k) {
					// Push the new nearest vertex in the heap
					NearestPathVertex nearestVertex{ distanceSq, &vertex };
					maxHeap.push(nearestVertex);
					// If we exceed k, discard immediately
					if (maxHeap.size() > k)	maxHeap.pop();
					// Adjust the radius if the heap is full
					if (maxHeap.size() == k) dynamicRadiusSq = maxHeap.top().distanceSq;
				}
			}
			return;
		}

		// Get left and right child indexes
		int leftChildIndex = node.leftFirst;
		int rightChildIndex = node.leftFirst + 1;

		// We are not in the leaf, calculate left and right child distance
		Vec4 leftClosestPoint;
		leftClosestPoint.x = std::max(std::min(hitPosition.x, nodes[leftChildIndex].bounds.max.x), nodes[leftChildIndex].bounds.min.x);
		leftClosestPoint.y = std::max(std::min(hitPosition.y, nodes[leftChildIndex].bounds.max.y), nodes[leftChildIndex].bounds.min.y);
		leftClosestPoint.z = std::max(std::min(hitPosition.z, nodes[leftChildIndex].bounds.max.z), nodes[leftChildIndex].bounds.min.z);
		float distanceSqLeftChild = (SQ(leftClosestPoint.x - hitPosition.x) + SQ(leftClosestPoint.y - hitPosition.y) + SQ(leftClosestPoint.z - hitPosition.z));

		Vec4 rightClosestPoint;
		rightClosestPoint.x = std::max(std::min(hitPosition.x, nodes[rightChildIndex].bounds.max.x), nodes[rightChildIndex].bounds.min.x);
		rightClosestPoint.y = std::max(std::min(hitPosition.y, nodes[rightChildIndex].bounds.max.y), nodes[rightChildIndex].bounds.min.y);
		rightClosestPoint.z = std::max(std::min(hitPosition.z, nodes[rightChildIndex].bounds.max.z), nodes[rightChildIndex].bounds.min.z);
		float distanceSqRightChild = (SQ(rightClosestPoint.x - hitPosition.x) + SQ(rightClosestPoint.y - hitPosition.y) + SQ(rightClosestPoint.z - hitPosition.z));
		
		// Branch to decide which child is the closest, and what order to traverse
		if (distanceSqLeftChild < distanceSqRightChild) {
			// Left Child is more closer, traverse left first and then right
			if (distanceSqLeftChild <= dynamicRadiusSq) kNNSearchNode(leftChildIndex, hitPosition, dynamicRadiusSq, maxHeap, k);
			if (distanceSqRightChild <= dynamicRadiusSq) kNNSearchNode(rightChildIndex, hitPosition, dynamicRadiusSq, maxHeap, k);
		} else {
			// Right Child is more closer, traverse right first and then left
			if (distanceSqRightChild <= dynamicRadiusSq) kNNSearchNode(rightChildIndex, hitPosition, dynamicRadiusSq, maxHeap, k);
			if (distanceSqLeftChild <= dynamicRadiusSq) kNNSearchNode(leftChildIndex, hitPosition, dynamicRadiusSq, maxHeap, k);
		}
	}
public:
	// Public Methods
	void build(std::vector<PathVertex>&& inputPathVertices) {
		// Handle degenerate case where the passed vector is empty
		if (inputPathVertices.empty()) return;
		pathVertices = std::move(inputPathVertices);

		nodes.clear();
		size_t allocatedNodeSpace = 4 * (pathVertices.size() / MAX_CHILDNODE_RECORDS) + 1;
		nodes.reserve(allocatedNodeSpace);

		// Initialize root
		int rootIdx = 0;
		nodes.push_back(PointBVHNode());
		nodes[rootIdx].leftFirst = 0;
		nodes[rootIdx].used = (int)(pathVertices.size());

		// Time the build time using chrono
		auto start = std::chrono::high_resolution_clock::now();

		// Update bounds and subdivide the root node
		updateBounds(rootIdx);
		subdivide(rootIdx);

		// End the timing
		auto end = std::chrono::high_resolution_clock::now();

		// Save the build time, and call validate and stats
		buildTime = std::chrono::duration<double, std::milli>(end - start).count();
		#if DEBUG_BVH
		validate(rootIdx);
		stats(rootIdx);
		#endif
	}

	void validate(int nodeIdx) {
		// Start from the root node, depth at 0
		int counted = validateNode(nodeIdx, 0);
		assert(counted == (int)(pathVertices.size()) && "Records lost or duplicated during the build phase.");
	}

	PointBVHNodeStats stats(int nodeIdx) {
		PointBVHNodeStats bvhStats;
		bvhStats.memoryInBytes = pathVertices.size() * sizeof(PathVertex);
		statsNode(nodeIdx, bvhStats, 0);
		bvhStats.buildTimeMs = buildTime;
		std::cout << "PointBVHNode["
			<< "\n  -- path vertex records: " << pathVertices.size()
			<< "\n  -- nodes: " << bvhStats.nodeCount
			<< "\n  -- leaf nodes: " << bvhStats.leafNodeCount
			<< "\n  -- depth: " << bvhStats.minLeafDepth << "-" << bvhStats.maxLeafDepth
			<< " (mean " << (double)bvhStats.sumLeafDepth / (double)bvhStats.leafNodeCount << ")"
			<< "\n  -- size: " << bvhStats.memoryInBytes / SQ(1024.0) << " MB"
			<< "\n  -- build time: " << bvhStats.buildTimeMs << " ms\n]\n";
		return bvhStats;
	}

	// --- New Method ---
	void radiusSearch(Vec4 hitPosition, float radiusSq, std::vector<const PathVertex*>& nearbyVertices) {
		// Start the radius search from the root
		if (nodes.empty()) return;
		int rootIdx = 0;
		radiusSearchNode(rootIdx, hitPosition, radiusSq, nearbyVertices);
	}

	void kNNSearch(Vec4 hitPosition, float dynamicRadiusSq, std::priority_queue<NearestPathVertex>& maxHeap, int k) {
		// Start the kNN search from the root
		if (nodes.empty()) return;
		int rootIdx = 0;
		kNNSearchNode(rootIdx, hitPosition, dynamicRadiusSq, maxHeap, k);
	}
};
// --- Spatial-Tree Component to Store PathVertex Caches End ---

// --- KD-Tree for Photon Mapping [Jensen 1995, 1996] Start ---
class PhotonMap {
private:
	// Attribute - List to store all Photons
	std::vector<Photon> photons;

	// Methods
	void buildRecursive(int startIdx, int endIdx, int depth) {
		// If start index is bigger or equal than end index, no point in this
		if (startIdx >= endIdx) return;

		// Determine the split axis
		int ax = depth % 3;
		int midIdx = startIdx + (endIdx - startIdx) / 2;

		// Get the first, last, and nth element indexes (similar to the PointBVH for my Path Guiding)
		auto first = photons.begin() + startIdx;
		auto nth = photons.begin() + midIdx;
		auto last = photons.begin() + endIdx;

		// From those indexes, sort elements via a comparator
		std::nth_element(first, nth, last,
			// Lambda function as a comparator, capture the split axis by value
			// Compare the values of corresponding axis value of position vectors (similar to the PointBVH for my Path Guiding)
			[ax](Photon& photon1, Photon& photon2) { return photon1.position[ax] < photon2.position[ax]; }
		);

		// Assign the key index to the middle Photon
		photons[midIdx].key = ax;

		// Recursive build
		buildRecursive(startIdx, midIdx, depth + 1);
		buildRecursive(midIdx + 1, endIdx, depth + 1);
	}

	void searchKNN(int startIdx, int endIdx, int maxPhotons, const Vec4& targetPosition, float& maxDistanceSq, std::priority_queue<NearestPhoton>& nearestPhotons) {
		// If start index is bigger or equal than end index, no point in this
		if (startIdx >= endIdx) return;

		// Obtain the middle index, and the corresponding Photon
		int midIdx = startIdx + (endIdx - startIdx) / 2;
		const Photon& photon = photons[midIdx];

		// Calculate distance between the photon and targetPosition
		float distanceSq = (targetPosition - photon.position).lengthSquare();
		if (distanceSq < maxDistanceSq) {
			// Push in the max heap
			nearestPhotons.push({ distanceSq, midIdx });
			int heapSize = (int)(nearestPhotons.size());

			// If max-heap size is bigger than the limit, pop and adjust the max distance
			if (heapSize > maxPhotons) {
				nearestPhotons.pop();
				maxDistanceSq = nearestPhotons.top().distanceSq;
			}
		}

		// Which side of the plane to split?
		float axisDistance = 0.f;
		if (photon.key == 0) axisDistance = targetPosition.x - photon.position.x;
		else if (photon.key == 1) axisDistance = targetPosition.y - photon.position.y;
		else if (photon.key == 2) axisDistance = targetPosition.z - photon.position.z;

		// Left and right childs
		int leftStartIdx = startIdx, leftEndIdx = midIdx;
		int rightStartIdx = midIdx + 1, rightEndIdx = endIdx;

		// Recursive traverse based on nearest child
		if (axisDistance < 0.f) {
			searchKNN(leftStartIdx, leftEndIdx, maxPhotons, targetPosition, maxDistanceSq, nearestPhotons);
			if (axisDistance * axisDistance < maxDistanceSq) {
				searchKNN(rightStartIdx, rightEndIdx, maxPhotons, targetPosition, maxDistanceSq, nearestPhotons);
			}
		} else {
			searchKNN(rightStartIdx, rightEndIdx, maxPhotons, targetPosition, maxDistanceSq, nearestPhotons);
			if (axisDistance * axisDistance < maxDistanceSq) {
				searchKNN(leftStartIdx, leftEndIdx, maxPhotons, targetPosition, maxDistanceSq, nearestPhotons);
			}
		}
	}
public:
	// Methods
	void build() {
		// If we don't have any photons, we cannot build the KD-Tree
		if (photons.empty()) return;
		int rootIdx = 0, lastElementIdx = (int)(photons.size());
		buildRecursive(rootIdx, lastElementIdx, 0);
	}

	// .size() returns size_t, which is an unsigned int give or take
	size_t getSize() const { return photons.size(); }

	// Need an insert and clear method since the vector is a private attribute!
	void insertPhoton(Photon& photon) { photons.push_back(photon); }
	void clearMap() { photons.clear(); }

	Colour estimateRadiance(ShadingData& shadingData, const Vec4& position, const Vec4& normal, int maxPhotons = 100, float maxSearchRadius = 1.f) {
		// We cannot estimate radiance if we don't have Photons
		if (photons.empty()) return Colour(0.f, 0.f, 0.f);

		// Define the variables we'll pass to the kNN-Search
		std::priority_queue<NearestPhoton> nearestPhotons;
		float maxDistanceSq = maxSearchRadius * maxSearchRadius;

		// Do kNN-Search
		int startIdx = 0, endIdx = (int)(photons.size());
		searchKNN(startIdx, endIdx, maxPhotons, position, maxDistanceSq, nearestPhotons);
		if (nearestPhotons.empty()) return Colour(0.f, 0.f, 0.f);
		
		float radiusSq = maxDistanceSq;
		if (radiusSq < EPSILON) return Colour(0.f, 0.f, 0.f);

		// Accumulate Radiance
		float maxRadius = std::sqrt(radiusSq);
		Colour accumulatedRadiance(0.f, 0.f, 0.f);

		while (!nearestPhotons.empty()) {
			// Pop the top Photon
			int photonIdx = nearestPhotons.top().key;
			float distSq = nearestPhotons.top().distanceSq;
			const Photon& photon = photons[photonIdx];
			nearestPhotons.pop();

			// Do not contribute this Photon to the Accumulate Radiance
			if (Dot(photon.normal, shadingData.sNormal) < 0.f) continue;

			// Cone Filter [Jensen 1996]
			// I'll assume filter constant, k, is 1
			float distance = std::sqrt(distSq);
			float weight = std::max(0.f, 1.f - (distance / maxRadius));
			
			// Density Estimation
			Colour fr = shadingData.bsdf->evaluate(shadingData, photon.wi);
			accumulatedRadiance = accumulatedRadiance + fr * photon.flux *  weight;
		}
		// Divide by area and return Accumulated Radiance
		// Cone Filter Normalization Factor: 1 - (2/3) = 1/3
		float invArea = 3.f / (M_PI * radiusSq);
		return accumulatedRadiance * invArea;
	}
};
// --- KD-Tree for Photon Mapping [Jensen 1995, 1996] End ---

class RayTracer {
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom* samplers;
	std::thread** threads;
	unsigned int numProcs;

	// Cached vertices will be stored in a BVH structure
	PointBVH* cacheBVH;
	std::vector<PathVertex> globalCacheList;
	// Ground Truth: 8192/16384					(should really be an absurd number to eliminate variance)
	// Testing SPPs: 128/256/512/1024/2048/4096 (render, and compare error metrics with the groung truth)
	// int maxSPP = 8192;
	int maxSPP = 128;
	int learningThreshold = maxSPP / 8;

	// Path Vertex vector to then cache saved items over at a Spatial Accelleration Structure
	std::vector<std::vector<PathVertex>> perThreadPathVertexRecords;
	QTree* perThreadQTrees;

	// Stats for Profiling
	std::vector<ProfilerStats> perThreadStats;

	// Photon Mapping
	PhotonMap globalPhotonMap;
	PhotonMap causticPhotonMap;

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas) {
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];
		perThreadQTrees = new QTree[numProcs];
		perThreadPathVertexRecords.resize(numProcs);
		perThreadStats.resize(numProcs);
		clear();
		// Assuming our scene is ready to go
		// Shoot Photons here for Photon Mapping
		#if PHOTON_MAPPING
		std::cout << "Photon Mapping First Pass: Shooting Photons...\n";
		shootPhotons(samplers, NUM_OF_PHOTONS_TO_SHOOT);
		std::cout << "Photon Mapping First Pass Completed! Global Photon Map Size: " << globalPhotonMap.getSize() 
				  << " | Caustic Photon Map Size: " << causticPhotonMap.getSize() << std::endl;
		#endif
	}

	void clear() {
		film->clear();
	}

	Colour computeDirect(const ShadingData& shadingData, Sampler* sampler, const std::function<float(const Vec4&)>& pdfFunction = nullptr) {
		// Is surface is specular we cannot computing direct lighting
		if (shadingData.bsdf->isPureSpecular() == true) return Colour(0.f, 0.f, 0.f);
		
		// Sample Light
		float pmfLight = 0.f;
		Light* light = scene->sampleLight(sampler, pmfLight);
		if (light == nullptr || pmfLight <= 0.f) return Colour(0.f, 0.f, 0.f);
		
		// Area Light
		if (light->isArea()) {
			// Sample point on light and store returned emission
			float pdfLight = 0.f;
			Colour emittedColour;
			Vec4 pointOnLight = light->sample(shadingData, sampler, emittedColour, pdfLight);
			if (pdfLight <= 0.f) return Colour(0.f, 0.f, 0.f);

			// Get surface to light and direction to light
			Vec4 surfaceToLight = pointOnLight - shadingData.x;
			Vec4 wi = surfaceToLight.normalize();

			// Calculate Geometry Term
			float denominator = surfaceToLight.lengthSquare();
			if (denominator < EPSILON) return Colour(0.f, 0.f, 0.f);
			float cosTheta = std::max(Dot(wi, shadingData.sNormal), 0.f);
			float cosThetaPrime = std::max(Dot(-wi, light->normal(shadingData, wi)), 0.f);
			if (cosTheta <= 0.f || cosThetaPrime <= 0.f) return Colour(0.f, 0.f, 0.f);
			float geometryTerm = (cosTheta * cosThetaPrime) / denominator;

			// Calculate visibility
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Vec4 shadowRayOffset(shadingData.x + shadingData.gNormal * (EPSILON * sign));
			if (scene->visible(shadowRayOffset, pointOnLight)) {
				// Evaluate BSDF and PDF of it
				Colour bsdf = shadingData.bsdf->evaluate(shadingData, wi);

				// Using pdfFunction if we ever use Path Guiding for Defensive Sampling MIS
				float pdfBsdf = (pdfFunction != nullptr) ? pdfFunction(wi) : shadingData.bsdf->PDF(shadingData, wi);

				// Calculate Weight for MIS
				float pALight = pdfLight * pmfLight;
				if (pALight <= 1e-8f) return Colour(0.f, 0.f, 0.f);
				float pABsdf = pdfBsdf * cosThetaPrime / denominator;
				float wd = weightPowerHeuristics(pALight, pABsdf);

				// Multiply terms, divide by pALight, and return
				return (emittedColour * bsdf * geometryTerm * wd) / pALight;
			}
			return Colour(0.f, 0.f, 0.f);
		}
		// Environment Map
		else {
			// Sample from light, returns direction instead of point
			float pdfLight = 0.f;
			Colour emittedColour;
			Vec4 wi = light->sample(shadingData, sampler, emittedColour, pdfLight);
			if (pdfLight <= 0.f) return Colour(0.f, 0.f, 0.f);

			// Evaluate Geometry Term for environment maps
			// It's just cosTheta because there is no next surface
			float cosTheta = std::max(Dot(wi, shadingData.sNormal), 0.f);
			if (cosTheta <= 0.f) return Colour(0.f, 0.f, 0.f);

			// Evaluate visibility to outside scene bounds
			// Scene Bounds Fix: Replaces calculated SceneBounds AABB length with SceneBounds sceneRadius and sceneCentre
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Vec4 shadowRayOffset(shadingData.x + shadingData.gNormal * (EPSILON * sign));
			Vec4 sceneBoundOffset = shadingData.x + (wi * (2.f * use<SceneBounds>().sceneRadius));
			if (scene->visible(shadowRayOffset, sceneBoundOffset)) {
				// Evaluate BSDF and multiply terms and return
				Colour bsdf = shadingData.bsdf->evaluate(shadingData, wi);

				// Evaluate pALight and pABsdf for MIS - cosTheta is pretty much geometry term for Environment Mapping so pdfBsdf turns into pABsdf
				float pALight = pdfLight * pmfLight;
				if (pALight <= 1e-8f) return Colour(0.f, 0.f, 0.f);

				// Using pdfFunction if we ever use Path Guiding for Defensive Sampling MIS
				float pABsdf = (pdfFunction != nullptr) ? pdfFunction(wi) : shadingData.bsdf->PDF(shadingData, wi);
				
				// Calculate Weight for MIS
				float wd = weightPowerHeuristics(pALight, pABsdf);

				// Multiply terms, divide by pALight, and return
				return (emittedColour * bsdf * cosTheta * wd) / pALight;
			}
			return Colour(0.f, 0.f, 0.f);
		}
	}

	Colour direct(Ray& r, Sampler* sampler) {
		// Compute direct lighting for an image sampler here
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) return shadingData.bsdf->emit(shadingData, shadingData.wo);
			return computeDirect(shadingData, sampler);
		}
		return scene->background->evaluate(r.dir);
	}

	// --- Photon Mapping [Jensen 1995, 1996] Functions Start ---
	// First Pass - Shoot Photons
	// Emit photons
	// Trace photons
	// Store in Photon Map
	void shootPhotons(Sampler* sampler, int numPhotons) {
		// Shoot fixed number photons
		// For each photon
		//  • Sample light
		//  • Sample outgoing direction
		//  • Recursively trace into the scene
		//  • Similar to light tracing
		globalPhotonMap.clearMap();
		causticPhotonMap.clearMap();
		for (int i = 0; i < numPhotons; i++) {
			// Sample a light
			float pmfLight = 0.f;
			Light* light = scene->sampleLight(sampler, pmfLight);
			if (light == nullptr || pmfLight <= 0.f) continue;

			// Sampled light is an area light
			if (light->isArea()) {
				// Sample outgoing direction
				float pdfDirection = 0.f;
				Vec4 wi = light->sampleDirectionFromLight(sampler, pdfDirection);
				if (pdfDirection <= 0.f) continue;

				// Sample position
				float pdfPosition = 0.f;
				Vec4 position = light->samplePositionFromLight(sampler, pdfPosition);
				if (pdfPosition <= 0.f) continue;

				// Get emission (Le)
				Colour emission = light->evaluate(-wi);
				if (emission.Lum() < 1e-8f) continue;

				// Sample light normal
				Vec4 lightNormal = light->normal(ShadingData(position, wi), wi);

				// Calculate beta
				float cosine = std::fabs(Dot(wi, lightNormal));
				float invPA = 1.f / (pdfDirection * pdfPosition * pmfLight * (float)numPhotons);
				Colour flux = emission * cosine * invPA;

				int sign = (Dot(wi, lightNormal) > 0.f) ? 1 : -1;
				Ray r(position + lightNormal * sign * EPSILON, wi);
				tracePhotonPath(r, sampler, flux, 0);
			}
		}
		// TO:DO - Sampled Light is an Environment Map
		globalPhotonMap.build();
		causticPhotonMap.build();
	}

	// Second Pass - Ray Trace From Eye
	void tracePhotonPath(Ray& r, Sampler* sampler, Colour flux, int depth, bool isCausticPath = false) {
		// Depends on the material
		// Trace rays further OR
		//  • Access photon map
		//  • Compute Density estimation
		if (depth >= MAX_DEPTH) return;
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			// Return is it's a light
			if (shadingData.bsdf->isLight()) return;

			// Store the Photon
			bool isSpecular = shadingData.bsdf->isPureSpecular();
			if (depth > 0 && !isSpecular) {
				Photon photon;
				photon.position = shadingData.x;
				photon.normal = shadingData.sNormal;
				photon.wi = -r.dir;				
				photon.flux = flux;

				// Check condition to store in caustic or global photon map
				if (isCausticPath) causticPhotonMap.insertPhoton(photon);
				else globalPhotonMap.insertPhoton(photon);
			}

			// Sample BSDF
			float pdfBsdf = 0.f;
			Colour fBsdf(0.f, 0.f, 0.f);
			Vec4 wiNext = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfBsdf);
			if (pdfBsdf <= 0.f) return;

			// Calculate new flux
			float cosTheta = std::fabs(Dot(shadingData.sNormal, wiNext));
			Colour bounceWeight = fBsdf * (cosTheta / pdfBsdf);

			// Apply Russian Roulette
			if (depth >= 3) {
				float maxChannel = std::max({ bounceWeight.r, bounceWeight.g, bounceWeight.b });
				float rrp = std::min(std::max(maxChannel, 0.05f), 0.95f);
				if (sampler->next() > rrp) return;
				bounceWeight = bounceWeight / rrp;
			}
			
			flux = flux * bounceWeight;
			if (flux.Lum() < 1e-8f) return;
			
			int sign = (Dot(wiNext, shadingData.gNormal) > 0.f) ? 1 : -1;
			Ray nextRay(shadingData.x + shadingData.gNormal * sign * EPSILON, wiNext);
			bool nextIsCaustic = (depth == 0) ? isSpecular : (isCausticPath && isSpecular);
			tracePhotonPath(nextRay, sampler, flux, depth + 1, nextIsCaustic);
		}
	}

	Colour traceCameraRay(Ray& r, Sampler* sampler, int depth = 0, bool isFinalGatherRay = false, bool isPreviousSpecular = false, float previousBsdfPdf = 0.f) {
		if (depth >= MAX_DEPTH) return Colour(0.f, 0.f, 0.f);
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			// First depth ternary check to avoid double emission contribution
			if (shadingData.bsdf->isLight()) {
				Colour emission = shadingData.bsdf->emit(shadingData, shadingData.wo);
				if (depth == 0 || isPreviousSpecular) return emission;
			
				// Evaluate MIS for Area Light
				// Area Light PDF and PMF
				float pmfLight = 1.f / scene->lights.size();
				float pdfLight = 1.f / scene->triangles[intersection.ID].area;

				float cosThetaPrime = std::max(Dot(-r.dir, scene->triangles[intersection.ID].gNormal()), 0.f);
				if (cosThetaPrime <= 0.f) return Colour(0.f, 0.f, 0.f);
				float distanceSquare = SQ(intersection.t);
				if (distanceSquare < EPSILON) return Colour(0.f, 0.f, 0.f);

				// Calculate pA of Light and BSDF for MIS
				float pALight = pdfLight * pmfLight;
				float pABsdf = previousBsdfPdf * cosThetaPrime / distanceSquare;

				// Calculate Weight for MIS
				float wind = weightPowerHeuristics(pABsdf, pALight);
				return emission * wind;
			}

			// Perfect Specular or Glossy Surfaces
			bool requiresRayTrace = (shadingData.bsdf->isPureSpecular() || shadingData.bsdf->isHighlyGlossy());
			if (requiresRayTrace) {
				// Specular Surface
				float pdfBsdf = 0.f;
				Colour fBsdf(0.f, 0.f, 0.f);
				Vec4 wiNext = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfBsdf);
				
				if (pdfBsdf <= EPSILON) return Colour(0.f, 0.f, 0.f);
				
				int sign = (Dot(wiNext, shadingData.gNormal) > 0.f) ? 1 : -1;
				Ray nextRay(shadingData.x + shadingData.gNormal * sign * EPSILON, wiNext);
				Colour incoming = traceCameraRay(nextRay, sampler, depth + 1, isFinalGatherRay, true, pdfBsdf);
				
				float cosTheta = std::fabs(Dot(shadingData.sNormal, wiNext));
				return incoming * fBsdf * (cosTheta / pdfBsdf);
			}

			// Diffuse Surfaces - Compute Direct Lighting
			Colour direct = computeDirect(shadingData, sampler);

			// Radius for Caustics & Compute Indirect Lighting for Caustic Photon Map
			float sceneDiagonal = (scene->bounds.max - scene->bounds.min).length();
			float causticRadius = sceneDiagonal * 0.005f;
			Colour indirectCaustic = causticPhotonMap.estimateRadiance(shadingData, shadingData.x, shadingData.sNormal, N_PHOTONS_CAUSTIC, causticRadius);

			// Compute Indirect Lighting for Global Photon Map
			Colour indirectGlobal(0.f, 0.f, 0.f);
			if (!isFinalGatherRay) {
				// Final Gathering Stage to Battle with Blurry Images and Light Leaks
				// Shoot extra rays from first non-specular hit point
				float pdfBsdf = 0.f;
				Colour fBsdf(0.f, 0.f, 0.f);
				Vec4 wiNext = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfBsdf);

				if (pdfBsdf > EPSILON) {
					// Shoot final gather ray and apply the formula as usual
					float cosTheta = std::fabs(Dot(shadingData.sNormal, wiNext));
					int sign = (Dot(wiNext, shadingData.gNormal) > 0.f) ? 1 : -1;
					Ray finalGatherRay(shadingData.x + shadingData.gNormal * sign * EPSILON, wiNext);
					Colour incomingRadiance = traceCameraRay(finalGatherRay, sampler, depth + 1, true, false, pdfBsdf);
					indirectGlobal = incomingRadiance * fBsdf * (cosTheta / pdfBsdf);
				}
			} else {
				// Extra bounce is over, we can estimate the density radiance now at secondary hit point/s!
				float globalRadius = sceneDiagonal * 0.03f;
				indirectGlobal = globalPhotonMap.estimateRadiance(shadingData, shadingData.x, shadingData.sNormal, N_PHOTONS_GLOBAL, globalRadius);
			}
			// Combine results - Ld + Lg + Lc
			return direct + indirectGlobal + indirectCaustic;
		}
		return scene->background->evaluate(r.dir);
	}
	// --- Photon Mapping [Jensen 1995, 1996] Functions End ---

	// --- Path Guiding Algorithm Work Start ---
	Colour guidedPath(Ray& r, Sampler* sampler, std::vector<PathVertex>& pathVertices, PointBVH* sTree, QTree& dTree, bool isGuidingPhase, ProfilerStats& stats, bool enableNEE) {
		// --- 1. Forward Pass Phase ---
		thread_local std::vector<ForwardPassRecord> records;
		thread_local std::vector<const PathVertex*> nearbyVertices;
		thread_local std::priority_queue<NearestPathVertex> maxHeap;
		records.reserve(10);                          // Max depth: 8, + 2 buffer space
		nearbyVertices.reserve(MAX_NEARBY_VERTICES);  // Pre-allocate max number of wanted vertices
		
		// Clear these vectors and the max heap from the previous bounce just in case
		records.clear();
		nearbyVertices.clear();
		while (!maxHeap.empty()) { maxHeap.pop(); }

		// Generate the path vertices in the forward pass (only populates the vector)
		generatePathRecursive(r, 0, sampler, records, sTree, dTree, isGuidingPhase, nearbyVertices, maxHeap, stats);

		// --- 2. Backpropagation Phase ---
		// Store Each Path Vertex to the vector via Backpropagation
		Colour guidingRadiance(0.f, 0.f, 0.f);
		for (int i = (int)(records.size() - 1); i >= 0; i--) {
			if (!isGuidingPhase) {
				// Learning Phase - Data Collection
				if (records[i].storeRecord) {
					PathVertex pathVertex;
					pathVertex.position = records[i].position;
					pathVertex.normal = records[i].normal;
					pathVertex.wi = records[i].wi;
					pathVertex.Li = guidingRadiance;
					if (pathVertex.Li.isValid() && pathVertex.Li.Lum() > 0) pathVertices.push_back(pathVertex);
				}
			}
			// Update the incoming and guiding radiance so that Li-1 can use this previous Li
			Colour emissionAndDirect = enableNEE ? (records[i].misEmission + records[i].directLighting) : records[i].emission;
			guidingRadiance = emissionAndDirect + (records[i].bsdfWeight * guidingRadiance);
		}
		return guidingRadiance.isValid() ? guidingRadiance : Colour(0.f, 0.f, 0.f);
	}

	// --- NEW METHOD ---
	// -> When sampling, replace BSDF with the new method, i.e.
	//    -> Search for nearby vertices from 1.
	//    -> Project wi into PSS
	//    -> Invert BSDF sampling
	//    -> Sample PSS
	void generatePathRecursive(Ray& r, int depth, Sampler* sampler, std::vector<ForwardPassRecord>& records, PointBVH* cache, QTree& qTree, bool isGuidingPhase, std::vector<const PathVertex*>& nearbyVertices, std::priority_queue<NearestPathVertex>& maxHeap, ProfilerStats& stats, float previousBsdfPdf = 0.f, bool previousSurfaceSpecular = false) {
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		// Create an empty record for now, will fill in later in this function
		ForwardPassRecord record;
		record.position = (shadingData.t < FLT_MAX) ? shadingData.x : Vec4(0.f, 0.f, 0.f);
		record.normal = (shadingData.t < FLT_MAX) ? shadingData.sNormal : Vec4(0.f, 0.f, 0.f);
		record.wi = Vec4(0.f, 0.f, 0.f);
		record.bsdfWeight = Colour(0.f, 0.f, 0.f);
		record.directLighting = Colour(0.f, 0.f, 0.f);
		record.emission = Colour(0.f, 0.f, 0.f);
		record.misEmission = Colour(0.f, 0.f, 0.f);
		record.storeRecord = false;

		if (shadingData.t < FLT_MAX) {
			// --- Emitter has been hit, do emission and MIS contribution ---
			if (shadingData.bsdf->isLight()) {
				Colour emittedColour = shadingData.bsdf->emit(shadingData, shadingData.wo);
				record.emission = emittedColour;
				if (depth == 0 || previousSurfaceSpecular) { record.misEmission = emittedColour; }
				else {
					// --- Evaluate MIS for Area Light ---
					// Area Light PDF and PMF
					float pmfLight = 1.f / scene->lights.size();
					float pdfLight = 1.f / scene->triangles[intersection.ID].area;

					// Calculate cosine term and distance
					float cosThetaPrime = std::max(Dot(-r.dir, scene->triangles[intersection.ID].gNormal()), 0.f);
					float distanceSquare = SQ(intersection.t);

					if (distanceSquare > EPSILON) {
						// Calculate pA of Light and BSDF for MIS
						float pALight = pdfLight * pmfLight;
						float pABsdf = previousBsdfPdf * cosThetaPrime / distanceSquare;

						// Calculate Weight for MIS
						float wind = weightPowerHeuristics(pABsdf, pALight);
						record.misEmission = emittedColour * wind;
					}
				}
				records.push_back(record);
				return;
			}

			// Store if the current surface is specular
			bool isSpecular = shadingData.bsdf->isPureSpecular();

			// Why have I applied RRP at the last depth...
			// Terminate when the ray depth exceeds 8 bounces, to avoid infinite recursion
			// We will work on SD-domain unlike Guo et al. 2018, in which they were restricted with n = m = 2
			if (depth >= MAX_DEPTH) {
				record.directLighting = computeDirect(shadingData, sampler, [](const Vec4&) { return 0.f; });
				records.push_back(record);
				return;
			}

			// Must run Path Guiding before calculating direct lighting
			bool usePathGuiding = false;
			float best_lobe_selection = sampler->next();
			if (isGuidingPhase && cache != nullptr && !isSpecular) {
				// We are in the Path Guiding Phase
				#if SEARCH_KNN
				// BVH will do kNN Search when retrieving nearest vertices (faster)
				float radiusSq = FLT_MAX;
				nearbyVertices.clear();
				// BVH Nearest Path Vertices Search will be timed inside this scope
				{
					Timer bvhSearchTimer(stats.bvhSearchTimeMs);
					cache->kNNSearch(shadingData.x, radiusSq, maxHeap, MAX_NEARBY_VERTICES);
				}

				while (!maxHeap.empty()) {
					const PathVertex* poppedVertex = maxHeap.top().vertex;
					nearbyVertices.push_back(poppedVertex);
					maxHeap.pop();
				}
				#else
				// BVH will do radius based search when retrieving nearest vertices (slower)
				float radius = 0.05f;
				float radiusSq = radius * radius;
				nearbyVertices.clear();

				// Define a Random Device and Initialize the Mersenne Twister to Generate RNGs
				thread_local std::mt19937 mt{ std::random_device {}() };

				// BVH Nearest Path Vertices Search will be timed inside this scope
				{
					Timer bvhSearchTimer(stats.bvhSearchTimeMs);
					cache->radiusSearch(shadingData.x, radiusSq, nearbyVertices);
					if (nearbyVertices.size() > MAX_NEARBY_VERTICES) {
						// Shuffle and resize the vector list we obtain from BVH Search
						std::shuffle(std::begin(nearbyVertices), std::end(nearbyVertices), mt);
						nearbyVertices.resize(MAX_NEARBY_VERTICES);
					}
				}
				#endif

				// Debugging
				#if DEBUG_GUIDED_PATH
				thread_local int debugSearchCount = 0;
				thread_local int debugTotalVertexCount = 0;

				debugSearchCount++;
				debugTotalVertexCount += nearbyVertices.size();

				if (debugSearchCount == 10000) {
					int average = debugTotalVertexCount / 10000;
					printf("Thread Debug: Avg. Vertices Found Per Bounce: %d\n", average);
					debugSearchCount = 0;
					debugTotalVertexCount = 0;
				}
				#endif
				// Debugging End

				int acceptedVertices = 0;
				float totalInsertedWeight = 0.f;
				// QTree Building will be timed inside this scope
				{
					Timer qTreeBuildTimer(stats.qTreeBuildTimeMs);
					// float max_weight_seen = -1.f;
					for (const auto* vertex : nearbyVertices) {
						if (vertex == nullptr) continue;
						float cosTheta = Dot(shadingData.sNormal, vertex->wi);
						if (cosTheta < 0.f) continue;
						
						float sqrtLum = std::sqrt(vertex->Li.Lum());
						float weight = std::max(sqrtLum, 0.f);
						if (weight < EPSILON) continue;

						// BSDF inversion to get u, v, and u_lobe
						float u, v, u_lobe;
						// BSDF Inversion will be timed inside this scope
						{
							Timer bsdfInvertTimer(stats.bsdfInvertTimeMs);
							shadingData.bsdf->invert(shadingData, vertex->wi, u, v, u_lobe);
						}
						// Timing completed
						if (u < 0.f || u > 1.f || v < 0.f || v > 1.f || std::isnan(u) || std::isnan(v)) continue;

						// Insert into the QTree
						qTree.insert(u, v, weight);
						totalInsertedWeight += weight;
						acceptedVertices++;

						// if (weight > max_weight_seen) {
							// max_weight_seen = weight;
							// best_lobe_selection = u_lobe;
						// }
					}
				}
				// Timing completed
				usePathGuiding = (acceptedVertices >= MIN_ACCEPTED_INSERTIONS) && (totalInsertedWeight > EPSILON);
				if (usePathGuiding) stats.guidedPathBounceCount++;
				if (!usePathGuiding) qTree.clear();
			}

			// Lambda Function - Returns PDF of the BSDF
			auto forwardPdf = [&](const Vec4& wi) -> float {
				// If not path guiding, return the BSDFs PDF
				float basePdf = shadingData.bsdf->PDF(shadingData, wi);
				if (!usePathGuiding) return basePdf;

				// Sample the new pdf via BSDF inversion
				float u, v, u_lobe;
				shadingData.bsdf->invert(shadingData, wi, u, v, u_lobe);

				// If not valid u, return BSDF PDF
				if (u < 0.f || u > 1.f || v < 0.f || v > 1.f || std::isnan(u) || std::isnan(v)) return basePdf * BSDF_FRACTION;

				// If inversion OK, return MIS combined PDF for Path Guiding
				float qTreePdf = qTree.pdf(u, v);
				return basePdf * (BSDF_FRACTION + QTREE_FRACTION * qTreePdf);
			};

			// Save the direct lighting (NEE) to the record
			record.directLighting = computeDirect(shadingData, sampler, forwardPdf);

			// Apply Russian Roulette
			// Russian Roulette should kick in normally between at depth 3 to 5
			float rrpRecord = 1.f;
			if (depth > 3) {
				// Using a fixed RRP because we don't have access to pathThroughput anymore...
				float rrp = 0.7f;
				if (sampler->next() < rrp) { rrpRecord = rrp; }
				else { qTree.clear(); records.push_back(record); return; }
			}

			// Calculate Indirect Lighting - Sampling Proportional to BSDF (Materials)
			float pdfCombined = 0.f;
			Colour fBsdf;
			Vec4 wi;

			if (usePathGuiding) {
				// Pick a stragety to either guide the sampling or standard BSDF sampling
				float strategy = sampler->next();
				if (strategy < QTREE_FRACTION) {
					// Do Guided Sampling
					float r1 = sampler->next();
					float r2 = sampler->next();
					float u_out = 0.f, v_out = 0.f, treePdfIgnored = 0.f;
					qTree.sample(r1, r2, u_out, v_out, treePdfIgnored);

					// Then do BSDF sampling with the new numbers obtained
					GuidedPathSampler dummySampler;
					dummySampler.set(u_out, v_out, best_lobe_selection);
					float pdfIgnored = 0.f;
					wi = shadingData.bsdf->sample(shadingData, &dummySampler, fBsdf, pdfIgnored);
				} else {
					// Do Standart Sampling
					float pdfIgnored = 0.f;
					wi = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfIgnored);
				}
				// Defensive Sampling - MIS Mixture PDF
				pdfCombined = forwardPdf(wi);
			} else {
				wi = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfCombined);
			}

			#if DEBUG_GUIDED_PATH
			// If this is the FIRST bounce (depth == 0) and we are guiding
			if (depth == 0 && usePathGuiding) {
				// Invert debug sample
				float uDebug = 0.f, vDebug = 0.f, lobeDebug = 0.f;
				shadingData.bsdf->invert(shadingData, wi, uDebug, vDebug, lobeDebug);
				float qDebug = (uDebug >= 0.f) ? qTree.pdf(uDebug, vDebug) : 0.f;
				
				// Scale for visibility
				float debugColor = std::min(qDebug * 0.1f, 1.0f); 

				// Modify the record to output QTree debug colours
				record.emission = Colour(debugColor, debugColor, debugColor);
				record.misEmission = record.emission;
				record.directLighting = Colour(0.f, 0.f, 0.f);
				record.bsdfWeight = Colour(0.f, 0.f, 0.f);
				record.storeRecord = false;
				records.push_back(record);
				qTree.clear();

				// We just want to see the QTree value here.
				return;
			}
			#endif

			// Clear the QTree before the next bounce
			qTree.clear();

			// Do not contribute if we have pdf anomalies
			if (pdfCombined <= 1e-8f || std::isnan(pdfCombined) || std::isinf(pdfCombined)) {
				records.push_back(record);
				return;
			}
			// Check cosine term
			float cosTheta = fabs(Dot(wi, shadingData.sNormal));
			if (cosTheta <= 0.f) { records.push_back(record); return; }

			// Store the necessary records in the record structure
			record.wi = wi;
			record.bsdfWeight = (fBsdf * cosTheta) / (pdfCombined * rrpRecord);
			record.storeRecord = !isSpecular;
			records.push_back(record);

			// Define indirect ray for the next bounce, and recurse through the function
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Ray indirectRay(shadingData.x + shadingData.gNormal * (EPSILON * sign), wi);
			generatePathRecursive(indirectRay, depth + 1, sampler, records, cache, qTree, isGuidingPhase, nearbyVertices, maxHeap, stats, pdfCombined, isSpecular);
			return;
		}
		// --- Environment Map ---
		Colour backgroundColour = scene->background->evaluate(r.dir);
		record.emission = backgroundColour;
		if (depth == 0 || previousSurfaceSpecular) { record.misEmission = backgroundColour; }
		else if (backgroundColour.Lum() > 1e-8f) {
			// Evaluate MIS for Environment Map
			// Environment Map PDF and PMF
			float pmfLight = 1.f / scene->lights.size();
			float pdfLight = scene->background->PDF(shadingData, r.dir);

			// Calculate pA of Light and BSDF for MIS
			float pALight = pmfLight * pdfLight;
			float pABsdf = previousBsdfPdf;

			// Calculate Weight for MIS
			float wind = weightPowerHeuristics(pABsdf, pALight);
			record.misEmission = backgroundColour * wind;
		}
		records.push_back(record);
		return;
	}
	// --- Path Guiding Algorithm Work End ---

	// --- PSS Debug ---
	void viewPrimarySampleSpace(Ray& r, PointBVH* cache, int k) {
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX) {
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return visualisePSSMap(shadingData, cache, k);
		}
	}

	void visualisePSSMap(ShadingData& shadingData, PointBVH* cache, int k) {
		// Create debug structures
		std::cout << "\[DEBUG] Generating Primary Sample Space Visualisation..." << std::endl;
		std::priority_queue<NearestPathVertex> debugMaxHeap;
		std::vector<const PathVertex*> debugNearbyVertices;
		QTree debugQTree;

		// Perform kNN Search
		float debugRadiusSq = FLT_MAX;
		cache->kNNSearch(shadingData.x, debugRadiusSq, debugMaxHeap, k);
		while (!debugMaxHeap.empty()) {
			const PathVertex* poppedDebugVertex = debugMaxHeap.top().vertex;
			debugNearbyVertices.push_back(poppedDebugVertex);
			debugMaxHeap.pop();
		}

		// Store uv's for plotting
		std::vector<std::pair<float, float>> plottedPoints;

		// BSDF Inversion and QTree Insertions
		for (const auto* vertex : debugNearbyVertices) {
			if (vertex == nullptr) continue;
			if (Dot(shadingData.sNormal, vertex->wi) < 0.f) continue;

			// BSDF Invert
			float uDebug, vDebug, selectDebug;
			shadingData.bsdf->invert(shadingData, vertex->wi, uDebug, vDebug, selectDebug);
			if (uDebug < 0.f || uDebug > 1.f || vDebug < 0.f || vDebug > 1.f || std::isnan(uDebug) || std::isnan(vDebug)) continue;

			// QTree Insert
			float sqrtLum = std::sqrt(vertex->Li.Lum());
			float weight = std::max(sqrtLum, 0.f);
			if (weight > 0.f) {
				debugQTree.insert(uDebug, vDebug, weight);
				plottedPoints.push_back({ uDebug, vDebug });
			}
		}
		// Generate Image Buffer to Plot the Heat Map
		// 512x512, RGB Channels, fill with zeroes first
		const int SIZE = 512;
		std::vector<unsigned char> buffer(SIZE * SIZE * 3, 0);
		std::vector<unsigned char> bufferDefensive(SIZE * SIZE * 3, 0);
		
		// Draw the QTree as a grayscale heatmap
		// Calculate maxPDF factor
		float maxPdf = EPSILON;
		for (int y = 0; y < SIZE; y++) {
			for (int x = 0; x < SIZE; x++) {
				float u = (x + 0.5f) / SIZE;
				float v = (y + 0.5f) / SIZE;
				float pdf = debugQTree.pdf(u, v);
				maxPdf = std::max(pdf, maxPdf);
			}
		}

		// Heat-Map Lambda Function as Helper Function
		auto getHeatmapRGB = [](float t, unsigned char& r, unsigned char& g, unsigned char& b) {
			float val = std::max(0.f, std::min(t, 1.f));
			r = (unsigned char)(std::max(0.f, val * 255.f));
			g = (unsigned char)(std::max(0.f, val * 255.f));
			b = (unsigned char)(std::max(0.f, val * 255.f));
		};

		// Visualise
		for (int y = 0; y < SIZE; y++) {
			for (int x = 0; x < SIZE; x++) {
				float u = (x + 0.5f) / SIZE;
				float v = (y + 0.5f) / SIZE;

				// Create Guided Sampler for dummy bsdf
				GuidedPathSampler dummySampler;
				dummySampler.set(u, v, 0.5f);

				// Sample BSDF for Defensive Sampling Test
				Colour dummyBsdf;
				float dummyPdf;
				shadingData.bsdf->sample(shadingData, &dummySampler, dummyBsdf, dummyPdf);

				// Calculate PDF and Defensive PDF
				float qTreePdfPSS = debugQTree.pdf(u, v);
				float pdfDefensive = (BSDF_FRACTION * dummyPdf) + (QTREE_FRACTION * (dummyPdf * qTreePdfPSS));

				// Normalize PDFs and Apply Gamma Correction
				// BSDF is solid angle, QTree is PSS... need to multiply with Jacobian
				float normPdf = std::pow(qTreePdfPSS / maxPdf, 1.f / 2.2f);
				float normPdfDefensive = std::pow(pdfDefensive / (maxPdf * dummyPdf + EPSILON), 1.f / 2.2f);
				
				// Save the pixel into the buffer
				int pixelIdx = ((SIZE - 1 - y) * SIZE + x) * 3;
				getHeatmapRGB(normPdf, buffer[pixelIdx], buffer[pixelIdx + 1], buffer[pixelIdx + 2]);
				getHeatmapRGB(normPdfDefensive, bufferDefensive[pixelIdx], bufferDefensive[pixelIdx + 1], bufferDefensive[pixelIdx + 2]);
			}
		}

		// Draw the vertices
		for (auto& p : plottedPoints) {
			// Retrieve points from the pair
			int px = std::min((int)(p.first * SIZE), SIZE - 1);
			int py = std::min((int)(p.second * SIZE), SIZE - 1);

			// Draw a 3x3 dot
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int nx = px + dx, ny = py + dy;
					if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE) {
						int pixelIdx = ((SIZE - 1 - ny) * SIZE + nx) * 3;
						buffer[pixelIdx] = 255; buffer[pixelIdx + 1] = 0; buffer[pixelIdx + 2] = 0;
						bufferDefensive[pixelIdx] = 0; bufferDefensive[pixelIdx + 1] = 0; bufferDefensive[pixelIdx + 2] = 255;
					}
				}
			}
		}
		// 8. Save the Image
		stbi_write_png("pss_qtree_debug_map.png", SIZE, SIZE, 3, buffer.data(), SIZE * 3);
		stbi_write_png("pss_defensive_debug_map.png", SIZE, SIZE, 3, bufferDefensive.data(), SIZE * 3);
		std::cout << "[DEBUG] Saved Maps! Total plotted points: " << plottedPoints.size() << std::endl;
	}
	// --- PSS Debug End ---

	Colour pathTraceRecursive(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, float previousBsdfPdf = 0.f, bool previousSurfaceSpecular = false) {
		// Trace ray
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) {
				Colour emittedColour = shadingData.bsdf->emit(shadingData, shadingData.wo);
				if (depth == 0 || previousSurfaceSpecular) return pathThroughput * emittedColour;
				// Evaluate MIS for Area Light
				// Area Light PDF and PMF
				float pmfLight = 1.f / scene->lights.size();
				float pdfLight = 1.f / scene->triangles[intersection.ID].area;

				float cosThetaPrime = std::max(Dot(-r.dir, scene->triangles[intersection.ID].gNormal()), 0.f);
				if (cosThetaPrime <= 0.f) return Colour(0.f, 0.f, 0.f);
				float distanceSquare = SQ(intersection.t);
				if (distanceSquare < EPSILON) return Colour(0.f, 0.f, 0.f);

				// Calculate pA of Light and BSDF for MIS
				float pALight = pdfLight * pmfLight;
				float pABsdf = previousBsdfPdf * cosThetaPrime / distanceSquare;

				// Calculate Weight for MIS
				float wind = weightPowerHeuristics(pABsdf, pALight);
				return pathThroughput * emittedColour * wind;
			}

			// Calculate Direct Lighting
			Colour direct = pathThroughput * computeDirect(shadingData, sampler);

			// Terminate when the ray depth exceeds 8 bounces, to avoid infinite recursion
			if (depth >= MAX_DEPTH) return direct;

			// Apply Russian Roulette Starting at the ray depth 4
			// Russian Roulette should kick in normally between at depth 3 to 5
			if (depth > 3) {
				float rrp = std::min(std::max(EPSILON, pathThroughput.Lum()), 1.f);
				if (sampler->next() < rrp) pathThroughput = pathThroughput / rrp;
				else return direct;
			}
			
			// Calculate Indirect Lighting - Sampling Proportional to BSDF (Materials)
			float pdfBsdf = 0.f;
			Colour indirect;
			Vec4 wi = shadingData.bsdf->sample(shadingData, sampler, indirect, pdfBsdf);
			if (pdfBsdf <= 1e-8f) return direct;
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Ray indirectRay(shadingData.x + shadingData.gNormal * (EPSILON * sign), wi);

			// Update throughput
			// Taking absolute value of cosTheta to fix GlassBSDF rendering
			float cosTheta = fabs(Dot(wi, shadingData.sNormal));
			if (cosTheta <= 0.f) return direct;
			pathThroughput = (pathThroughput * indirect * cosTheta) / pdfBsdf;

			// Eliminate non-valid bounces
			if (!pathThroughput.isValid() || pathThroughput.Lum() < 0.f) return direct;
			if (pathThroughput.r < 0.f || pathThroughput.g < 0.f || pathThroughput.b < 0.f) return direct;
			
			// Recurse until path terminated
			bool isPreviousSurfaceSpecular = shadingData.bsdf->isPureSpecular();
			return direct + pathTraceRecursive(indirectRay, pathThroughput, depth + 1, sampler, pdfBsdf, isPreviousSurfaceSpecular);
		}
		Colour backgroundColour = scene->background->evaluate(r.dir);
		if (depth == 0 || previousSurfaceSpecular) return pathThroughput * backgroundColour;
		if (backgroundColour.Lum() < 1e-8f) return Colour(0.f, 0.f, 0.f);
		// Evaluate MIS for Environment Map
		// Infinite Light PDF and PMF
		float pmfLight = 1.f / scene->lights.size();
		float pdfLight = scene->background->PDF(shadingData, r.dir);

		// Calculate pA of Light and BSDF for MIS
		float pALight = pmfLight * pdfLight;
		float pABsdf = previousBsdfPdf;

		// Calculate Weight for MIS
		float wind = weightPowerHeuristics(pABsdf, pALight);
		return pathThroughput * backgroundColour * wind;
	}

	Colour pathTrace(Ray& r, Sampler* sampler) {
		Colour pathThroughput(1.f, 1.f, 1.f);
		return pathTraceRecursive(r, pathThroughput, 0, sampler);
	}

	Colour albedo(Ray& r) {
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) return shadingData.bsdf->emit(shadingData, shadingData.wo);
			return shadingData.bsdf->evaluate(shadingData, Vec4(0.f, 1.f, 0.f));
		}
		return scene->background->evaluate(r.dir);
	}

	Colour viewNormals(Ray& r) {
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX) {
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.f, 0.f, 0.f);
	}

	Colour viewBarycentrics(Ray& r) {
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX) {
			return Colour(fabsf(intersection.alpha), fabsf(intersection.beta), fabsf(intersection.gamma));
		}
		return Colour(0.f, 0.f, 0.f);
	}

	// Tile-based Rendering
	void render() {
		// Increment SPP and define Atomic ID Counter to battle race conditions
		film->incrementSPP();
		std::atomic<int> id = 0;

		// Get total tile count
		unsigned int tile_size = 32;
		unsigned int tiles_x = (film->width + tile_size - 1u) / tile_size;
		unsigned int tiles_y = (film->height + tile_size - 1u) / tile_size;
		unsigned int total_tile_count = tiles_x * tiles_y;

		// Are we in the learning phase or guiding phase?
		std::atomic<bool> isGuidingPhase = (getSPP() >= learningThreshold);

		// Threads
		for (unsigned int i = 0; i < numProcs; ++i) {
			threads[i] = new std::thread(
				// Capture i by value, not reference, or else samplers[i] will go out of bounds!
				[&, i]() {
					// Lambda function to render tiles
					unsigned int tile_id = 0;
					std::vector<PathVertex>& currentThreadPathVertexRecords = perThreadPathVertexRecords[i];
					QTree& currentThreadQTree = perThreadQTrees[i];
					ProfilerStats& currentThreadStats = perThreadStats[i];

					while ((tile_id = id.fetch_add(1)) < total_tile_count) {
						// Initialize Screen Tile
						ScreenTile tile;
						tile.tile_size = tile_size;
						tile.x = (tile_id % tiles_x) * tile_size;
						tile.y = (tile_id / tiles_x) * tile_size;

						// Render Tiles
						for (unsigned int y = tile.start_tile_y(); y <= tile.end_tile_y(film); ++y) {
							for (unsigned int x = tile.start_tile_x(); x <= tile.end_tile_x(film); ++x) {
								// Sample a point in the pixel
								float px = x + samplers[i].next();  // + 0.5f
								float py = y + samplers[i].next();  // + 0.5f
								Ray ray = scene->camera.generateRay(px, py);

								// View Barycentrics / Shading Normals / Albedo / Direct Lighting / Path Trace / Path Guiding
								// Colour col = viewBarycentrics(ray);
								// Colour col = viewNormals(ray);
								// Colour col = albedo(ray);
								// Colour col = direct(ray, &samplers[i]);

								#if PHOTON_MAPPING
								// Photton Mapping [Jensen 1995]
								Colour col = traceCameraRay(ray, &samplers[i]);
								#elif GUIDED_PATH
								// Our novel Path Guiding in PSS
								Colour col = guidedPath(ray, &samplers[i], currentThreadPathVertexRecords, cacheBVH, currentThreadQTree, isGuidingPhase, currentThreadStats, enableNEE);
								if (isGuidingPhase && getSPP() == learningThreshold && x == film->width / 2 && y == film->height / 2) {
									viewPrimarySampleSpace(ray, cacheBVH, MAX_NEARBY_VERTICES);
								}
								#else
								// Unidirectional Path Trace
								Colour col = pathTrace(ray, &samplers[i]);
								#endif

								// Check for NaN/Inf values and then Splat, Tonemap, and Draw to Pixel
								if (!col.isValid()) continue;
								film->splat(px, py, col);
								unsigned char r, g, b;
								film->tonemap(x, y, r, g, b);
								canvas->draw(x, y, r, g, b);
							}
						}
					}
				}
			);
		}

		// Join the threads, and delete to save memory after joined
		for (unsigned int i = 0; i < numProcs; ++i) {
			threads[i]->join();
			delete threads[i];
		}

		#if GUIDED_PATH
		// --- Guiding Phase ---
		if (isGuidingPhase) {
			// Generate the profiling record	
			double totalBVHTime = 0.0;
			double totalQTreeTime = 0.0;
			double totalBSDFInvertTime = 0.0;
			long long totalGuidedPathBounces = 0;

			// Retrieve all blocks obtained from the threads
			for (int i = 0; i < numProcs; i++) {
				totalBVHTime += perThreadStats[i].bvhSearchTimeMs;
				totalQTreeTime += perThreadStats[i].qTreeBuildTimeMs;
				totalBSDFInvertTime += perThreadStats[i].bsdfInvertTimeMs;
				totalGuidedPathBounces += perThreadStats[i].guidedPathBounceCount;
			}

			// Divide by thread count to average everything
			double avgBVHTime = totalBVHTime / numProcs;
			double avgQTreeTime = totalQTreeTime / numProcs;
			double avgBSDFInvertTime = totalBSDFInvertTime / numProcs;

			// Print the profiling report
			std::cout << "\n=========================================\n";
			std::cout << "      PATH GUIDING PROFILING REPORT      \n";
			std::cout << "=========================================\n";
			std::cout << "Total Guided Path Bounces : " << totalGuidedPathBounces << std::endl;
			std::cout << "Average BVH Search Time   : " << avgBVHTime << " ms / frame" << std::endl;
			std::cout << "Average QTree Build Time  : " << avgQTreeTime << " ms / frame" << std::endl;
			std::cout << "Average BSDF Invert Time  : " << avgBSDFInvertTime << " ms / frame" << std::endl;

			if (totalGuidedPathBounces > 0) {
				double bvhSearchPerBounce = (totalBVHTime * 1000.0) / totalGuidedPathBounces;
				double qTreeBuildPerBounce = (totalQTreeTime * 1000.0) / totalGuidedPathBounces;
				double bsdfInvertPerBounce = (totalBSDFInvertTime * 1000.0) / totalGuidedPathBounces;
				std::cout << "-----------------------------------------\n";
				std::cout << "Time Per Bounce Stats:\n";
				std::cout << "  BVH Search  : " << bvhSearchPerBounce << " microseconds" << std::endl;
				std::cout << "  QTree Build : " << qTreeBuildPerBounce << " microseconds" << std::endl;
				std::cout << "  BSDF Invert : " << bsdfInvertPerBounce << " microseconds" << std::endl;
			}
			std::cout << "=========================================\n\n";

			// Reset the records
			for (int i = 0; i < numProcs; i++) {
				perThreadStats[i] = ProfilerStats();
			}
		}

		// --- Learning Phase ---
		if (!isGuidingPhase) {
			// Get the total size and allocate space on the global cache list
			size_t total_size = 0;
			for (auto& pathVertexRecords : perThreadPathVertexRecords) {
				total_size += pathVertexRecords.size();
			}
			globalCacheList.reserve(globalCacheList.size() + total_size);

			// Carry the data obtained from tiled rendering to the global cache list
			for (auto& recordsList : perThreadPathVertexRecords) {
				// Insert the records in the global cache list
				globalCacheList.insert(
					globalCacheList.end(),
					std::make_move_iterator(recordsList.begin()),
					std::make_move_iterator(recordsList.end())
				);

				// Free up the memory for the current thread's records
				recordsList.clear();
				recordsList.shrink_to_fit();
			}
		}

		// Build the BVH exactly once, after collecting cache estimates in the learning phase
		if ((getSPP() == learningThreshold - 1)) {
			std::cout << "Learning phase finished. Building BVH for cached path vertices..." << std::endl;
			std::cout << "Total cached path vertices: " << globalCacheList.size() << std::endl;
			cacheBVH = new PointBVH();
			cacheBVH->build(std::move(globalCacheList));

			// Clear Global Cache Estimates List
			globalCacheList.clear();
			globalCacheList.shrink_to_fit();
			std::cout << "BVH built successfully. Entering guiding phase..." << std::endl;
		}
		#endif
	}

	int getSPP() { return film->SPP; }
	void saveHDR(std::string filename) { film->save(filename); }
	void savePNG(std::string filename) { stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3); }
};