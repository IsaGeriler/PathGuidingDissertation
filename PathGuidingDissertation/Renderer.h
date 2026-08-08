#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <ratio>
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
#define GUIDED_PATH true
#define DEBUG_GUIDED_PATH false

// Enable NEE or not for Incoming Radiance (Li)
constexpr bool enableNEE = true;

// Defensive Sampling & Mixture (50-50)
static const int MAX_NEARBY_VERTICES = 800;
static const int MIN_ACCEPTED_INSERTIONS = 800;
static const float BSDF_FRACTION = 0.5f;
static const float QTREE_FRACTION = 0.5f;
static const float QTREE_MIX = 0.9f;
static const float UNIFORM_MIX = 0.1f;
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
	Colour directLighting;	   // Next Event Estimation
	Colour emission;		   // Unweighted emitted colour
	Colour misEmission;		   // MIS weight applied to emitted colour

	bool storeRecord = false;  // Do not store if previous surface is pure specular
};

struct PointBVHNodeStats {
	int nodeCount = 0, leafNodeCount = 0;
	int minLeafDepth = INT_MAX, maxLeafDepth = -INT_MAX;
	long long sumLeafDepth = 0;
	size_t memory_in_bytes = 0;
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
class PointBVHNode {
private:
	// Attributes
	AABB bounds;
	PointBVHNode* r;
	PointBVHNode* l;
	int offset = 0, used = 0;
	const int MAX_CHILDNODE_RECORDS = 8;
	double buildTime = 0.0;
	std::vector<PathVertex> pathVertexRecords;

	// Private Methods
	bool isLeaf() const { return l == nullptr && r == nullptr; }
	
	void subdivide(std::vector<PathVertex>& pathVertices) {
		// Return if the used node count exceeds max child node count
		if (used <= MAX_CHILDNODE_RECORDS) return;

		// Get the extend vector from the AABB bounds
		Vec4 extendVector = bounds.max - bounds.min;

		// Find the split axis
		int ax = 0;
		if (extendVector.y > extendVector.x) ax = 1;
		if (extendVector.z > extendVector[ax]) ax = 2;

		// Get the first, last, and nth element indexes
		auto first = pathVertices.begin() + offset;
		auto nth = pathVertices.begin() + offset + used / 2;
		auto last = pathVertices.begin() + offset + used;

		// From those indexes, sort elements via a comparator
		std::nth_element(first, nth, last, 
			// Lambda function as a comparator, capture the split axis by value
			// Unlike scene triangle BVH we do not use centroids
			// Instead, we compare the values of corresponding axis value of position vectors
			[ax](PathVertex& vertex1, PathVertex& vertex2) { return vertex1.position[ax] < vertex2.position[ax]; }
		);

		// Get the middle index
		int middle = offset + used / 2;

		// Create left child and assign the values to it's attributes
		l = new PointBVHNode();
		l->offset = offset;
		l->used = used / 2;

		// Create right child and assign the values to it's attributes
		r = new PointBVHNode();
		r->offset = middle;
		r->used = used - used / 2;

		// Reset the node count for the parent
		used = 0;

		// Update AABB bounds
		l->updateBounds(pathVertices);
		r->updateBounds(pathVertices);

		// Subdivide the tree
		l->subdivide(pathVertices);
		r->subdivide(pathVertices);
	}

	void updateBounds(std::vector<PathVertex>& pathVertices) {
		// Reset the bounds
		bounds.reset();
		// Extend the bounds according to the position vector
		for (int i = offset; i < offset + used; i++) {
			bounds.extend(pathVertices[i].position);
		}
	}

	int validateNode(std::vector<PathVertex>& pathVertices, int depth) const {
		// Check for leaf node case
		if (isLeaf()) {
			// Check if we have more used records than the max amount
			assert(used <= MAX_CHILDNODE_RECORDS && "Leaf carries records over capacity");
			// Check if the record positions are inside the bounding box or not
			for (int i = offset; i < offset + used; i++) {
				assert(bounds.containsPoint(pathVertices[i].position, EPSILON) && "Record outside of its leaf's bounding box.");
			}
			return used;
		}
		// Handle not leaf node case
		assert(used == 0 && "Parent node is abruptly carrying records");
		assert(l != nullptr && r != nullptr && "Parent node is missing a child node");
		assert(bounds.containsAABB(l->bounds, EPSILON) && "Left child's bounding box is not in parent bounding box.");
		assert(bounds.containsAABB(r->bounds, EPSILON) && "Right child's bounding box is not in parent bounding box.");
		return l->validateNode(pathVertices, depth + 1) + r->validateNode(pathVertices, depth + 1);
	}

	void statsNode(PointBVHNodeStats& bvhStats, int depth) {
		bvhStats.nodeCount++;
		bvhStats.memory_in_bytes += sizeof(PointBVHNode);
		// Check if leaf node
		if (isLeaf()) {
			bvhStats.leafNodeCount++;
			bvhStats.minLeafDepth = std::min(bvhStats.minLeafDepth, depth);
			bvhStats.maxLeafDepth = std::max(bvhStats.maxLeafDepth, depth);
			bvhStats.sumLeafDepth += depth;
			return;
		}
		// Recurse through the child nodes
		if (l != nullptr) l->statsNode(bvhStats, depth + 1);
		if (r != nullptr) r->statsNode(bvhStats, depth + 1);
	}

	void searchNode(Vec4 hitPosition, float radiusSq, std::vector<PathVertex>& pathVertices, std::vector<const PathVertex*>& nearbyVertices, int maxVertices) {
		// Return if the nearby vertices exceed the max vertices count
		if (nearbyVertices.size() >= maxVertices) return;
		
		// Find the closest point in the AABB
		Vec4 closestPoint;
		closestPoint.x = std::max(std::min(hitPosition.x, this->bounds.max.x), this->bounds.min.x);
		closestPoint.y = std::max(std::min(hitPosition.y, this->bounds.max.y), this->bounds.min.y);
		closestPoint.z = std::max(std::min(hitPosition.z, this->bounds.max.z), this->bounds.min.z);

		// Calculate distance square to prevent sqrt and omit the ones less than radiusSq
		float distanceSq = (SQ(closestPoint.x - hitPosition.x) + SQ(closestPoint.y - hitPosition.y) + SQ(closestPoint.z - hitPosition.z));
		if (distanceSq > radiusSq) return;

		if (isLeaf()) {
			for (int i = offset; i < offset + used; i++) {
				if (nearbyVertices.size() >= maxVertices) return;
				const PathVertex& vertex = pathVertices[i];
				float distanceSq2 = (SQ(vertex.position.x - hitPosition.x) + SQ(vertex.position.y - hitPosition.y) + SQ(vertex.position.z - hitPosition.z));
				if (distanceSq2 <= radiusSq) nearbyVertices.push_back(&vertex);
			}
			return;
		}
		else {
			if (l != nullptr) l->searchNode(hitPosition, radiusSq, pathVertices, nearbyVertices, maxVertices);
			if (r != nullptr) r->searchNode(hitPosition, radiusSq, pathVertices, nearbyVertices, maxVertices);
		}
	}
public:
	// Constructor
	PointBVHNode() {
		r = nullptr;
		l = nullptr;
	}

	// Destructor
	~PointBVHNode() {
		if (r != nullptr) delete r;
		if (l != nullptr) delete l;
	}

	// Public Methods
	void buildPointBVHNode(std::vector<PathVertex>&& inputPathVertices) {
		// Handle degenerate case where the passed vector is empty
		if (inputPathVertices.empty()) return;
		pathVertexRecords = std::move(inputPathVertices);

		// Time the build time using chrono
		auto start = std::chrono::high_resolution_clock::now();

		// Set these values for the root node
		offset = 0; used = (int)pathVertexRecords.size();

		// Update and subdivide the root node
		updateBounds(pathVertexRecords);
		subdivide(pathVertexRecords);

		// End the timing
		auto end = std::chrono::high_resolution_clock::now();

		// Save the build time, and call validate and stats
		buildTime = std::chrono::duration<double, std::milli>(end - start).count();
		validate();
		stats();
	}

	void validate() {
		// Start from the root node, depth at 0
		int counted = validateNode(pathVertexRecords, 0);
		assert(counted == (int)(pathVertexRecords.size()) && "Records lost or duplicated during the build phase.");
	}

	PointBVHNodeStats stats() {
		PointBVHNodeStats bvhStats;
		bvhStats.memory_in_bytes = pathVertexRecords.size() * sizeof(PathVertex);
		statsNode(bvhStats, 0);
		bvhStats.buildTimeMs = buildTime;
		std::cout << "PointBVHNode["
			<< "\n  -- path vertex records: " << pathVertexRecords.size()
			<< "\n  -- nodes: " << bvhStats.nodeCount
			<< "\n  -- leaf nodes: " << bvhStats.leafNodeCount
			<< "\n  -- depth: " << bvhStats.minLeafDepth << "-" << bvhStats.maxLeafDepth
			<< " (mean " << (double)bvhStats.sumLeafDepth / (double)bvhStats.leafNodeCount << ")"
			<< "\n  -- size: " << bvhStats.memory_in_bytes / SQ(1024.0) << "MB"
			<< "\n  -- build time: " << bvhStats.buildTimeMs << "ms\n]\n";
		return bvhStats;
	}

	// --- New Method ---
	void search(Vec4 hitPosition, float radiusSq, std::vector<const PathVertex*>& nearbyVertices, int maxVertices) {
		// Search from the root
		searchNode(hitPosition, radiusSq, pathVertexRecords, nearbyVertices, maxVertices);
	}
};
// --- Spatial-Tree Component to Store PathVertex Caches End ---

class RayTracer {
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom* samplers;
	std::thread** threads;
	unsigned int numProcs;

	// Cached vertices will be stored in a BVH structure
	PointBVHNode* cacheBVH;
	std::vector<PathVertex> globalCacheList;
	// Ground Truth: 8192/16384					(should really be an absurd number to eliminate variance)
	// Testing SPPs: 128/256/512/1024/2048/4096 (render, and compare error metrics with the groung truth)
	int maxSPP = 128;
	int learningThreshold = maxSPP / 8;

	// Path Vertex vector to then cache saved items over at a Spatial Accelleration Structure
	std::vector<std::vector<PathVertex>> perThreadPathVertexRecords;
	QTree* perThreadQTrees;

	// Stats for Profiling
	std::vector<ProfilerStats> perThreadStats;

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
	}

	void clear() {
		film->clear();
	}

	Colour computeDirect(const ShadingData& shadingData, Sampler* sampler) {
		// Is surface is specular we cannot computing direct lighting
		if (shadingData.bsdf->isPureSpecular() == true) return Colour(0.f, 0.f, 0.f);
		// Compute direct lighting here
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
				float pdfBsdf = shadingData.bsdf->PDF(shadingData, wi);

				// Calculate Weight for MIS
				float pALight = pdfLight * pmfLight;
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

				// Evaluate pALight and pABsdf for MIS
				// cosTheta is pretty much geometry term for Environment Mapping so pdfBsdf turns into pABsdf
				float pALight = pdfLight * pmfLight;
				float pABsdf = shadingData.bsdf->PDF(shadingData, wi);
				
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

	// --- Path Guiding Algorithm Work Start ---
	Colour computeDirectGuidedPath(const ShadingData& shadingData, Sampler* sampler, const std::function<float(const Vec4&)>& pdfFunction) {
		// Is surface is specular we cannot computing direct lighting
		if (shadingData.bsdf->isPureSpecular() == true) return Colour(0.f, 0.f, 0.f);
		// Compute direct lighting here
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

				// It was shadingData.bsdf->PDF() at first, but when guiding the paths
				// this may cause pdf mismatches. So this is an attempt to fix it
				float pdfBsdf = pdfFunction(wi);

				// Calculate Weight for MIS
				float pALight = pdfLight * pmfLight;
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

				// Evaluate pALight and pABsdf for MIS
				// cosTheta is pretty much geometry term for Environment Mapping so pdfBsdf turns into pABsdf
				float pALight = pdfLight * pmfLight;

				// It was shadingData.bsdf->PDF() at first, but when guiding the paths
				// this may cause pdf mismatches. So this is an attempt to fix it
				float pABsdf = pdfFunction(wi);

				// Calculate Weight for MIS
				float wd = weightPowerHeuristics(pALight, pABsdf);

				// Multiply terms, divide by pALight, and return
				return (emittedColour * bsdf * cosTheta * wd) / pALight;
			}
			return Colour(0.f, 0.f, 0.f);
		}
	}

	Colour guidedPath(Ray& r, Sampler* sampler, std::vector<PathVertex>& pathVertices, PointBVHNode* sTree, QTree& dTree, bool isGuidingPhase, ProfilerStats& stats, bool enableNEE) {
		// --- 1. Forward Pass Phase ---
		std::vector<ForwardPassRecord> records;
		std::vector<const PathVertex*> nearbyVertices;
		records.reserve(10);                          // Max depth: 8, + 2 buffer space
		nearbyVertices.reserve(MAX_NEARBY_VERTICES);  // Pre-allocate max number of wanted vertices

		// Generate the path vertices in the forward pass (only populates the vector)
		generatePathRecursive(r, 0, sampler, records, sTree, dTree, isGuidingPhase, nearbyVertices, stats);

		// --- 2. Backpropagation Phase ---
		// Store Each Path Vertex to the vector via Backpropagation
		Colour incomingRadiance(0.f, 0.f, 0.f);
		for (int i = (int)(records.size() - 1); i >= 0; i--) {
			if (!isGuidingPhase) {
				// Learning Phase - Data Collection
				if (records[i].storeRecord) {
					PathVertex pathVertex;
					pathVertex.position = records[i].position;
					pathVertex.normal = records[i].normal;
					pathVertex.wi = records[i].wi;
					pathVertex.Li = incomingRadiance;
					if (pathVertex.Li.isValid() && pathVertex.Li.Lum() > 0) pathVertices.push_back(pathVertex);
				}
			}
			// Update the incoming radiance so that Li-1 can use this previous Li
			if (enableNEE) incomingRadiance = records[i].misEmission + records[i].directLighting + (records[i].bsdfWeight * incomingRadiance);
			else incomingRadiance = records[i].emission + (records[i].bsdfWeight * incomingRadiance);
		}
		if (!incomingRadiance.isValid()) return Colour(0.f, 0.f, 0.f);
		return incomingRadiance;
	}

	// --- NEW METHOD ---
	// -> When sampling, replace BSDF with the new method, i.e.
	//    -> Search for nearby vertices from 1.
	//    -> Project wi into PSS
	//    -> Invert BSDF sampling
	//    -> Sample PSS
	void generatePathRecursive(Ray& r, int depth, Sampler* sampler, std::vector<ForwardPassRecord>& records, PointBVHNode* cache, QTree& qTree, bool isGuidingPhase, std::vector<const PathVertex*>& nearbyVertices, ProfilerStats& stats, float previousBsdfPdf = 0.f, bool previousSurfaceSpecular = false) {
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

						// Handle degenerate pALight and pABsdf
						if (pALight > 0.f && pABsdf > 0.f) {
							// Calculate Weight for MIS
							float wind = weightPowerHeuristics(pABsdf, pALight);
							record.misEmission = emittedColour * wind;
						} else {
							record.misEmission = emittedColour;
						}
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
			if (depth == 8) {
				record.directLighting = computeDirectGuidedPath(shadingData, sampler, [](const Vec4&) { return 0.f; });
				records.push_back(record);
				return;
			}

			// Must run Path Guiding before calculating direct lighting
			bool usePathGuiding = false;
			if (isGuidingPhase && cache != nullptr && !isSpecular) {
				// We are in the Path Guiding Phase
				float radius = 0.2f;
				float radiusSq = radius * radius;
				nearbyVertices.clear();

				// BVH Nearest Path Vertices Search will be timed inside this scope
				{
					Timer bvhSearchTimer(stats.bvhSearchTimeMs);
					cache->search(shadingData.x, radiusSq, nearbyVertices, MAX_NEARBY_VERTICES);
				}

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
				// QTree Building will be timed inside this scope
				{
					Timer qTreeBuildTimer(stats.qTreeBuildTimeMs);
					for (auto& vertex : nearbyVertices) {
						if (Dot(shadingData.sNormal, vertex->normal) < 0.5f) continue;
						Vec4 wiLocal = shadingData.frame.toLocal(vertex->wi);
						if (wiLocal.z < 0.05f) continue;

						// BSDF inversion to get u, v, and u_lobe
						float u, v, u_lobe;

						// BSDF Inversion will be timed inside this scope
						{
							Timer bsdfInvertTimer(stats.qTreeBuildTimeMs);
							shadingData.bsdf->invert(shadingData, vertex->wi, u, v, u_lobe);
						}
						// Timing completed
						if (u < 0.f || u > 1.f || v < 0.f || v > 1.f || std::isnan(u) || std::isnan(v)) continue;

						// Insert into the QTree
						float weight = std::sqrt(std::max(0.f, vertex->Li.Lum()));
						qTree.insert(u, v, weight);
						acceptedVertices++;
					}
				}
				// Timing completed
				usePathGuiding = acceptedVertices >= 50;
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

				// If not valid u, return BSDF PDF, multiplied with BSDF fraction
				if (u < 0.f) return basePdf * BSDF_FRACTION;

				// If inversion OK, return MIS combined PDF for Path Guiding
				float qTreePdf = (QTREE_MIX * qTree.pdf(u, v)) + UNIFORM_MIX;
				return basePdf * (BSDF_FRACTION + QTREE_FRACTION * qTreePdf);
			};

			// Save the direct lighting (NEE) to the record
			record.directLighting = computeDirectGuidedPath(shadingData, sampler, forwardPdf);

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
					float u_lobe_out = sampler->next();

					float u_out = 0.f, v_out = 0.f, treePdfIgnored = 0.f;
					if (sampler->next() < UNIFORM_MIX) { u_out = sampler->next(); v_out = sampler->next(); }
					else { qTree.sample(r1, r2, u_out, v_out, treePdfIgnored); }

					// Then do BSDF sampling with the new numbers obtained
					GuidedPathSampler dummySampler;
					dummySampler.set(u_out, v_out, u_lobe_out);
					float pdfIgnored = 0.f;
					wi = shadingData.bsdf->sample(shadingData, &dummySampler, fBsdf, pdfIgnored);
				} else {
					// Do Standart Sampling
					float pdfIgnored = 0.f;
					wi = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfIgnored);
				}
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
				float qDebug = (uDebug >= 0.f) ? (QTREE_MIX * qTree.pdf(uDebug, vDebug) + UNIFORM_MIX) : 0.f;
				
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
			if (pdfCombined <= 0.f || std::isnan(pdfCombined) || std::isinf(pdfCombined)) {
				records.push_back(record);
				return;
			}

			// Check cosine term
			float cosTheta = fabs(Dot(wi, shadingData.sNormal));
			if (cosTheta <= 0.f) { records.push_back(record); return; }

			// Store the necessary records in the record structure
			record.wi = wi;
			record.bsdfWeight = (fBsdf * cosTheta) / (pdfCombined * rrpRecord);

			// Just a way to deal with Firefly Artifactings
			float maxLuminance = 20.f;  // Should be somewhere between 10 and 50
			float currentLuminance = record.bsdfWeight.Lum();

			if (currentLuminance > maxLuminance) {
				record.bsdfWeight = record.bsdfWeight * (maxLuminance / currentLuminance);
			}

			record.storeRecord = !isSpecular;
			records.push_back(record);

			// Define indirect ray for the next bounce, and recurse through the function
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Ray indirectRay(shadingData.x + shadingData.gNormal * (EPSILON * sign), wi);
			generatePathRecursive(indirectRay, depth + 1, sampler, records, cache, qTree, isGuidingPhase, nearbyVertices, stats, pdfCombined, isSpecular);
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

			// Handle degenerate pAs
			if (pALight > 0.f && pABsdf > 0.f) {
				// Calculate Weight for MIS
				float wind = weightPowerHeuristics(pABsdf, pALight);
				record.misEmission = backgroundColour * wind;
			}
			else {
				record.misEmission = backgroundColour;
			}
		}
		records.push_back(record);
		return;
	}
	// --- Path Guiding Algorithm Work End ---

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
			if (depth == 8) return direct;

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
			if (pdfBsdf <= 0.f) return direct;
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

		// Handle degenerate PMF / PDF cases
		if (pmfLight <= 0.f || pdfLight <= 0.f) return Colour(0.f, 0.f, 0.f);

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

	Colour viewBarycentrics(Ray & r) {
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

								#if GUIDED_PATH
								Colour col = guidedPath(ray, &samplers[i], currentThreadPathVertexRecords, cacheBVH, currentThreadQTree, isGuidingPhase, currentThreadStats, enableNEE);
								#else
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
			cacheBVH = new PointBVHNode();
			cacheBVH->buildPointBVHNode(std::move(globalCacheList));

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