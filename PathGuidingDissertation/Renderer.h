#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <chrono>
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
#include "Sampling.h"
#include "Scene.h"

#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

#define GUIDED_PATH true

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
	Vec4 normal;	// Shading Normal
	Vec4 wi;	    // Incoming Direction
	Colour Li;		// Incoming Radiance
};

struct Record {
	Vec4 position;			   // Hit Point (shadingData.x)
	Vec4 normal;			   // Shading Normal
	Vec4 wi;				   // Incoming Direction
	Colour direct;			   // Direct Lighting, calculated via NEE
	Colour indirect;		   // Indirect Lighting without pathThroughput, (fBsdf * cosTheta) / (pdfBsdf * rrp)
	bool storeRecord = false;  // Do not store if previous surface is pure specular
};

struct PointBVHNodeStats {
	int nodeCount = 0, leafNodeCount = 0;
	int minLeafDepth = INT_MAX, maxLeafDepth = -INT_MAX;
	long long sumLeafDepth = 0;
	size_t memory_in_bytes = 0;
	double buildTimeMs = 0.0;
};

// --- STree Component to Store PathVertex Caches ---
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
public:
	// Constructor
	PointBVHNode() { r = nullptr; l = nullptr; }

	// Destructor
	~PointBVHNode() {
		if (r != nullptr) delete r;
		if (l != nullptr) delete l;
		if (!pathVertexRecords.empty()) pathVertexRecords.clear();
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
};

class RayTracer {
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom* samplers;
	std::thread** threads;
	unsigned int numProcs;
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
		clear();
	}

	void clear() {
		film->clear();
	}

	Colour computeDirect(ShadingData shadingData, Sampler* sampler) {
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

			// Calculate visibility
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Vec4 shadowRayOffset(shadingData.x + shadingData.gNormal * (EPSILON * sign));
			if (scene->visible(shadowRayOffset, pointOnLight)) {
				// Calculate Geometry Term
				float denominator = surfaceToLight.lengthSquare();
				if (denominator < EPSILON) return Colour(0.f, 0.f, 0.f);
				float cosTheta = std::max(Dot(wi, shadingData.sNormal), 0.f);
				float cosThetaPrime = std::max(Dot(-wi, light->normal(shadingData, wi)), 0.f);
				float geometryTerm = (cosTheta * cosThetaPrime) / denominator;

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
			Vec4 directionToLight = light->sample(shadingData, sampler, emittedColour, pdfLight);
			if (pdfLight <= 0.f) return Colour(0.f, 0.f, 0.f);

			// Normalize direction to light
			Vec4 wi = directionToLight.normalize();

			// Evaluate visibility to outside scene bounds
			// FIX: Scene Bounds Fix, most noticable in Sibenik Scene
			// Replaces calculated SceneBounds AABB length with SceneBounds sceneRadius and sceneCentre
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Vec4 shadowRayOffset(shadingData.x + shadingData.gNormal * (EPSILON * sign));
			Vec4 sceneBoundOffset = use<SceneBounds>().sceneCentre + (wi * use<SceneBounds>().sceneRadius);
			if (scene->visible(shadowRayOffset, sceneBoundOffset)) {
				// Evaluate Geometry Term for environment maps
				// It's just cosTheta because there is no next surface
				float cosTheta = std::max(Dot(wi, shadingData.sNormal), 0.f);
				if (cosTheta <= 0.f) return Colour(0.f, 0.f, 0.f);

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
	Colour guidedPath(Ray& r, Sampler* sampler, std::vector<PathVertex>& pathVertices) {
		std::vector<Record> records;
		records.clear();

		Colour pathThroughput(1.f, 1.f, 1.f);
		Colour terminatedColour(0.f, 0.f, 0.f);
		Colour result = guidedPathRecursive(r, pathThroughput, terminatedColour, 0, sampler, records);

		// Store Each Path Vertex to the vector via Backpropagation
		Colour incomingRadiance = terminatedColour;
		for (int i = (int)(records.size() - 1); i >= 0; i--) {
			// If store record flag is true, store the path vertex
			if (records[i].storeRecord) {
				PathVertex pathVertex;
				pathVertex.position = records[i].position;
				pathVertex.normal = records[i].normal;
				pathVertex.wi = records[i].wi;
				pathVertex.Li = incomingRadiance;
				pathVertices.push_back(pathVertex);
			}
			incomingRadiance = records[i].direct + (records[i].indirect * incomingRadiance);
		}
		float difference = fabs(incomingRadiance.Lum() - result.Lum());
		assert(difference <= (std::max(result.Lum(), 1.f) * EPSILON) && "Backpropagation is not equal to the estimator! Capture side bug, assertion failed.");
		return result;
	}

	Colour guidedPathRecursive(Ray& r, Colour& pathThroughput, Colour& terminatedColour, int depth, Sampler* sampler, std::vector<Record>& records, float previousBsdfPdf = 0.f, bool previousSurfaceSpecular = false) {
		// The logic for the path guiding algorithm will go here...
		// Sampling will be done on the quad-tree (directional), path vertex will be stored in a BVH (spatial)
		// So unlike Guo et al. 2018, we will be working on the Spatio-Directional Space
		// Some notes, so far...
		//
		// 1. Path Trace (BSDF)
		// -> Store each path vertex in an accelleration structire (BVH)
		//    -> Position
		//    -> wi (incoming direction)
		//    -> Incoming radiance (Li)
		//
		// 2. Path Trace
		// -> When sampling, replace BSDF with the new method, i.e.
		//    -> Search for nearby vertices from 1.
		//    -> Project wi into PSS
		//    -> Invert BSDF sampling
		//    -> Sample PSS
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) {
				Colour emittedColour = shadingData.bsdf->emit(shadingData, shadingData.wo);
				if (depth == 0 || previousSurfaceSpecular) { terminatedColour = emittedColour; return pathThroughput * emittedColour; }
				// Evaluate MIS for Area Light
				// Area Light PDF and PMF
				float pmfLight = 1.f / scene->lights.size();
				float pdfLight = 1.f / scene->triangles[intersection.ID].area;

				// Handle degenerate PMF / PDF cases
				if (pmfLight <= 0.f || pdfLight <= 0.f) return Colour(0.f, 0.f, 0.f);

				float cosThetaPrime = std::max(Dot(-r.dir, scene->triangles[intersection.ID].gNormal()), 0.f);
				if (cosThetaPrime <= 0.f) return Colour(0.f, 0.f, 0.f);
				float distanceSquare = SQ(intersection.t);
				if (distanceSquare < EPSILON) return Colour(0.f, 0.f, 0.f);

				// Calculate pA of Light and BSDF for MIS
				float pALight = pdfLight * pmfLight;
				float pABsdf = previousBsdfPdf * cosThetaPrime / distanceSquare;

				// Handle degenerate pA
				if (pALight < 0.f || pABsdf < 0.f) return Colour(0.f, 0.f, 0.f);

				// Calculate Weight for MIS
				float wind = weightPowerHeuristics(pABsdf, pALight);
				terminatedColour = emittedColour * wind;
				return pathThroughput * emittedColour * wind;
			}
			// Calculate Direct Lighting (NEE) and multiply with pathThroughput
			Colour NEE = computeDirect(shadingData, sampler);
			Colour direct = pathThroughput * NEE;

			// Store the necessary records in the record structure (do this before Russian Roulette step)
			Record record;
			record.position = shadingData.x;
			record.normal = shadingData.sNormal;
			record.wi = Vec4(0.f, 0.f, 0.f);
			record.direct = NEE;  // Obtained from computeDirect() without * pathThroughput
			record.indirect = Colour(0.f, 0.f, 0.f);
			record.storeRecord = false;
			records.push_back(record);

			// Since the list will grow as bounces happen, the size - 1 will always give index for the current bounce
			int index = (int)(records.size() - 1);

			// Apply Russian Roulette Starting at the ray depth 4
			// Russian Roulette should kick in normally between at depth 3 to 5
			float rrpRecord = 1.f;
			if (depth > 3) {
				// Clamp between EPSILON and 1 to avoid division by zero
				float rrp = std::min(std::max(EPSILON, pathThroughput.Lum()), 1.f);
				if (sampler->next() < rrp) { rrpRecord = rrp; pathThroughput = pathThroughput / rrp; }
				else return direct;
			}

			// Terminate when the ray depth exceeds 8 bounces, to avoid infinite recursion
			// We will work on SD-domain unlike Guo et al. 2018, in which they were restricted with n = m = 2
			if (depth == 8) return direct;

			// Calculate Indirect Lighting - Sampling Proportional to BSDF (Materials)
			float pdfBsdf = 0.f;
			Colour fBsdf;
			Vec4 wi = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfBsdf);
			if (pdfBsdf <= 0.f) return direct;
			
			// Define indirect ray (for the next bounce)
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Ray indirectRay(shadingData.x + shadingData.gNormal * (EPSILON * sign), wi);

			// Update throughput
			// Taking absolute value of cosTheta to fix GlassBSDF rendering
			float cosTheta = fabs(Dot(wi, shadingData.sNormal));
			if (cosTheta <= 0.f) return direct;
			pathThroughput = ((pathThroughput * fBsdf * cosTheta) / pdfBsdf);

			// Eliminate zero-luminance throughput
			if (pathThroughput.Lum() <= 0.f) return direct;

			// Check for NaN/Inf or negative channel values
			if (std::isnan(pathThroughput.r) || std::isnan(pathThroughput.g) || std::isnan(pathThroughput.b)) return direct;
			if (std::isinf(pathThroughput.r) || std::isinf(pathThroughput.g) || std::isinf(pathThroughput.b)) return direct;
			if (pathThroughput.r < 0.f || pathThroughput.g < 0.f || pathThroughput.b < 0.f) return direct;

			// Now we update wi, Li, and storeRecord as we got the indirect radiance and it's bounce
			bool isPreviousSurfaceSpecular = shadingData.bsdf->isPureSpecular();
			records[index].wi = wi;
			records[index].indirect = (fBsdf * cosTheta) / (pdfBsdf * rrpRecord);
			records[index].storeRecord = !isPreviousSurfaceSpecular;

			// Recurse through the function, contribute with direct lighting
			return direct + guidedPathRecursive(indirectRay, pathThroughput, terminatedColour, depth + 1, sampler, records, pdfBsdf, isPreviousSurfaceSpecular);
		}
		Colour backgroundColour = scene->background->evaluate(r.dir);
		if (depth == 0 || previousSurfaceSpecular) { terminatedColour = backgroundColour; return pathThroughput * backgroundColour; }
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

		// Handle degenerate pA
		if (pALight < 0.f || pABsdf < 0.f) return Colour(0.f, 0.f, 0.f);

		// Calculate Weight for MIS
		float wind = weightPowerHeuristics(pABsdf, pALight);
		terminatedColour = backgroundColour * wind;
		return pathThroughput * backgroundColour * wind;
	}
	// --- Path Guiding Algorithm Work End ---

	Colour pathTraceRecursive(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, float previousBsdfPdf = 0.f, bool previousSurfaceSpecular = false) {
		// Trace ray
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) {
				if (depth == 0 || previousSurfaceSpecular) return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
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
				return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo) * wind;
			}
			// Calculate Direct Lighting
			Colour direct = pathThroughput * computeDirect(shadingData, sampler);

			// Apply Russian Roulette Starting at the ray depth 4
			// Russian Roulette should kick in normally between at depth 3 to 5
			if (depth > 3) {
				float rrp = std::min(std::max(EPSILON, pathThroughput.Lum()), 1.f);
				if (sampler->next() < rrp) pathThroughput = pathThroughput / rrp;
				else return direct;
			}

			// Terminate when the ray depth exceeds 8 bounces, to avoid infinite recursion
			if (depth == 8) return direct;
			
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

			// Eliminate zero-luminance
			if (pathThroughput.Lum() <= 0.f) return direct;

			// Check for NaN/Inf or negative channel values
			if (std::isnan(pathThroughput.r) || std::isnan(pathThroughput.g) || std::isnan(pathThroughput.b)) return direct;
			if (std::isinf(pathThroughput.r) || std::isinf(pathThroughput.g) || std::isinf(pathThroughput.b)) return direct;
			if (pathThroughput.r < 0.f || pathThroughput.g < 0.f || pathThroughput.b < 0.f) return direct;
			
			// Recurse until path terminated
			bool isPreviousSurfaceSpecular = shadingData.bsdf->isPureSpecular();
			return direct + pathTraceRecursive(indirectRay, pathThroughput, depth + 1, sampler, pdfBsdf, isPreviousSurfaceSpecular);
		}
		if (depth == 0 || previousSurfaceSpecular) return pathThroughput * scene->background->evaluate(r.dir);
		if (scene->background->evaluate(r.dir).Lum() < 1e-8f) return Colour(0.f, 0.f, 0.f);
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
		return pathThroughput * scene->background->evaluate(r.dir) * wind;
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

		// Cached vertices will be stored in a BVH structure
		PointBVHNode cache;

		// Path Vertex vector to then cache saved items over at a Spatial Accelleration Structure
		std::vector<std::vector<PathVertex>> perThreadPathVertexRecords(numProcs);

		// Get total tile count
		unsigned int tile_size = 32;
		unsigned int tiles_x = (film->width + tile_size - 1u) / tile_size;
		unsigned int tiles_y = (film->height + tile_size - 1u) / tile_size;
		unsigned int total_tile_count = tiles_x * tiles_y;

		// Threads
		for (unsigned int i = 0; i < numProcs; ++i) {
			threads[i] = new std::thread(
				// Capture i by value, not reference, or else samplers[i] will go out of bounds!
				[&, i]() {
					// Lambda function to render tiles
					unsigned int tile_id = 0;
					std::vector<PathVertex>& threadPathVertexRecords = perThreadPathVertexRecords[i];

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

								// View Barycentrics / Shading Normals / Albedo / Direct Lighting / Path Trace
								// TO:DO - View all via DearImGui
								//Colour col = viewBarycentrics(ray);
								//Colour col = viewNormals(ray);
								//Colour col = albedo(ray);
								//Colour col = direct(ray, &samplers[i]);
								
								#if GUIDED_PATH
								Colour col = guidedPath(ray, &samplers[i], threadPathVertexRecords);
								#else
								Colour col = pathTrace(ray, &samplers[i]);
								#endif

								// Check for NaN/Inf values
								if (std::isnan(col.r) || std::isnan(col.g) || std::isnan(col.b) ||
									std::isinf(col.r) || std::isinf(col.g) || std::isinf(col.b)) {
									continue;
								}

								// Splat, Tonemap, and Draw to Pixel
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
		std::vector<PathVertex> globalPathVertexRecords;
		size_t total_size = 0;

		// Get the total size and allocate space on the global list
		for (auto& recordsList : perThreadPathVertexRecords) {
			total_size += recordsList.size();
		}
		globalPathVertexRecords.reserve(total_size);

		// Carry the data obtained from tiles to global list
		for (auto& recordsList : perThreadPathVertexRecords) {
			globalPathVertexRecords.insert(
				globalPathVertexRecords.end(),
				std::make_move_iterator(recordsList.begin()),
				std::make_move_iterator(recordsList.end())
			);
			recordsList.clear();
			recordsList.shrink_to_fit();
		}

		// Then build the BVH after gathering the global records
		cache.buildPointBVHNode(std::move(globalPathVertexRecords));
		#endif
	}

	int getSPP() { return film->SPP; }
	void saveHDR(std::string filename) { film->save(filename); }
	void savePNG(std::string filename) { stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3); }
};