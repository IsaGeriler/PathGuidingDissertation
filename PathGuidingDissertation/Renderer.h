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
#include "QTree.h"
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
	Vec4 wi;	    // Incoming Direction
	Colour Li;		// Incoming Radiance
};

struct ForwardPassRecord {
	Vec4 position;			   // Hit Point (shadingData.x)
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
	int maxSPP = 128;
	// int maxSPP = 8192;
	int learningThreshold = maxSPP / 8;

	// Path Vertex vector to then cache saved items over at a Spatial Accelleration Structure
	std::vector<std::vector<PathVertex>> perThreadPathVertexRecords;
	QTree* perThreadQTrees;

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
			Vec4 wi = light->sample(shadingData, sampler, emittedColour, pdfLight);
			if (pdfLight <= 0.f) return Colour(0.f, 0.f, 0.f);

			// Evaluate visibility to outside scene bounds
			// Scene Bounds Fix: Replaces calculated SceneBounds AABB length with SceneBounds sceneRadius and sceneCentre
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
	Colour guidedPath(Ray& r, Sampler* sampler, std::vector<PathVertex>& pathVertices, PointBVHNode* sTree, QTree& dTree, bool isGuidingPhase) {
		// --- 1. Forward Pass Phase ---
		std::vector<ForwardPassRecord> records;
		std::vector<const PathVertex*> nearbyVertices;
		records.reserve(10);          // Max depth: 8, + 2 buffer space
		nearbyVertices.reserve(200);  // Pre-allocate max number of wanted vertices

		// Generate the path vertices in the forward pass (only populate the vector)
		generatePathRecursive(r, 0, sampler, records, sTree, dTree, isGuidingPhase, nearbyVertices);

		// --- 2. Backpropagation Phase ---
		// Store Each Path Vertex to the vector via Backpropagation
		Colour guidingLi(0.f, 0.f, 0.f);
		Colour pixelColour(0.f, 0.f, 0.f);
		for (int i = (int)(records.size() - 1); i >= 0; i--) {
			// Learning Phase: Data Collection
			if (!isGuidingPhase) {
				if (records[i].storeRecord) {
					PathVertex pathVertex;
					pathVertex.position = records[i].position;
					pathVertex.wi = records[i].wi;
					pathVertex.Li = guidingLi;
					// Check if the incoming/incident radiance is valid before feeding the vector
					// This will also avoid memory bloat since we discard zero luminance contributions!
					if (pathVertex.Li.isValid() && pathVertex.Li.Lum() > 0) pathVertices.push_back(pathVertex);
				}
				// No NEE, no MIS, just backpropagate the incoming radiance
				guidingLi = records[i].emission + (records[i].bsdfWeight * guidingLi);
			}
			// Rendering Phase: Compute the pixel colour with NEE and MIS (camera)
			pixelColour = records[i].misEmission + records[i].directLighting + (records[i].bsdfWeight * pixelColour);
		}
		// Return the pixel colour for the rendering phase or return the guidingLi for the learning phase
		return (!isGuidingPhase) ? guidingLi : pixelColour;
	}

	void generatePathRecursive(Ray& r, int depth, Sampler* sampler, std::vector<ForwardPassRecord>& records, PointBVHNode* cache, QTree& qTree, bool isGuidingPhase, std::vector<const PathVertex*>& nearbyVertices, float previousBsdfPdf = 0.f, bool previousSurfaceSpecular = false) {
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		// Create an empty record for now, will fill in later in this function
		ForwardPassRecord record;
		record.position = (shadingData.t < FLT_MAX) ? shadingData.x : Vec4(0.f, 0.f, 0.f);
		record.wi = Vec4(0.f, 0.f, 0.f);
		record.bsdfWeight = Colour(0.f, 0.f, 0.f);
		record.directLighting = Colour(0.f, 0.f, 0.f);
		record.emission = Colour(0.f, 0.f, 0.f);
		record.misEmission = Colour(0.f, 0.f, 0.f);
		record.storeRecord = false;

		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) {
				Colour emittedColour = shadingData.bsdf->emit(shadingData, shadingData.wo);
				record.emission = emittedColour;
				if (depth == 0 || previousSurfaceSpecular) { record.misEmission = emittedColour; }
				else {
					// Evaluate MIS for Area Light
					// Area Light PDF and PMF
					float pmfLight = 1.f / scene->lights.size();
					float pdfLight = 1.f / scene->triangles[intersection.ID].area;

					// Handle degenerate PMF / PDF cases
					if (pmfLight >= 0.f && pdfLight >= 0.f) {
						float cosThetaPrime = std::max(Dot(-r.dir, scene->triangles[intersection.ID].gNormal()), 0.f);
						float distanceSquare = SQ(intersection.t);
						if (cosThetaPrime >= 0.f && distanceSquare > EPSILON) {
							// Calculate pA of Light and BSDF for MIS
							float pALight = pdfLight * pmfLight;
							float pABsdf = previousBsdfPdf * cosThetaPrime / distanceSquare;

							// Handle degenerate pA
							if (pALight >= 0.f && pABsdf >= 0.f) {
								// Calculate Weight for MIS
								float wind = weightPowerHeuristics(pABsdf, pALight);
								record.misEmission = emittedColour * wind;
							}
						}
					}
				}
				records.push_back(record);
				return;
			}
			// Save the direct lighting (NEE) to the record
			record.directLighting = computeDirect(shadingData, sampler);

			// Apply Russian Roulette Starting at the ray depth 4
			// Russian Roulette should kick in normally between at depth 3 to 5
			float rrpRecord = 1.f;
			if (depth > 3) {
				// Using a fixed RRP because we don't have access to pathThroughput anymore...
				float rrp = 0.7f;
				if (sampler->next() < rrp) { rrpRecord = rrp; }
				else { records.push_back(record); return; }
			}

			// Terminate when the ray depth exceeds 8 bounces, to avoid infinite recursion
			// We will work on SD-domain unlike Guo et al. 2018, in which they were restricted with n = m = 2
			if (depth == 8) { records.push_back(record); return; }

			// Calculate Indirect Lighting - Sampling Proportional to BSDF (Materials)
			float pdfBsdf = 0.f;
			Colour fBsdf;
			Vec4 wi;

			// --- NEW ---
			// -> When sampling, replace BSDF with the new method, i.e.
			//    -> Search for nearby vertices from 1.
			//    -> Project wi into PSS
			//    -> Invert BSDF sampling
			//    -> Sample PSS
			bool isSpecular = shadingData.bsdf->isPureSpecular();
			if (isGuidingPhase && cache != nullptr && !isSpecular) {
				// Guiding Phase
				// Search for nearby vertices
				float radius = 0.2f;
				float radiusSq = radius * radius;
				nearbyVertices.clear();
				cache->search(shadingData.x, radiusSq, nearbyVertices, 200);
				if (nearbyVertices.size() < 50) {
					wi = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfBsdf);
				} else {
					for (auto& vertex : nearbyVertices) {
						// BSDF inversion to get u, v, and u_lobe for QTree insertion
						float u = 0.f, v = 0.f, u_lobe = 0.f;
						shadingData.bsdf->invert(shadingData, vertex->wi, u, v, u_lobe);
						qTree.insert(u, v, vertex->Li.Lum());
					}
					float strategy = sampler->next();
					float qTree_pdf = 0.f;
					
					// Pick a stragety to either guide the sampling or standard BSDF sampling
					if (strategy < 0.5f) {
						// Do Guided Sampling
						float r1 = sampler->next();
						float r2 = sampler->next();
						float u_lobe_out = sampler->next();
						
						float u_out = 0.f, v_out = 0.f;
						qTree.sample(r1, r2, u_out, v_out, qTree_pdf);
						
						// Then do BSDF sampling with the new numbers obtained
						GuidedPathSampler dummySampler;
						dummySampler.set(u_out, v_out, u_lobe_out);
						wi = shadingData.bsdf->sample(shadingData, &dummySampler, fBsdf, pdfBsdf);
					} else {
						// Do Standart Sampling
						wi = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfBsdf);

						// What would be the pdf of the tree would be like if we were to invert BSDF?
						float u_out = 0.f, v_out = 0.f, u_lobe = 0.f;
						shadingData.bsdf->invert(shadingData, wi, u_out, v_out, u_lobe);
						qTree_pdf = qTree.pdf(u_out, v_out);
					}
					// finalPdf = c1 * pdfBsdf + c2 * pdfLi; where c1 = 0.5, and c2 = 0.5
					// float qTree_solid_angle = qTree_pdf * pdfBsdf;
					// pdfBsdf = (c1 * pdfBsdf) + (c2 * qTree_solid_angle);
					pdfBsdf = pdfBsdf * (0.5f + 0.5f * qTree_pdf);
					if (pdfBsdf <= 0.f || std::isnan(pdfBsdf) || std::isinf(pdfBsdf)) {
						qTree.clear();
						records.push_back(record);
						return;
					}
					qTree.clear();
				}
			} else {
				// Learning Phase
				// Sample BSDF as normal
				wi = shadingData.bsdf->sample(shadingData, sampler, fBsdf, pdfBsdf);
			}
			float cosTheta = fabs(Dot(wi, shadingData.sNormal));
			if (pdfBsdf <= 0.f || cosTheta <= 0.f) { records.push_back(record); return; }

			// Now we update wi, Li, and storeRecord as we got the indirect radiance and it's bounce
			bool isPreviousSurfaceSpecular = shadingData.bsdf->isPureSpecular();

			// Store the necessary records in the record structure
			record.wi = wi;
			record.bsdfWeight = (fBsdf * cosTheta) / (pdfBsdf * rrpRecord);
			record.storeRecord = !isPreviousSurfaceSpecular;
			records.push_back(record);

			// Define indirect ray (for the next bounce) and recurse through the function
			float sign = (Dot(wi, shadingData.gNormal) >= 0.f) ? 1.f : -1.f;
			Ray indirectRay(shadingData.x + shadingData.gNormal * (EPSILON * sign), wi);
			generatePathRecursive(indirectRay, depth + 1, sampler, records, cache, qTree, isGuidingPhase, nearbyVertices, pdfBsdf, isPreviousSurfaceSpecular);
			return;
		}
		Colour backgroundColour = scene->background->evaluate(r.dir);
		record.emission = backgroundColour;
		if (depth == 0 || previousSurfaceSpecular) { record.misEmission = backgroundColour; }
		else if (backgroundColour.Lum() > 1e-8f) {
			// Evaluate MIS for Environment Map
			// Infinite Light PDF and PMF
			float pmfLight = 1.f / scene->lights.size();
			float pdfLight = scene->background->PDF(shadingData, r.dir);

			// Handle degenerate PMF / PDF cases
			if (pmfLight >= 0.f && pdfLight >= 0.f) {
				// Calculate pA of Light and BSDF for MIS
				float pALight = pmfLight * pdfLight;
				float pABsdf = previousBsdfPdf;

				// Handle degenerate pA
				if (pALight >= 0.f && pABsdf >= 0.f) {
					// Calculate Weight for MIS
					float wind = weightPowerHeuristics(pABsdf, pALight);
					record.misEmission = backgroundColour * wind;
				}
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

			// Apply Russian Roulette Starting at the ray depth 4
			// Russian Roulette should kick in normally between at depth 3 to 5
			if (depth > 3) {
				float rrp = std::min(std::max(EPSILON, pathThroughput.Lum()), 1.f);
				if (sampler->next() < rrp) pathThroughput = pathThroughput / rrp;
				else return direct;
			}

			// Terminate when the ray depth exceeds 16 bounces, to avoid infinite recursion
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

								// View Barycentrics / Shading Normals / Albedo / Direct Lighting / Path Trace OR Path Guiding
								//Colour col = viewBarycentrics(ray);
								//Colour col = viewNormals(ray);
								//Colour col = albedo(ray);
								//Colour col = direct(ray, &samplers[i]);

								#if GUIDED_PATH
								Colour col = guidedPath(ray, &samplers[i], currentThreadPathVertexRecords, cacheBVH, currentThreadQTree, isGuidingPhase);
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
		if (!isGuidingPhase) {
			// --- Learning Phase ---
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

		if ((getSPP() == learningThreshold - 1)) {
			// Build the BVH exactly once, after collecting cache estimates in the learning phase
			std::cout << "Learning phase finished. Building BVH for cached path vertices..." << std::endl;
			std::cout << "Total cached path vertices: " << globalCacheList.size() << std::endl;
			cacheBVH = new PointBVHNode();
			cacheBVH->buildPointBVHNode(std::move(globalCacheList));

			// Clear Global Cache Estimates List
			globalCacheList.clear();
			globalCacheList.shrink_to_fit();
			
			// Clear the canvas to avoid confusion between learning and guiding phases
			// film->clear() resets the SPP counter, so I implemented film->clearCanvas()
			// which resets the canvas, and the accumulated SPP, while keeping the SPP counter intact for guiding phase
			std::cout << "BVH built successfully. Entering guiding phase..." << std::endl;
			film->clearCanvas();
		}
		#endif
	}

	int getSPP() { return film->SPP; }
	void saveHDR(std::string filename) { film->save(filename); }
	void savePNG(std::string filename) { stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3); }
};