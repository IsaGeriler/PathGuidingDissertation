#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>

#include "Geometry.h"
#include "Imaging.h"
#include "Lights.h"
#include "Materials.h"
#include "MyMath.h"
#include "Sampling.h"
#include "Scene.h"

#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

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

// --- STree Component to Store PathVertex Caches ---
class PointBVHNode {
private:
	// Attributes
	AABB bounds;
	PointBVHNode* r;
	PointBVHNode* l;
	int offset = 0, used = 0;

	// Private Methods
	bool isLeaf() const { return l == nullptr && r == nullptr; }
	
	void subdivide(std::vector<PathVertex>& pathVertices) {
		// Add recursive subdivision here with median splitting
		if (used <= 8) return;
		Vec4 extend = bounds.max - bounds.min;
		int ax = 0;
		if (extend.y > extend.x) ax = 1;
		if (extend.z > extend[ax]) ax = 2;
		float splitPos = bounds.min[ax] + extend[ax] * 0.5;
		int i = offset;
		int j = i + used - 1;
		while (i <= j) {
			if (pathVertices[i].position[ax] < splitPos) i++;
			else std::swap(pathVertices[i], pathVertices[j--]);
		}
		int leftCount = i - offset;
		if (leftCount == 0 || leftCount == used) return;

		l = new PointBVHNode();
		l->offset = offset;
		l->used = leftCount;

		r = new PointBVHNode();
		r->offset = i;
		r->used = used - leftCount;

		used = 0;

		l->updateBounds(pathVertices);
		r->updateBounds(pathVertices);

		l->subdivide(pathVertices);
		r->subdivide(pathVertices);
	}

	void updateBounds(std::vector<PathVertex>& pathVertices) {
		// Add update bounds code here
		bounds.reset();
		for (int i = offset; i < offset + used; i++) {
			Vec4 positionVector = pathVertices[i].position;
			bounds.extend(positionVector);
		}
	}
public:
	// Constructor
	PointBVHNode() { r = nullptr; l = nullptr; }

	// Destructor
	~PointBVHNode() { if (r != nullptr) delete r; if (l != nullptr) delete l; }

	// Public Methods
	void buildPointBVHNode(std::vector<PathVertex>& inputPathVertices) {
		// Add PointBVHNode building code here
		if (inputPathVertices.empty()) return;
		offset = 0; used = (int)inputPathVertices.size();
		updateBounds(inputPathVertices);
		subdivide(inputPathVertices);
		std::cout << "PointBVHBuild Successfull\n";
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
		// Using thread_local to avoid crashing on tiled rendering due to accessing bad memory
		thread_local std::vector<Record> records;
		records.clear();

		Colour pathThroughput(1.f, 1.f, 1.f);
		Colour terminatedColour(0.f, 0.f, 0.f);
		Colour result = guidedPathRecursive(r, pathThroughput, terminatedColour, 0, sampler, records);

		// Store Each Path Vertex to the vector via Backpropogation
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
				Colour emitted = shadingData.bsdf->emit(shadingData, shadingData.wo);
				if (depth == 0 || previousSurfaceSpecular) { terminatedColour = emitted; return pathThroughput * emitted; }
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
				terminatedColour = emitted * wind;
				return pathThroughput * emitted * wind;
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
		Colour background = scene->background->evaluate(r.dir);
		if (depth == 0 || previousSurfaceSpecular) { terminatedColour = background; return pathThroughput * background; }
		if (background.Lum() < 1e-8f) return Colour(0.f, 0.f, 0.f);
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
		terminatedColour = background * wind;
		return pathThroughput * background * wind;
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
								//Colour col = pathTrace(ray, &samplers[i]);
								Colour col = guidedPath(ray, &samplers[i], threadPathVertexRecords);

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

		std::vector<PathVertex> globalPathVertexRecords;
		size_t total_size = 0;
		for (auto& recordsList : perThreadPathVertexRecords) {
			total_size += recordsList.size();
		}
		globalPathVertexRecords.reserve(total_size);

		for (auto& recordsList : perThreadPathVertexRecords) {
			globalPathVertexRecords.insert(
				globalPathVertexRecords.end(),
				std::make_move_iterator(recordsList.begin()),
				std::make_move_iterator(recordsList.end())
			);
			recordsList.clear();
			recordsList.shrink_to_fit();
		}

		// Testing
		std::cout << "Global Path Vertex List Size: " << globalPathVertexRecords.size() << std::endl;
		// List 1000 elements 
		for (int i = 0; i < 1000; i++) {
			std::cout << "Global Path Vertex #" << i << " ["
				<< "\n\tPosition: (" << globalPathVertexRecords[i].position.x << ", " << globalPathVertexRecords[i].position.y << ", " << globalPathVertexRecords[i].position.z << ")"
				<< "\n\tNormal: (" << globalPathVertexRecords[i].normal.x << ", " << globalPathVertexRecords[i].normal.y << ", " << globalPathVertexRecords[i].normal.z << ")"
				<< "\n\tIncoming Direction: (" << globalPathVertexRecords[i].wi.x << ", " << globalPathVertexRecords[i].wi.y << ", " << globalPathVertexRecords[i].wi.z << ")"
				<< "\n\tIncoming Radiance: (" << globalPathVertexRecords[i].Li.r << ", " << globalPathVertexRecords[i].Li.g << ", " << globalPathVertexRecords[i].Li.b << ")"
				<< "\n]\n" << std::endl;
		}
		// Then build the BVH after gathering the records
		cache.buildPointBVHNode(globalPathVertexRecords);
	}

	int getSPP() { return film->SPP; }
	void saveHDR(std::string filename) { film->save(filename); }
	void savePNG(std::string filename) { stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3); }
};