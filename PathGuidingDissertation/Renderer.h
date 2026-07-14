#pragma once

#include <algorithm>
#include <atomic>
#include <thread>

#include "Geometry.h"
#include "Imaging.h"
#include "Lights.h"
#include "Materials.h"
#include "MyMath.h"
#include "Sampling.h"
#include "Scene.h"

#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

#define ENABLE_MULTITHREAD true

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
		//film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new MitchellNetravaliFilter());
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

			// Terminate when the ray depth exceeds 25 bounces, to avoid infinite recursion
			if (depth > 24) return direct;
			
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

			// Apply Russian Roulette Starting at the ray depth 4
			// Russian Roulette should kick in normally between at depth 3 to 5
			if (depth > 3) {
				float rrp = std::min(std::max(EPSILON, pathThroughput.Lum()), 1.f);
				if (sampler->next() < rrp) pathThroughput = pathThroughput / rrp;
				else return direct;
			}
			
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
	#if ENABLE_MULTITHREAD
	void render() {
		// Increment SPP and define Atomic ID Counter to battle race conditions
		film->incrementSPP();
		std::atomic<int> id = 0;

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
								Colour col = pathTrace(ray, &samplers[i]);

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
	}
	#else
	void render() {
		film->incrementSPP();
		for (unsigned int y = 0; y < film->height; y++) {
			for (unsigned int x = 0; x < film->width; x++) {
				// Sample a point in the pixel
				float px = x + samplers->next();  // + 0.5f
				float py = y + samplers->next();  // + 0.5f
				Ray ray = scene->camera.generateRay(px, py);

				// View Barycentrics / Shading Normals / Albedo / Direct Lighting / Path Trace
				//Colour col = viewBarycentrics(ray);
				//Colour col = viewNormals(ray);
				//Colour col = albedo(ray);
				//Colour col = direct(ray, samplers);
				Colour col = pathTrace(ray, samplers);

				// Check for NaN/Inf values
				if (std::isnan(col.r) || std::isnan(col.g) || std::isnan(col.b) ||
					std::isinf(col.r) || std::isinf(col.g) || std::isinf(col.b)) {
					continue;
				}

				film->splat(px, py, col);
				unsigned char r, g, b;
				film->tonemap(x, y, r, g, b);
				canvas->draw(x, y, r, g, b);
			}
		}
	}
	#endif

	int getSPP() { return film->SPP; }
	void saveHDR(std::string filename) { film->save(filename); }
	void savePNG(std::string filename) { stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3); }
};