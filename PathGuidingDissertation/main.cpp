#define NOMINMAX

#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

#include "QTree.h"
#include "Renderer.h"
#include "SceneLoader.h"

#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

// --- BSDF Inversion Tests ---
static void runInversionTestLoop(BSDF* testBSDF, const std::string& testName) {
	// Define the constants
	const int NUMBER_OF_TESTS = 100000;
	const float TEST_EPSILON = 1e-4f;

	// Create both uniform and guided sampler
	MTRandom sampler;
	GuidedPathSampler testSampler;

	// Keep track of how many tests have we successfully passed
	int passedTestCount = 0;

	// Run tests
	for (int i = 0; i < NUMBER_OF_TESTS; i++) {
		// Randomly sample wo for test cases
		float r1 = sampler.next();
		float r2 = sampler.next();

		// Cosine hemisphere sampling
		Vec4 woLocal = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		// Set up a dummy shading data
		ShadingData testShadingData;
		testShadingData.sNormal = Vec4(0.f, 0.f, 1.f);
		testShadingData.gNormal = Vec4(0.f, 0.f, 1.f);
		testShadingData.bsdf = testBSDF;
		testShadingData.frame.fromVector(testShadingData.sNormal);
		testShadingData.wo = testShadingData.frame.toWorld(woLocal);

		// Generate random u, v, u_lobe
		float u_in = sampler.next(), v_in = sampler.next(), u_lobe = sampler.next();
		float pdf = 0.f;
		Colour col(0.f, 0.f, 0.f);
		testSampler.set(u_in, v_in, u_lobe);

		// Sample BSDF
		Vec4 wi = testShadingData.bsdf->sample(testShadingData, &testSampler, col, pdf);

		// Degenerate case - Invalid PDF
		// e.g. ConductorBSDF returns 0 if it tries to sample non-visible normals
		if (pdf <= 0.f) continue;

		// Invert BSDF
		float u_out = 0.f, v_out = 0.f, sampleProbability = 0.f;
		testShadingData.bsdf->invert(testShadingData, &sampler, wi, u_out, v_out, sampleProbability);
		testSampler.set(u_out, v_out, sampleProbability);

		float pdfReconstructed = 0.f;
		Colour colourReconstructed(0.f, 0.f, 0.f);
		Vec4 wiReconstructed = testShadingData.bsdf->sample(testShadingData, &testSampler, colourReconstructed, pdfReconstructed);

		float dotDifference = 1.f - Dot(wi.normalize(), wiReconstructed.normalize());
		if (dotDifference > TEST_EPSILON || std::isnan(dotDifference)) {
			std::cerr << "\n[FAILED] " << testName << std::endl;
			std::cerr << "INPUT UV: u=" << u_in << ", v=" << v_in << std::endl;
			std::cerr << "OUTPUT UV: u=" << u_out << ", v=" << v_out << std::endl;
			std::cerr << "ORIGINAL WI: wi=<" << wi.x << ", " << wi.y << ", " << wi.z << ">" << std::endl;
			std::cerr << "RECONSTRUCTED WI: wi=<" << wiReconstructed.x << ", " << wiReconstructed.y << ", " << wiReconstructed.z << ">" << std::endl;
			assert(false && "BSDF Inversion Test Failed!");
		}
		passedTestCount++;
	}
	std::cout << "[PASSED] " << testName << " (" << passedTestCount << " valid samples)" << std::endl;
}

static void testDiffuseBSDF(Texture* tex) {
	BSDF* bsdf = new DiffuseBSDF(tex);
	std::string testName = "Diffuse";
	runInversionTestLoop(bsdf, testName);
	delete bsdf;
}

static void testOrenNayarBSDF(Texture* tex) {
	float roughnessList[] = { 0.1f, 0.5f, 0.9f };
	std::string testName = "";

	for (float roughness : roughnessList) {
		BSDF* bsdf = new OrenNayarBSDF(tex, roughness);
		testName = "Oren-Nayar (Roughness: " + std::to_string(roughness) + ")";
		runInversionTestLoop(bsdf, testName);
		delete bsdf;
	}
}

static void testConductorBSDF(Texture* tex) {
	// Gold eta and k estimations
	Colour etaGold(0.14f, 0.37f, 1.44f);
	Colour kGold(3.18f, 2.61f, 1.9f);

	// Copper eta and k estimations
	Colour etaCopper(0.24f, 0.93f, 1.1f);
	Colour kCopper(3.11f, 2.61f, 2.42f);

	float roughnessList[] = { 0.1f, 0.6f };
	std::string testName = "";

	for (float roughness : roughnessList) {
		// Gold ConductorBSDF
		BSDF* goldBSDF = new ConductorBSDF(tex, etaGold, kGold, roughness);
		testName = "Conductor - Gold   (Roughness: " + std::to_string(roughness) + ")";
		runInversionTestLoop(goldBSDF, testName);
		delete goldBSDF;

		// Copper ConductorBSDF
		BSDF* copperBSDF = new ConductorBSDF(tex, etaCopper, kCopper, roughness);
		testName = "Conductor - Copper (Roughness: " + std::to_string(roughness) + ")";
		runInversionTestLoop(copperBSDF, testName);
		delete copperBSDF;
	}
}

static void testDielectricBSDF(Texture* tex) {
	float extIOR = 1.f;
	float intIORList[] = { 1.33f, 1.5f };
	float roughnessList[] = { 0.05f, 0.4f, 0.8f };
	std::string testName = "";

	for (float intIOR : intIORList) {
		for (float roughness : roughnessList) {
			BSDF* bsdf = new DielectricBSDF(tex, intIOR, extIOR, roughness);
			testName = "Dielectric (IOR: " + std::to_string(intIOR) + ", Roughness: " + std::to_string(roughness) + ")";
			runInversionTestLoop(bsdf, testName);
			delete bsdf;
		}
	}
}

static void testPlasticBSDF(Texture* tex) {
	float extIOR = 1.f;
	float intIORList[] = { 1.33f, 1.5f };
	float roughnessList[] = { 0.05f, 0.4f, 0.8f };
	std::string testName = "";

	for (float intIOR : intIORList) {
		for (float roughness : roughnessList) {
			BSDF* bsdf = new PlasticBSDF(tex, intIOR, extIOR, roughness);
			testName = "Plastic (IOR: " + std::to_string(intIOR) + ", Roughness: " + std::to_string(roughness) + ")";
			runInversionTestLoop(bsdf, testName);
			delete bsdf;
		}
	}
}

static void testGlassBSDF(Texture* tex) {
	float extIOR = 1.f;
	float intIORList[] = { 1.33f, 1.5f };
	std::string testName = "";

	for (float intIOR : intIORList) {
		BSDF* bsdf = new GlassBSDF(tex, intIOR, extIOR);
		testName = "Glass (IOR: " + std::to_string(intIOR) + ")";
		runInversionTestLoop(bsdf, testName);
		delete bsdf;
	}
}

static void testMirrorBSDF(Texture* tex) {
	BSDF* bsdf = new MirrorBSDF(tex);
	std::string testName = "Mirror";
	runInversionTestLoop(bsdf, testName);
	delete bsdf;
}

static void runAllBSDFInversionTests() {
	std::cout << "========================================" << std::endl;
	std::cout << "       START BSDF INVERSION TESTS!      " << std::endl;
	std::cout << "========================================" << std::endl;

	Texture* dummyTexture = new Texture();
	dummyTexture->loadDefault();

	// Call BSDF Inversions
	testDiffuseBSDF(dummyTexture);
	testOrenNayarBSDF(dummyTexture);
	testConductorBSDF(dummyTexture);
	testDielectricBSDF(dummyTexture);
	testPlasticBSDF(dummyTexture);
	testGlassBSDF(dummyTexture);
	testMirrorBSDF(dummyTexture);
	// testLayeredBSDF(dummyTexture);
	delete dummyTexture;

	std::cout << "========================================" << std::endl;
	std::cout << "    ALL BSDF INVERSION TESTS PASSED!    " << std::endl;
	std::cout << "========================================" << std::endl;
}
// --- BSDF Inversion Tests End ---

// --- QTree Tests ---
static void testPDFIntegration() {
	// Assuming maxDepth is 4 for this test
	QTree qtree(4);

	// Populate the QTree with sample data
	qtree.insert(0.1f, 0.1f, 100.f);
	qtree.insert(0.9f, 0.9f, 50.f);
	qtree.insert(0.5f, 0.1f, 10.f);

	float integral = 0.f;

	// maxDepth 4, the grid is 16 cells accross
	int numCells = 16;
	float cellSize = 1.f / numCells;
	float cellArea = cellSize * cellSize;

	// Iterate over every leaf's centre
	for (int i = 0; i < numCells; ++i) {
		for (int j = 0; j < numCells; ++j) {
			// Get the centre of the cell and then do PDF * dx * dy
			float u = (i + 0.5f) * cellSize;
			float v = (j + 0.5f) * cellSize;
			integral += qtree.pdf(u, v) * cellArea;
		}
	}
	std::cout << "QTree PDF Integration Test: Integral = " << integral << std::endl;
	assert(std::fabs(integral - 1.f) < 1e-5f && "PDF Integration Test Failed...");
}

static void testSamplePDFConsistency() {
	// Populate the QTree with sample data
	QTree qtree(3);
	qtree.insert(0.2f, 0.2f, 10.f);

	// Deterministic test values
	float r1 = 0.123f;
	float r2 = 0.456f;
	float u, v, pdfSampled, pdfEvaluated;

	// Sample from the QTree
	qtree.sample(r1, r2, u, v, pdfSampled);
	pdfEvaluated = qtree.pdf(u, v);

	std::cout << "Sampled PDF: " << pdfSampled << ", Evaluated PDF: " << pdfEvaluated << std::endl;
	assert(std::fabs(pdfSampled - pdfEvaluated) < 1e-4f && "Sample/PDF Consistency Test Failed...");
}

static void testNullSpace() {
	// Populate the QTree with sample data
	QTree qtree(2);
	qtree.insert(0.1f, 0.1f, 10.f);

	// Evaluate top-right corner
	float pdfEvaluated = qtree.pdf(0.9f, 0.9f);
	std::cout << "Evaluated PDF at (0.9, 0.9): " << pdfEvaluated << std::endl;
	assert(pdfEvaluated == 0.f && "Null Space Test Failed...");
}

static void testBoundariesAndExtremes() {
	// Populate the QTree with sample data
	QTree qtree(4);
	qtree.insert(0.f, 0.f, 10.f);
	qtree.insert(0.9999f, 0.9999f, 10.f);
	qtree.insert(1.f, 1.f, 10.f);
	qtree.insert(0.5f, 0.5f, 0.f);
	qtree.insert(0.5f, 0.5f, -5.f);

	// Sample with extreme cases
	float u, v, pdf;
	qtree.sample(0.f, 0.f, u, v, pdf);
	assert(!std::isnan(u) && !std::isnan(v) && !std::isnan(pdf));
	std::cout << "u: " << u << ", v: " << v << ", pdf=" << pdf << std::endl;

	qtree.sample(0.99999f, 0.99999f, u, v, pdf);
	assert(!std::isnan(u) && !std::isnan(v) && !std::isnan(pdf));
	std::cout << "u: " << u << ", v: " << v << ", pdf=" << pdf << std::endl;

	// Test PDF at extreme cases
	float pdfLow = qtree.pdf(0.f, 0.f);
	float pdfHigh = qtree.pdf(0.9999f, 0.9999f);
	std::cout << "pdfLow=" << pdfLow << ", pdfHigh=" << pdfHigh << std::endl;
	assert(!std::isnan(pdfLow) && !std::isnan(pdfHigh));
}

static void testMemoryAllocationStress() {
	// Initialize a deep tree (2^8 = 65536 leaves)
	QTree qtree(8);
	MTRandom sampler;

	// Insert 100000 random samples
	for (int i = 0; i < 100000; i++) {
		float u = sampler.next();
		float v = sampler.next();
		float luminance = (float)(rand() % 100) / 10.f;
		qtree.insert(u, v, luminance);
	}
	float uOut, vOut, pdfOut;
	qtree.sample(0.5f, 0.5f, uOut, vOut, pdfOut);
	std::cout << "Memory Stress Test Has Been Successfull..." << std::endl;
	assert(pdfOut > 0.f && "Memory Stress Test Failed");
}

static void testStatisticalDistribution() {
	// Initialize a shallow tree
	QTree qtree(2);
	qtree.insert(0.25f, 0.25f, 1.f);
	qtree.insert(0.75f, 0.25f, 2.f);
	qtree.insert(0.25f, 0.75f, 3.f);
	qtree.insert(0.75f, 0.75f, 4.f);

	MTRandom sampler;
	int counts[4]{};
	int numberOfSamples = 100000;

	// Insert 100000 random samples
	for (int i = 0; i < 100000; i++) {
		float r1 = sampler.next();
		float r2 = sampler.next();
		float u, v, pdf;
		qtree.sample(r1, r2, u, v, pdf);

		// Determine which quadrant the sample fell into
		if (u < 0.5f && v < 0.5f) counts[0]++;
		else if (u >= 0.5f && v < 0.5f) counts[1]++;
		else if (u < 0.5f && v >= 0.5f) counts[2]++;
		else if (u >= 0.5f && v >= 0.5f) counts[3]++;
	}

	// Check if the empirical distribution matches the expected distribution
	float tolerance = 0.015f;  // 1.5% tolerance
	assert(std::fabs((counts[0] / (float)numberOfSamples) - 0.1f) < tolerance && "Statistical Distribution Test Failed...");
	assert(std::fabs((counts[1] / (float)numberOfSamples) - 0.2f) < tolerance && "Statistical Distribution Test Failed...");
	assert(std::fabs((counts[2] / (float)numberOfSamples) - 0.3f) < tolerance && "Statistical Distribution Test Failed...");
	assert(std::fabs((counts[3] / (float)numberOfSamples) - 0.4f) < tolerance && "Statistical Distribution Test Failed...");
	std::cout << "Statistical Distribution Test Passed..." << std::endl;
}

static void testClearAndReuse() {
	QTree qtree(3);
	qtree.insert(0.1f, 0.1f, 100.f);
	float initialPdf = qtree.pdf(0.1f, 0.1f);
	assert(initialPdf > 1.0f && "Tree should have concentrated PDF.");

	// Clear the tree
	qtree.clear();

	// Verify fallback to uniform behavior when empty
	float emptyPdf = qtree.pdf(0.5f, 0.5f);
	assert((emptyPdf == 1.f) && "Cleared tree should act uniform.");

	// Insert new data elsewhere and check
	qtree.insert(0.9f, 0.9f, 100.f);
	float newPdfOldSpot = qtree.pdf(0.1f, 0.1f);
	assert((newPdfOldSpot == 0.f) && "Old data should be completely gone.");
	float newPdfNewSpot = qtree.pdf(0.9f, 0.9f);
	assert((newPdfNewSpot > 1.f) && "New data should dictate the PDF.");
	std::cout << "Clear and Reuse Test Passed..." << std::endl;
}

static void testEmptyTree() {
	// Testing PDF on an empty tree
	QTree qtree(3);
	assert(qtree.pdf(0.2f, 0.3f) == 1.f);
	assert(qtree.pdf(0.8f, 0.9f) == 1.f);

	// Test Sampling
	float u, v, pdf;
	qtree.sample(0.123f, 0.456f, u, v, pdf);
	assert(u == 0.123f && v == 0.456f && "Empty tree should pass r1/r2 straight to u/v.");
	assert(pdf == 1.f && "Empty tree PDF should be 1.");
	std::cout << "Empty Tree Test Passed..." << std::endl;
}

static void testExactMidpoint() {
	// Initialize QTree and insert a point in the middle
	QTree qtree(4);
	qtree.insert(0.5f, 0.5f, 10.f);
	float pdfBottomLeft = qtree.pdf(0.49f, 0.49f);
	float pdfTopRight = qtree.pdf(0.51f, 0.51f);
	assert(pdfBottomLeft == 0.0f && "Exact midpoint should not leak to bottom-left.");
	assert(pdfTopRight > 0.0f && "Exact midpoint should evaluate in top-right.");
	std::cout << "Exact Midpoint Boundary Test Passed..." << std::endl;
}

static void testNaNInFCatch() {
	QTree qtree(3);
	float nanVal = std::numeric_limits<float>::quiet_NaN();
	float infVal = std::numeric_limits<float>::infinity();

	qtree.insert(0.5f, 0.5f, nanVal);
	qtree.insert(0.2f, 0.2f, -nanVal);
	qtree.insert(0.8f, 0.8f, infVal);
	qtree.insert(0.3f, 0.3f, -infVal);

	float pdfEvaluated = qtree.pdf(0.5f, 0.5f);
	assert(pdfEvaluated == 1.f && "Tree should remain uniform after rejecting NaN and InF.");
	std::cout << "NaN and InF Resilience Test Passed..." << std::endl;
}

static void runAllQTreeTests() {
	std::cout << "\n========================================" << std::endl;
	std::cout << "            START QTREE TESTS!          " << std::endl;
	std::cout << "========================================" << std::endl;

	// Call QTree Tests
	testSamplePDFConsistency();
	testPDFIntegration();
	testNullSpace();
	testBoundariesAndExtremes();
	testMemoryAllocationStress();
	testStatisticalDistribution();
	testClearAndReuse();
	testEmptyTree();
	testExactMidpoint();
	testNaNInFCatch();

	std::cout << "========================================" << std::endl;
	std::cout << "         ALL QTREE TESTS PASSED!        " << std::endl;
	std::cout << "========================================" << std::endl;
}
// --- QTree Tests End ---

// Run all the tests here
static void runTest() {
	// BSDF Inversion Test
	runAllBSDFInversionTests();

	// QTree Test
	runAllQTreeTests();
}

int main(int argc, char* argv[]) {
	// Run testing code first before rendering any stuff!
	runTest();

	// Note - Test on these scenes
	// Cornell Box +, Kitchen +, Bathroom +, Bathroom2 +, Staircase +
	// Classroom +, Sibenik, Dining Room +

	// -- Area Light Test Scenes --
	//std::string sceneName = "../Scenes/bathroom";
	//std::string sceneName = "../Scenes/bathroom2";
	//std::string sceneName = "../Scenes/bedroom";
	//std::string sceneName = "../Scenes/coffee";
	//std::string sceneName = "../Scenes/cornell-box";
	//std::string sceneName = "../Scenes/glass-of-water";
	std::string sceneName = "../Scenes/kitchen";
	//std::string sceneName = "../Scenes/living-room-2";
	//std::string sceneName = "../Scenes/living-room-3";
	//std::string sceneName = "../Scenes/staircase";
	//std::string sceneName = "../Scenes/staircase2";
	//std::string sceneName = "../Scenes/veach-bidir";
	//std::string sceneName = "../Scenes/veach-mis";
	
	// -- Environment Map Test Scenes --
	//std::string sceneName = "../Scenes/classroom";
	//std::string sceneName = "../Scenes/car2";
	//std::string sceneName = "../Scenes/dining-room";
	//std::string sceneName = "../Scenes/house";
	//std::string sceneName = "../Scenes/living-room";
	//std::string sceneName = "../Scenes/materialball";
	//std::string sceneName = "../Scenes/MaterialsScene";
	//std::string sceneName = "../Scenes/Sibenik";
	//std::string sceneName = "../Scenes/Terrain";

	// -- I don't know why it does not work Scene
	//std::string sceneName = "../Scenes/teapot-full";

	// Ground Truth: 8192/16384					(should really be an absurd number to eliminate variance)
	// Testing SPPs: 128/256/512/1024/2048/4096 (render, and compare error metrics with the groung truth)
	//unsigned int SPP = 8192;
	unsigned int SPP = 128;
	std::string filename = "GI.hdr";
	//std::string method = "path_trace";
	//std::string method = "photon_map";
	//std::string method = "path_guide_photon";
	std::string method = "path_guide_pss";
	double timeLimitSeconds = -1.0;  // Seconds!

	if (argc > 1) {
		std::unordered_map<std::string, std::string> args;
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (!arg.empty() && arg[0] == '-') {
				std::string argName = arg;
				if (i + 1 < argc) {
					std::string argValue = argv[++i];
					args[argName] = argValue;
				} else {
					std::cerr << "Error: Missing value for argument '" << arg << "'\n";
				}
			} else {
				std::cerr << "Warning: Ignoring unexpected argument '" << arg << "'\n";
			}
		}
		
		for (const auto& pair : args) {
			if (pair.first == "-scene") {
				sceneName = pair.second;
			}
			else if (pair.first == "-outputFilename") {
				filename = pair.second;
			}
			else if (pair.first == "-SPP") {
				SPP = stoi(pair.second);
			}
			else if (pair.first == "-method") {
				method = pair.second;
			}
			else if (pair.first == "-timeLimitSecond") {
				timeLimitSeconds = stod(pair.second);
			}
		}
	}

	Scene* scene = loadScene(sceneName);
	GamesEngineeringBase::Window canvas;
	canvas.create((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, "Tracer", false);
	RayTracer rt;
	rt.init(scene, &canvas, method, SPP);
	rt.sceneName = sceneName;
	bool running = true;
	GamesEngineeringBase::Timer timer;

	// Start the timer before the loop
	auto startTime = std::chrono::high_resolution_clock::now();

	while (running) {
		canvas.checkInput();
		canvas.clear();

		if (canvas.keyPressed(VK_ESCAPE)) break;
		if (canvas.keyPressed('W')) {
			viewcamera.forward();
			rt.clear();
		}
		if (canvas.keyPressed('S')) {
			viewcamera.back();
			rt.clear();
		}
		if (canvas.keyPressed('A')) {
			viewcamera.left();
			rt.clear();
		}
		if (canvas.keyPressed('D')) {
			viewcamera.right();
			rt.clear();
		}
		if (canvas.keyPressed('E')) {
			viewcamera.flyUp();
			rt.clear();
		}
		if (canvas.keyPressed('Q')) {
			viewcamera.flyDown();
			rt.clear();
		}
		// Time how long a render call takes
		timer.reset();
		rt.render();
		float t = timer.dt();
		// Write
		std::cout << "SPP: " << rt.getSPP() << " | Render Time: " << t << " s / frame" << std::endl;
		if (canvas.keyPressed('P')) {
			rt.saveHDR(filename);
		}
		if (canvas.keyPressed('L')) {
			size_t pos = filename.find_last_of('.');
			std::string ldrFilename = filename.substr(0, pos) + ".png";
			rt.savePNG(ldrFilename);
		}

		// Equal SPP or Equal Time
		auto currentTime = std::chrono::high_resolution_clock::now();
		double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();

		bool sppReached = (rt.getSPP() >= SPP);
		bool timeReached = (timeLimitSeconds > 0.0 && elapsedSeconds >= timeLimitSeconds);
		
		if (sppReached || timeReached) {
			std::cout << "\n[Render Complete] Method: " << method
				<< " | Achieved SPP: " << SPP
				<< " | Time: " << elapsedSeconds << "s"
				<< " (Exit Reason: " << (timeReached ? "Time Limit" : "Target SPP") << " Reached)"
				<< std::endl;
			// Save both HDR and LDR (SDR) images
			size_t pos = filename.find_last_of('.');
			std::string ldrFilename = filename.substr(0, pos) + ".png";
			rt.saveHDR(filename);
			rt.savePNG(ldrFilename);
			break;
		}
		canvas.present();
	}
	return 0;
}