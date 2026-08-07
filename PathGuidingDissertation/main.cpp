#define NOMINMAX

#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>

#include "QTree.h"
#include "Renderer.h"
#include "SceneLoader.h"

#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

static void invertBSDFTest() {
	// Add test codes here
	// Testing BSDF Inversion (DiffuseBSDF & OrenNayarBSDF for now)
	std::cout << "--- BSDF Inversion Test ---" << std::endl;
	const int NUMBER_OF_TESTS = 100000;
	const float TEST_EPSILON = 1e-4f;

	Texture* testTexture = new Texture();
	testTexture->loadDefault();

	std::string testMaterial = "Diffuse";
	BSDF* testBSDF = new DiffuseBSDF(testTexture);

	//std::string testMaterial = "Mirror";
	//BSDF* testBSDF = new MirrorBSDF(testTexture);

	//std::string testMaterial = "Glass";
	//BSDF* testBSDF = new GlassBSDF(testTexture, 1.f, 1.5f);

	//std::string testMaterial = "Oren-Nayar";
	//BSDF* testBSDF = new OrenNayarBSDF(testTexture, 0.5f);

	//std::string testMaterial = "Conductor";
	//BSDF* testBSDF = new ConductorBSDF(testTexture, Colour(1.f, 0.6f, 1.5f), Colour(0.65f, 0.85f, 1.f), 0.5f);

	//std::string testMaterial = "Dielectric";
	//BSDF* testBSDF = new DielectricBSDF(testTexture, 1.f, 1.5f, 0.5f);

	//std::string testMaterial = "Plastic";
	//BSDF* testBSDF = new PlasticBSDF(testTexture, 1.f, 1.5f, 0.5f);

	ShadingData testShadingData;
	testShadingData.sNormal = Vec4(0.f, 0.f, 1.f);
	testShadingData.gNormal = Vec4(0.f, 0.f, 1.f);
	testShadingData.wo = Vec4(0.f, 1.f, 0.f);
	testShadingData.bsdf = testBSDF;
	testShadingData.frame.fromVector(testShadingData.sNormal);

	MTRandom sampler;
	GuidedPathSampler testSampler;

	for (int i = 0; i < NUMBER_OF_TESTS; i++) {
		// Sample BSDF
		float u_in = sampler.next(), v_in = sampler.next(), u_lobe = sampler.next();
		float pdf = 0.f;
		Colour col(0.f, 0.f, 0.f);
		testSampler.set(u_in, v_in, u_lobe);
		Vec4 wi = testShadingData.bsdf->sample(testShadingData, &testSampler, col, pdf);

		// Invert Sample BSDF
		float u_out = 0.f, v_out = 0.f, sampleProbability = 0.f;
		testShadingData.bsdf->invert(testShadingData, wi, u_out, v_out, sampleProbability);

		if (fabs(v_in - 0.f) < TEST_EPSILON && fabs(v_out - 1.f) < TEST_EPSILON) v_out = 0.f;
		if (fabs(v_out - 0.f) < TEST_EPSILON && fabs(v_in - 1.f) < TEST_EPSILON) v_in = 0.f;

		// Assertions for Debug Mode
		float u_diff = fabs(u_in - u_out);
		float v_diff = fabs(v_in - v_out);

		if (u_diff > TEST_EPSILON || v_diff > TEST_EPSILON) {
			std::cerr << "INVERSION TEST FAILED!" << std::endl;
			std::cerr << "INPUT UV: u=" << u_in << ", v=" << v_in << std::endl;
			std::cerr << "OUTPUT UV: u=" << u_out << ", v=" << v_out << std::endl;
			std::cerr << "DIFFERENCE UV: u=" << u_diff << ", v=" << v_diff << std::endl;
			std::cerr << "SAMPLED WI: wi=<" << wi.x << "," << wi.y << "," << wi.z << ">" << std::endl;
			assert(false && "BSDF Inversion Test Failed...");
		}
	}
	std::cout << "PASSED: " << NUMBER_OF_TESTS << "/" << NUMBER_OF_TESTS << " " << testMaterial << "BSDF INVERSION TESTS!" << std::endl;
	std::cout << "---------------------------" << std::endl;
}

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

void testMemoryAllocationStress() {
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

// Run all the tests here
static void runTest() {
	// BSDF Inversion Test
	invertBSDFTest();

	// QTree Tests
	testSamplePDFConsistency();
	testPDFIntegration();
	testNullSpace();
	testBoundariesAndExtremes();
	testMemoryAllocationStress();
}

int main(int argc, char* argv[]) {
	// Run testing code first before rendering any stuff!
	runTest();

	// Note - Test on these scenes
	// Cornell Box +, Kitchen +, Bathroom +, Bathroom2 +, Bedroom +, Living-Room 2+
	// Classroom, Sibenik, Dining Room, Living Room 1

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

	Scene* scene = loadScene(sceneName);
	GamesEngineeringBase::Window canvas;
	canvas.create((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, "Tracer", false);
	RayTracer rt;
	rt.init(scene, &canvas);
	bool running = true;
	GamesEngineeringBase::Timer timer;

	unsigned int SPP = rt.maxSPP;  // Adjust maximum SPP inside Renderer.h, this is modified to ease up Path Guiding (unless prompted in CMD)
	std::string filename = "GI.hdr";

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
			if (pair.first == "-outputFilename") {
				filename = pair.second;
			}
			if (pair.first == "-SPP") {
				SPP = stoi(pair.second);
			}
		}
	}

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
		std::cout << "Render Time Per Frame: " << t << "s" << std::endl;
		if (canvas.keyPressed('P')) {
			rt.saveHDR(filename);
		}
		if (canvas.keyPressed('L')) {
			size_t pos = filename.find_last_of('.');
			std::string ldrFilename = filename.substr(0, pos) + ".png";
			rt.savePNG(ldrFilename);
		}
		if (SPP == rt.getSPP()) {
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