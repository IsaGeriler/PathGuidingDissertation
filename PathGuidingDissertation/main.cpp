#define NOMINMAX

#include <cmath>
#include <iostream>
#include <unordered_map>

#include "Renderer.h"
#include "SceneLoader.h"

#include "ThirdParty/GamesEngineering/GEMLoader.h"
#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

static void runTest() {
	// Add test codes here
	// Testing BSDF Inversion (DiffuseBSDF & OrenNayarBSDF for now)
	std::cout << "--- BSDF Inversion Test ---" << std::endl;
	const int NUMBER_OF_TESTS = 100000;
	const float TEST_EPSILON = 1e-4f;

	Texture* testTexture = new Texture();
	testTexture->loadDefault();

	//std::string testMaterial = "Diffuse";
	//BSDF* testBSDF = new DiffuseBSDF(testTexture);

	std::string testMaterial = "Conductor";
	BSDF* testBSDF = new ConductorBSDF(testTexture, Colour(1.f, 0.6f, 1.5f), Colour(0.65f, 0.85f, 1.f), 0.5f);

	//std::string testMaterial = "Oren-Nayar";
    //BSDF* testBSDF = new OrenNayarBSDF(testTexture, 0.5f);

	ShadingData testShadingData;
	testShadingData.sNormal = Vec4(0.f, 1.f, 0.f);
	testShadingData.gNormal = Vec4(0.f, 1.f, 0.f);
	testShadingData.wo = Vec4(0.f, 1.f, 0.f);
	testShadingData.bsdf = testBSDF;
	testShadingData.frame.fromVector(testShadingData.sNormal);
	
	MTRandom sampler;
	TestSampler testSampler;

	for (int i = 0; i < NUMBER_OF_TESTS; i++) {
		// Sample BSDF
		float u_in = sampler.next(), v_in = sampler.next();
		float pdf = 0.f;
		Colour col(0.f, 0.f, 0.f);
		testSampler.set(u_in, v_in);
		Vec4 wi = testShadingData.bsdf->sample(testShadingData, &testSampler, col, pdf);

		// Invert Sample BSDF
		float u_out = 0.f, v_out = 0.f;
		testShadingData.bsdf->invert(testShadingData, wi, u_out, v_out);

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

int main(int argc, char* argv[]) {
	// Run testing code first before rendering any stuff!
	runTest();

	// -- Area Light Test Scenes --
	//std::string sceneName = "../Scenes/bathroom";
	//std::string sceneName = "../Scenes/bathroom2";
	//std::string sceneName = "../Scenes/bedroom";
	//std::string sceneName = "../Scenes/coffee";
	std::string sceneName = "../Scenes/cornell-box";
	//std::string sceneName = "../Scenes/glass-of-water";
	//std::string sceneName = "../Scenes/kitchen";
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
	//std::string sceneName = "../Scenes/teapot-full";
	//std::string sceneName = "../Scenes/Terrain";

	std::string filename = "GI.hdr";
	unsigned int SPP = 1;       // Test Render SPP (128 to balance between fast/quality renders)
	//unsigned int SPP = 4096;    // Ground Truth Render SPP

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

	Scene* scene = loadScene(sceneName);
	GamesEngineeringBase::Window canvas;
	canvas.create((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, "Tracer", false);
	RayTracer rt;
	rt.init(scene, &canvas);
	bool running = true;
	GamesEngineeringBase::Timer timer;

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