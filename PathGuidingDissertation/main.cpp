#define NOMINMAX

#include <unordered_map>

#include "Renderer.h"
#include "SceneLoader.h"

#include "ThirdParty/GamesEngineering/GEMLoader.h"
#include "ThirdParty/GamesEngineering/GamesEngineeringBase.h"

int main(int argc, char* argv[]) {
	// -- Area Light Test Scenes --
	//std::string sceneName = "../Scenes/bathroom";
	//std::string sceneName = "../Scenes/bathroom2";
	//std::string sceneName = "../Scenes/bedroom";
	//std::string sceneName = "../Scenes/coffee";
	//std::string sceneName = "../Scenes/cornell-box";
	//std::string sceneName = "../Scenes/glass-of-water";
	//std::string sceneName = "../Scenes/kitchen";
	//std::string sceneName = "../Scenes/living-room-2";
	//std::string sceneName = "../Scenes/living-room-3";
	//std::string sceneName = "../Scenes/staircase";
	std::string sceneName = "../Scenes/staircase2";
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
	unsigned int SPP = 16;       // Test Render SPP
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
		std::cout << t << std::endl;
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