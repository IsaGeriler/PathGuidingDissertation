#pragma once

#include <iostream>
#include <vector>

struct Metrics {
public:
	float calculateMSE(std::vector<float>& testImage, std::vector<float>& groundTruthImage, int width, int height) {
		// Check if the both image vectors are same length
		if (testImage.size() != groundTruthImage.size()) {
			std::cerr << "ERROR: Test image, and ground truth image do not match!";
			return -1.f;
		}

		float sumSquaredError = 0.f;
		int iterSize = width * height * 3;  // or 4, if alpha channel is also needed... i don't know...

		// MSE formulae = 1/N Ʃ(y - ŷ)²
		for (int i = 0; i < iterSize; i++) {
			float y = testImage[i];
			float y_hat = groundTruthImage[i];
			float diff = y - y_hat;
			sumSquaredError += diff * diff;
		}
		return sumSquaredError / (float)iterSize;
	}

	float calculateRMSE(std::vector<float>& testImage, std::vector<float>& groundTruthImage, int width, int height) {
		// Check if the both image vectors are same length
		if (testImage.size() != groundTruthImage.size()) {
			std::cerr << "ERROR: Test image, and ground truth image do not match!";
			return -1.f;
		}

		float sumSquaredError = 0.f;
		int iterSize = width * height * 3;  // or 4, if alpha channel is also needed... i don't know...

		// RMSE formulae = √(1/N Ʃ(y - ŷ)²)
		for (int i = 0; i < iterSize; i++) {
			float y = testImage[i];
			float y_hat = groundTruthImage[i];
			float diff = y - y_hat;
			sumSquaredError += diff * diff;
		}
		return std::sqrtf(sumSquaredError / (float)iterSize);
	}

	float calculateRRMSE(std::vector<float>& testImage, std::vector<float>& groundTruthImage, int width, int height) {
		// Check if the both image vectors are same length
		if (testImage.size() != groundTruthImage.size()) {
			std::cerr << "ERROR: Test image, and ground truth image do not match!";
			return -1.f;
		}

		float sumSquaredError = 0.f;
		int iterSize = width * height * 3;  // or 4, if alpha channel is also needed... i don't know...
		float y_bar = 0.f;

		// RRMSE formulae = √(1/N Ʃ(y - ŷ)²) / ȳ
		for (int i = 0; i < iterSize; i++) {
			float y = testImage[i];
			float y_hat = groundTruthImage[i];
			float diff = y - y_hat;
			y_bar += y_hat * y_hat;
			sumSquaredError += diff * diff;
		}
		float rmse = std::sqrtf(sumSquaredError / (float)iterSize);
		return rmse / sqrtf(y_bar);
	}
};