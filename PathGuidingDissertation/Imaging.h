#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <string.h>

#include "Geometry.h"
#include "MyMath.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ThirdParty/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __STDC_LIB_EXT1__
#include "ThirdParty/stb/stb_image_write.h"

// Stop warnings about buffer overruns if size is zero. Size should never be zero and if it is the code handles it.
#pragma warning(disable : 6386)

constexpr float texelScale = 1.f / 255.f;

class Texture {
public:
	// Attributes
	Colour* texels;
	float* alpha;
	int width, height, channels;

	// Destructor
	~Texture() {
		delete[] texels;
		if (alpha != nullptr) delete alpha;
	}

	// Methods
	void loadDefault() {
		width = 1;
		height = 1;
		channels = 3;
		texels = new Colour[1];
		texels[0] = Colour(1.f, 1.f, 1.f);
	}

	void load(std::string filename) {
		alpha = nullptr;
		if (filename.find(".hdr") != std::string::npos) {
			float* textureData = stbi_loadf(filename.c_str(), &width, &height, &channels, 0);
			if (width == 0 || height == 0) {
				loadDefault();
				return;
			}

			texels = new Colour[width * height];
			for (int i = 0; i < (width * height); i++) {
				texels[i] = Colour(textureData[i * channels], textureData[(i * channels) + 1], textureData[(i * channels) + 2]);
			}
			stbi_image_free(textureData);
			return;
		}

		unsigned char* textureData = stbi_load(filename.c_str(), &width, &height, &channels, 0);
		if (width == 0 || height == 0) {
			loadDefault();
			return;
		}

		texels = new Colour[width * height];
		for (int i = 0; i < (width * height); i++) {
			texels[i] = Colour(textureData[i * channels] * texelScale, textureData[(i * channels) + 1] * texelScale, textureData[(i * channels) + 2] * texelScale);
		}

		if (channels == 4) {
			alpha = new float[width * height];
			for (int i = 0; i < (width * height); i++) {
				alpha[i] = textureData[(i * channels) + 3] * texelScale;
			}
		}
		stbi_image_free(textureData);
	}

	Colour sample(const float tu, const float tv) const {
		Colour tex;
		float u = std::max(0.f, fabsf(tu)) * width;
		float v = std::max(0.f, fabsf(tv)) * height;

		int x = static_cast<int>(floorf(u));
		int y = static_cast<int>(floorf(v));
		
		float frac_u = u - x;
		float frac_v = v - y;
		
		float w0 = (1.f - frac_u) * (1.f - frac_v);
		float w1 = frac_u * (1.f - frac_v);
		float w2 = (1.f - frac_u) * frac_v;
		float w3 = frac_u * frac_v;
		
		x = x % width;
		y = y % height;
		
		Colour s[4];
		s[0] = texels[y * width + x];
		s[1] = texels[y * width + ((x + 1) % width)];
		s[2] = texels[((y + 1) % height) * width + x];
		s[3] = texels[((y + 1) % height) * width + ((x + 1) % width)];
		tex = (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
		return tex;
	}

	float sampleAlpha(const float tu, const float tv) const {
		if (alpha == nullptr) return 1.f;

		float tex;
		float u = std::max(0.f, fabsf(tu)) * width;
		float v = std::max(0.f, fabsf(tv)) * height;

		int x = static_cast<int>(floorf(u));
		int y = static_cast<int>(floorf(v));

		float frac_u = u - x;
		float frac_v = v - y;

		float w0 = (1.f - frac_u) * (1.f - frac_v);
		float w1 = frac_u * (1.f - frac_v);
		float w2 = (1.f - frac_u) * frac_v;
		float w3 = frac_u * frac_v;

		x = x % width;
		y = y % height;

		float s[4]{};
		s[0] = alpha[y * width + x];
		s[1] = alpha[y * width + ((x + 1) % width)];
		s[2] = alpha[((y + 1) % height) * width + x];
		s[3] = alpha[((y + 1) % height) * width + ((x + 1) % width)];
		tex = (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
		return tex;
	}
};

class ImageFilter {
public:
	virtual float filter(const float x, const float y) const = 0;
	virtual int size() const = 0;
};

class BoxFilter : public ImageFilter {
public:
	float filter(float x, float y) const { return (fabs(x) < 0.5f && fabs(y) < 0.5f) ? 1.f : 0.f; }
	int size() const { return 1; }
};

class MitchellNetravaliFilter : public ImageFilter {
private:
	float B = 1.f / 3.f;
	float C = 1.f / 3.f;

	float separable(float val) const {
		// Store absolute value in a variable rather than calling it multiple times
		// Also store the square and cube powers in a variable too
		float x = std::fabs(val);
		float x2 = x * x;
		float x3 = x * x * x;

		if (x < 1.f) {
			return (1.f / 6.f) * ((12.f - 9.f * B - 6.f * C) * x3 + (-18.f + 12.f * B + 6.f * C) * x2 + (6.f - 2.f * B));
		}
		else if (x < 2.f) {
			return (1.f / 6.f) * ((-B - 6.f * C) * x3 + (6.f * B + 30.f * C) * x2 + (-12.f * B - 48.f * C) * x + (8.f * B + 24.f * C));
		}
		else return 0.f;
	}
public:
	float filter(float x, float y) const { return separable(x) * separable(y); }
	int size() const { return 2; }
};

class Film {
public:
	Colour* film;
	unsigned int width;
	unsigned int height;
	int SPP;
	ImageFilter* filter;

	void splat(const float x, const float y, const Colour& L) {
		// Code to splat a smaple with colour L into the image plane using an ImageFilter		
		int size = filter->size();
		int maxElements = static_cast<int>(pow(2 * size + 1, 2));
		
		unsigned int used = 0;
		float total = 0.f;

		std::vector<float>filterWeights(maxElements);   // Storage to cache weights
		std::vector<unsigned int>indices(maxElements);  // Store indices to minimize computations
		
		for (int i = -size; i <= size; i++) {
			for (int j = -size; j <= size; j++) {
				int px = static_cast<int>(std::floor(x + 0.5f)) + j;
				int py = static_cast<int>(std::floor(y + 0.5f)) + i;

				if (px >= 0 && px < width && py >= 0 && py < height) {
					indices[used] = (py * width) + px;
					filterWeights[used] = filter->filter(px - x, py - y);
					total += filterWeights[used];
					used++;
				}
			}
		}

		// Check for division by zero
		if (used > 0 && fabs(total) > 1e-4f) {
			for (int i = 0; i < used; i++) {
				film[indices[i]] = film[indices[i]] + (L * filterWeights[i] / total);
			}
		}
	}


	// Return a tonemapped pixel at coordinates x, y
	void tonemap(int x, int y, unsigned char& r, unsigned char& g, unsigned char& b, float exposure = 1.f, float LmaxWhite = 10.f) {
		// Get input colour and apply exposure
		Colour col = (SPP > 0) ? film[(y * width) + x] / static_cast<float>(SPP) : film[(y * width) + x];
		col = col * exposure;

		// Get Input Luminance
		float Lin = col.Lum();
		if (Lin < 1e-4f) { r = 0.f; g = 0.f; b = 0.f; return; }

		// Define Extended Reinhard Tonemapper and Gamma Correction
		float LmaxWhiteSq = LmaxWhite * LmaxWhite;
		float Lout = Lin * (1.f + (Lin / LmaxWhiteSq)) / (1.f + Lin);
		float extendedReinhardCoefficient = Lout / Lin;
		float gammaCorrectionExponent = 1.f / 2.2f;

		// Apply Extended Reinhard Tonemapper and Gamma Correction
		col.r = pow(col.r * extendedReinhardCoefficient, gammaCorrectionExponent);
		col.g = pow(col.g * extendedReinhardCoefficient, gammaCorrectionExponent);
		col.b = pow(col.b * extendedReinhardCoefficient, gammaCorrectionExponent);

		// Save RGB Channels
		r = static_cast<unsigned char>(std::max(0.f, std::min(col.r, 1.f)) * 255.f);
		g = static_cast<unsigned char>(std::max(0.f, std::min(col.g, 1.f)) * 255.f);
		b = static_cast<unsigned char>(std::max(0.f, std::min(col.b, 1.f)) * 255.f);
	}

	void init(int _width, int _height, ImageFilter* _filter) {
		width = _width;
		height = _height;
		film = new Colour[width * height];
		clear();
		filter = _filter;
	}

	void clear() {
		memset(film, 0, width * height * sizeof(Colour));
		SPP = 0;
	}

	void incrementSPP() {
		SPP++;
	}

	void save(std::string filename) {
		Colour* hdrpixels = new Colour[width * height];
		float invSPP = (SPP > 0) ? 1.f / static_cast<float>(SPP) : 1.f;

		for (unsigned int i = 0; i < (width * height); i++) {
			hdrpixels[i] = film[i] * invSPP;
		}
		stbi_write_hdr(filename.c_str(), width, height, 3, (float*)hdrpixels);
		delete[] hdrpixels;
	}
};