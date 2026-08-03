#pragma once

#include <iostream>

struct QTreeBox {
	// Attributes
	float minU; float maxU;
	float minV; float maxV;

	// Helper Area Method
	float area() const { return ((maxU - minU) * (maxV - minV)); }
};

// uv-coordinates
// Child Index 0: Bottom-Left  (0,0)
// Child Index 1: Bottom-Right (1,0)
// Child Index 2: Top-Left     (0,1)
// Child Index 3: Top-Right    (1,1)
struct QTreeNode {
	// Attributes
	float weight = 0.f;
	QTreeNode* children[4]{};

	// Helper isLeaf Method
	bool isLeaf() const {
		return (children[0] == nullptr) && (children[1] == nullptr) && (children[2] == nullptr) && (children[3] == nullptr);
	}
};

// --- Directional-Tree Component of the Path Guiding ---
class QTree {
private:
	// Attributes
	QTreeNode nodePool[512];
	QTreeNode* root;
	int nextFreeNode = 0;
	int maxDepth;

	// Private Methods
	QTreeNode* allocateNode() {
		// Pool exhausted, return null
		if (nextFreeNode >= 512) return nullptr;

		// Initialize and return the node
		QTreeNode* node = &nodePool[nextFreeNode++];
		node->weight = 0.f;
		node->children[0] = node->children[1] = node->children[2] = node->children[3] = nullptr;
		return node;
	}

	void insertRecursive(QTreeNode* node, float u, float v, float luminance, QTreeBox currentBox, int depth) {
		// Guard case for null node
		if (node == nullptr) return;

		// Add luminance to node->totalWeight
		node->weight += luminance;

		// Terminate the recursion when we reach the max depth of the tree
		if (depth == maxDepth) return;

		// Find quadrant index using bounding box midpoints
		// Calculate mid-ponts
		float midU = (currentBox.maxU + currentBox.minU) * 0.5f;
		float midV = (currentBox.maxV + currentBox.minV) * 0.5f;

		// Obtain index by comparing the points with the mid-points
		int index = 0;
		if (u >= midU) index += 1;
		if (v >= midV) index += 2;

		// If the child is null then create a new tree node
		if (node->children[index] == nullptr) node->children[index] = allocateNode();
		
		// Calculate the child's bounding box
		QTreeBox childrenBox{};
		childrenBox.minU = (index % 2 == 1) ? midU : currentBox.minU;
		childrenBox.maxU = (index % 2 == 1) ? currentBox.maxU : midU;
		childrenBox.minV = (index >= 2) ? midV : currentBox.minV;
		childrenBox.maxV = (index >= 2) ? currentBox.maxV : midV;
		
		// Recursive call
		insertRecursive(node->children[index], u, v, luminance, childrenBox, depth + 1);
	}

	void sampleRecursive(QTreeNode* node, float r1, float r2, QTreeBox currentBox, int depth, float& u, float& v, float& pdf, float currentPdf) {
		// Sample, and terminate the recursion when we are at the leaf node
		if (depth == maxDepth) {
			u = currentBox.minU + r1 * (currentBox.maxU - currentBox.minU);
			v = currentBox.minV + r2 * (currentBox.maxV - currentBox.minV);
			pdf = currentPdf;
			return;
		}
		// Calculate the total weight of the children nodes
		float sumWeights = 0.f;
		for (int i = 0; i < 4; i++) {
			if (node->children[i] != nullptr) {
				sumWeights += node->children[i]->weight;
			}
		}
		// Safety guard, treating as leaf node to avoid zero divisions
		if (sumWeights <= 0.f) {
			u = currentBox.minU + r1 * (currentBox.maxU - currentBox.minU);
			v = currentBox.minV + r2 * (currentBox.maxV - currentBox.minV);
			pdf = currentPdf;
			return;
		}
		// Find the chosen children
		float accumulatedProbability = 0.f;
		float probability = 0.f;
		int index = 0;

		for (int i = 0; i < 4; i++) {
			if (node->children[i] == nullptr) continue;
			float childProbability = node->children[i]->weight / sumWeights;
			if (r1 < accumulatedProbability + childProbability) {
				index = i;
				probability = childProbability;

				// Rescale r1 for reusability
				// Maybe rescale r2 as well..? I don't know...
				r1 = (r1 - accumulatedProbability) / probability;
				// r2 = (r2 - accumulatedProbability) / probability;
				break;
			}
			accumulatedProbability += childProbability;
		}
		// Calculate mid-ponts
		float midU = (currentBox.maxU + currentBox.minU) * 0.5f;
		float midV = (currentBox.maxV + currentBox.minV) * 0.5f;

		// Calculate the child's bounding box
		QTreeBox childrenBox{};
		childrenBox.minU = (index % 2 == 1) ? midU : currentBox.minU;
		childrenBox.maxU = (index % 2 == 1) ? currentBox.maxU : midU;
		childrenBox.minV = (index >= 2) ? midV : currentBox.minV;
		childrenBox.maxV = (index >= 2) ? currentBox.maxV : midV;

		// Recursive call
		// Update the currentPdf by multiplying with the probability of the chosen child and the area factor (4.f) since we are in a quadtree
		currentPdf *= probability * 4.f;
		sampleRecursive(node->children[index], r1, r2, childrenBox, depth + 1, u, v, pdf, currentPdf);
	}
	
	float pdfRecursive(QTreeNode* node, float u, float v, QTreeBox currentBox, int depth, float currentPdf) {
		// Handle null node case
		if (node == nullptr) return 0.f;

		// Terminate the recursion when we reach the max depth of the tree
		if (depth == maxDepth) return currentPdf;

		// Find quadrant index for (u,v) using bounding box midpoints
		// Calculate mid-ponts
		float midU = (currentBox.maxU + currentBox.minU) * 0.5f;
		float midV = (currentBox.maxV + currentBox.minV) * 0.5f;

		// Obtain index by comparing the points with the mid-points
		int index = 0;
		if (u >= midU) index += 1;
		if (v >= midV) index += 2;

		float sumWeights = 0.f;
		for (int i = 0; i < 4; i++) {
			if (node->children[i] != nullptr) {
				sumWeights += node->children[i]->weight;
			}
		}
		// If the child node is null OR it's weight is 0, return 0
		if (node->children[index] == nullptr || sumWeights <= 0.f) return 0.f;
		float probability = node->children[index]->weight / sumWeights;

		// Calculate the child's bounding box
		QTreeBox childrenBox{};
		childrenBox.minU = (index % 2 == 1) ? midU : currentBox.minU;
		childrenBox.maxU = (index % 2 == 1) ? currentBox.maxU : midU;
		childrenBox.minV = (index >= 2) ? midV : currentBox.minV;
		childrenBox.maxV = (index >= 2) ? currentBox.maxV : midV;

		// Recursive call
		// Update the currentPdf by multiplying with the probability of the chosen child and the area factor (4.f) since we are in a quadtree
		currentPdf *= probability * 4.f;
		return pdfRecursive(node->children[index], u, v, childrenBox, depth + 1, currentPdf);
	}

	void deleteRecursive(QTreeNode* node) {
		if (node == nullptr) return;
		for (int i = 0; i < 4; i++) deleteRecursive(node->children[i]);
		delete node;
	}
public:
	// Constructor & Destructor
	QTree(int _maxDepth = 4) {
		maxDepth = _maxDepth;
		nextFreeNode = 0;
		root = allocateNode();
	}

	// Methods
	void insert(float u, float v, float luminance) {
		if (luminance < 0 || u < 0.f || u >= 1.f || v < 0.f || v >= 1.f) return;
		QTreeBox rootBox{};
		rootBox.minU = 0.f; rootBox.maxU = 1.0f;
		rootBox.minV = 0.f; rootBox.maxV = 1.0f;
		insertRecursive(root, u, v, luminance, rootBox, 0);
	}

	void sample(float r1, float r2, float& u, float& v, float& pdf) {
		if (root->weight <= 0.f) { u = r1; v = r2; pdf = 1.f; return; }
		QTreeBox rootBox{};
		rootBox.minU = 0.f; rootBox.maxU = 1.0f;
		rootBox.minV = 0.f; rootBox.maxV = 1.0f;
		sampleRecursive(root, r1, r2, rootBox, 0, u, v, pdf, 1.f);
	}

	float pdf(float u, float v) {
		if (root->weight <= 0.f) return 1.f;
		if (u < 0.f || u >= 1.f || v < 0.f || v >= 1.f) return 0.f;
		QTreeBox rootBox{};
		rootBox.minU = 0.f; rootBox.maxU = 1.0f;
		rootBox.minV = 0.f; rootBox.maxV = 1.0f;
		return pdfRecursive(root, u, v, rootBox, 0, 1.f);
	}
};