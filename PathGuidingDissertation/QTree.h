#pragma once

#include <vector>

// uv-coordinates
// Child Index 0: Bottom-Left  (0,0)
// Child Index 1: Bottom-Right (1,0)
// Child Index 2: Top-Left     (0,1)
// Child Index 3: Top-Right    (1,1)

struct QTreeBox {
	// Attributes
	float minU; float maxU;
	float minV; float maxV;

	// Helper Function: Area
	float area() const { return ((maxU - minU) * (maxV - minV)); }
};

struct QTreeNode {
	float weight = 0.f;
	int children[4] = { -1, -1, -1, -1 };
};

// --- Directional-Tree Component of the Path Guiding ---
class QTree {
private:
	// Attributes
	std::vector<QTreeNode> nodePool;
	int rootIndex = -1;
	int maxDepth;

	// Private Methods
	int allocateNode() {
		// Add a new node to the pool (push_back() vs emplace_back())
		nodePool.emplace_back(QTreeNode());

		// Return the index of the newly added node
		return (int)(nodePool.size() - 1);
	}

	void insertRecursive(int nodeIndex, float u, float v, float luminance, QTreeBox currentBox, int depth) {
		// Guard case for null node
		if (nodeIndex == -1) return;

		// Add luminance to node->totalWeight
		nodePool[nodeIndex].weight += luminance;

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
		if (nodePool[nodeIndex].children[index] == -1) {
			int newIndex = allocateNode();
			nodePool[nodeIndex].children[index] = newIndex;
		}

		// Calculate the child's bounding box
		QTreeBox childrenBox{};
		childrenBox.minU = (index % 2 == 1) ? midU : currentBox.minU;
		childrenBox.maxU = (index % 2 == 1) ? currentBox.maxU : midU;
		childrenBox.minV = (index >= 2) ? midV : currentBox.minV;
		childrenBox.maxV = (index >= 2) ? currentBox.maxV : midV;
		
		// Recursive call
		insertRecursive(nodePool[nodeIndex].children[index], u, v, luminance, childrenBox, depth + 1);
	}

	void sampleRecursive(int nodeIndex, float r1, float r2, QTreeBox currentBox, int depth, float& u, float& v, float& pdf, float currentPdf) {
		// Sample, and terminate the recursion when we are at the leaf node
		if (depth == maxDepth) {
			u = currentBox.minU + r1 * (currentBox.maxU - currentBox.minU);
			v = currentBox.minV + r2 * (currentBox.maxV - currentBox.minV);
			pdf = currentPdf;
			return;
		}

		// Helper function to get weights to reduce clutter
		auto getChildWeight = [&](int childIndex) -> float {
			if (childIndex == -1) return 0.f;
			return nodePool[childIndex].weight;
		};

		float w0 = getChildWeight(nodePool[nodeIndex].children[0]);  // Bottom-Left
		float w1 = getChildWeight(nodePool[nodeIndex].children[1]);  // Bottom-Right
		float w2 = getChildWeight(nodePool[nodeIndex].children[2]);  // Top-Left
		float w3 = getChildWeight(nodePool[nodeIndex].children[3]);  // Top-Right
		float sumWeights = w0 + w1 + w2 + w3;

		// Marginal Decision for u and v
		// U-Axis
		float weightLeft = w0 + w2;   // Left children
		float weightRight = w1 + w3;  // Right children

		// Safety guard, treating as leaf node to avoid zero divisions
		if (sumWeights <= 0.f) {
			u = currentBox.minU + r1 * (currentBox.maxU - currentBox.minU);
			v = currentBox.minV + r2 * (currentBox.maxV - currentBox.minV);
			pdf = currentPdf;
			return;
		}

		float leftProbability = weightLeft / sumWeights;
		bool isRight;
		if (weightLeft <= 0.f) isRight = true;
		else if (weightRight <= 0.f) isRight = false;
		else isRight = (r1 >= leftProbability);

		if (!isRight) {
			// Rescale r1 for the left child
			r1 /= leftProbability;  
		} else {
			// Rescale r1 for the right child
			float rightProbability = 1.f - leftProbability;
			r1 = (r1 - leftProbability) / rightProbability;  
		}

		// V-Axis
		float weightBottom = !isRight ? w0 : w1;  // Bottom children
		float weightTop = !isRight ? w2 : w3;     // Top children
		float sumVertical = weightBottom + weightTop;

		// Safety guard, treating as leaf node to avoid zero divisions
		if (sumVertical <= 0.f) {
			u = currentBox.minU + r1 * (currentBox.maxU - currentBox.minU);
			v = currentBox.minV + r2 * (currentBox.maxV - currentBox.minV);
			pdf = currentPdf;
			return;
		}

		float bottomProbability = weightBottom / sumVertical;
		bool isTop;
		if (weightBottom <= 0.f) isTop = true;
		else if (weightTop <= 0.f) isTop = false;
		else isTop = (r2 >= bottomProbability);

		if (!isTop) {
			// Rescale r2 for the bottom child
			r2 /= bottomProbability;
		} else {
			// Rescale r2 for the top child
			float topProbability = 1.f - bottomProbability;
			r2 = (r2 - bottomProbability) / topProbability;
		}

		// Clamp r1 and r2 to [0, 1)
		r1 = std::max(0.f, std::min(r1, 0.99999f));
		r2 = std::max(0.f, std::min(r2, 0.99999f));

		// Combine u and v decisions to find the child index
		int index = 0;
		if (isRight) index += 1;
		if (isTop) index += 2;

		// Calculate the probability
		int childIndex = nodePool[nodeIndex].children[index];
		float probability = getChildWeight(childIndex) / sumWeights;

		if (childIndex == -1) {
			u = currentBox.minU + r1 * (currentBox.maxU - currentBox.minU);
			v = currentBox.minV + r2 * (currentBox.maxV - currentBox.minV);
			pdf = currentPdf;
			return;
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
		sampleRecursive(nodePool[nodeIndex].children[index], r1, r2, childrenBox, depth + 1, u, v, pdf, currentPdf);
	}
	
	float pdfRecursive(int nodeIndex, float u, float v, QTreeBox currentBox, int depth, float currentPdf) {
		// Handle null node case
		if (nodeIndex == -1) return 0.f;

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
			if (nodePool[nodeIndex].children[i] != -1) {
				int currentChildIndex = nodePool[nodeIndex].children[i];
				sumWeights += nodePool[currentChildIndex].weight;
			}
		}
		// Treat as leaf node to avoid zero divisions
		if (sumWeights <= 0.f) return currentPdf;

		// If the child node is null OR it's weight is 0, return 0
		int childIndex = nodePool[nodeIndex].children[index];
		if (childIndex == -1 || nodePool[childIndex].weight <= 0.f) return 0.f;
		float probability = nodePool[childIndex].weight / sumWeights;

		// Calculate the child's bounding box
		QTreeBox childrenBox{};
		childrenBox.minU = (index % 2 == 1) ? midU : currentBox.minU;
		childrenBox.maxU = (index % 2 == 1) ? currentBox.maxU : midU;
		childrenBox.minV = (index >= 2) ? midV : currentBox.minV;
		childrenBox.maxV = (index >= 2) ? currentBox.maxV : midV;

		// Recursive call
		// Update the currentPdf by multiplying with the probability of the chosen child and the area factor (4.f) since we are in a quadtree
		currentPdf *= probability * 4.f;
		return pdfRecursive(nodePool[nodeIndex].children[index], u, v, childrenBox, depth + 1, currentPdf);
	}
public:
	// Constructor & Destructor
	QTree(int _maxDepth) {
		maxDepth = _maxDepth;
		rootIndex = allocateNode();
	}

	QTree() {
		maxDepth = 3;
		rootIndex = allocateNode();
	}

	// Methods
	void insert(float u, float v, float luminance) {
		if (!(luminance > 0.f) || std::isnan(luminance) || std::isinf(luminance) || u < 0.f || u >= 1.f || v < 0.f || v >= 1.f) return;
		QTreeBox rootBox{};
		rootBox.minU = 0.f; rootBox.maxU = 1.0f;
		rootBox.minV = 0.f; rootBox.maxV = 1.0f;
		insertRecursive(rootIndex, u, v, luminance, rootBox, 0);
	}

	void clear() {
		nodePool.clear();
		rootIndex = allocateNode();
	}

	void sample(float r1, float r2, float& u, float& v, float& pdf) {
		if (nodePool[rootIndex].weight <= 0.f) { u = r1; v = r2; pdf = 1.f; return; }
		QTreeBox rootBox{};
		rootBox.minU = 0.f; rootBox.maxU = 1.0f;
		rootBox.minV = 0.f; rootBox.maxV = 1.0f;
		sampleRecursive(rootIndex, r1, r2, rootBox, 0, u, v, pdf, 1.f);
	}

	float pdf(float u, float v) {
		if (nodePool[rootIndex].weight <= 0.f) return 1.f;
		if (u < 0.f || u >= 1.f || v < 0.f || v >= 1.f) return 0.f;
		QTreeBox rootBox{};
		rootBox.minU = 0.f; rootBox.maxU = 1.0f;
		rootBox.minV = 0.f; rootBox.maxV = 1.0f;
		return pdfRecursive(rootIndex, u, v, rootBox, 0, 1.f);
	}
};