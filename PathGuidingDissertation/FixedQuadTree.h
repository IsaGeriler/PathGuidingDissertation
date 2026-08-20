#pragma once

#include <algorithm>
#include <cmath>

// Struct - Node (just store weights)
struct Node { float weight = 0.f; };

// New Quad Tree Class - Fixed Depth 2 (21 Nodes!)
class FixedQuadTree {
public:
	// Attributes
	static constexpr int MAX_NODES = 21;
	int maxDepth = 2;
	Node nodes[MAX_NODES];

	// Constructor
	FixedQuadTree(int depth = 2) : maxDepth(std::min(depth, 2)) { clear(); }

	// Method - Clear node weights
	void clear() { for (auto& node : nodes) { node.weight = 0.f; } }

	// Method - Insert uv into Quad Tree
	void insert(float u, float v, float weight) {
		// Check if the weight is valid or non-zero
		if (weight <= 0.f || std::isnan(weight) || std::isinf(weight)) return;

		// Clamp uv into [0,1) just in case...
		u = std::min(std::max(u, 0.f), 0.999999f);
		v = std::min(std::max(v, 0.f), 0.999999f);

		// Always insert to the root (index: 0)
		nodes[0].weight += weight;
		if (maxDepth == 0) return;

		// Depth 1 Check (index between 1 and 4)
		// Find the quadrant
		int uFirstQuadrant = (u >= 0.5f) ? 1 : 0;
		int vFirstQuadrant = (v >= 0.5f) ? 1 : 0;
		int childIdx1 = 1 + (uFirstQuadrant + vFirstQuadrant * 2);
		nodes[childIdx1].weight += weight;
		if (maxDepth == 1) return;

		// Depth 2 Check (index between 5 and 21)
		// Find the subquadrant within the first depth quadrant
		float uSubQuadrant = (u >= 0.5f) ? (u - 0.5f) * 2.f : u * 2.f;
		float vSubQuadrant = (v >= 0.5f) ? (v - 0.5f) * 2.f : v * 2.f;
		int uSecondQuadrant = (uSubQuadrant >= 0.5f) ? 1 : 0;
		int vSecondQuadrant = (vSubQuadrant >= 0.5f) ? 1 : 0;
		int childIdx2 = 5 + (childIdx1 - 1) * 4 + (uSecondQuadrant * vSecondQuadrant * 2);
		nodes[childIdx2].weight += weight;
	}

	// Method - Sample a given uv-point in Quad Tree
	void sample(float r1, float r2, float& u, float& v, float& pdf) {
		// If the tree is empty do a uniform sampling
		if (nodes[0].weight <= 0.f) { u = r1; v = r2; pdf = 1.f; return; }

		// Hypercube parameters
		float minU = 0.f, maxU = 1.f;
		float minV = 0.f, maxV = 1.f;
		int currentNodeIdx = 0;

		// Traverse through the tree
		for (int depth = 0; depth < maxDepth; depth++) {
			int firstChildIdx = (depth == 0) ? 1 : (5 + (currentNodeIdx - 1) * 4);
			
			// Save the weights of corresponding childs
			float w0 = nodes[firstChildIdx + 0].weight;
			float w1 = nodes[firstChildIdx + 1].weight;
			float w2 = nodes[firstChildIdx + 2].weight;
			float w3 = nodes[firstChildIdx + 3].weight;
			float sumWeights = w0 + w1 + w2 + w3;

			// Break the loop if the total weight is zero
			if (sumWeights <= 0.f) break;

			// 1. Marginal over U (left side vs right side)
			float weightLeft = w0 + w2;
			float probabilityLeft = weightLeft / sumWeights;

			// Pick left side at first
			int qU = 0;
			if (r1 < probabilityLeft) {
				r1 = (probabilityLeft > 0.f) ? (r1 / probabilityLeft) : 0.5f;
			} else {
				// Pick right side as r1 > probabilityLeft
				qU = 1;
				float probabilityRight = 1.f - probabilityLeft;
				r1 = (probabilityRight > 0.f) ? ((r1 - probabilityLeft) / probabilityRight) : 0.5f;
			}

			// 2. Conditional over V (bottom side vs top side)
			float weightBottom = (qU == 0) ? w0 : w1;
			float weightTop = (qU == 0) ? w2 : w3;
			float sumV = weightBottom + weightTop;
			float probabilityBottom = (sumV > 0.f) ? (weightBottom / sumV) : 0.5f;

			// Pick bottom side at first
			int qV = 0;
			if (r2 < probabilityBottom) {
				r2 = (probabilityBottom > 0.f) ? (r2 / probabilityBottom) : 0.5f;
			} else {
				// Pick top side as r2 > probabilityBottom
				qV = 1;
				float probabilityTop = 1.f - probabilityBottom;
				r2 = (probabilityTop > 0.f) ? ((r2 - probabilityBottom) / probabilityTop) : 0.5f;
			}

			// Subdivide the bounding box based on u and v selections
			float midU = 0.5f * (minU + maxU);
			float midV = 0.5f * (minV + maxV);
			if (qU == 1) minU = midU; else maxU = midU;
			if (qV == 1) minV = midV; else maxV = midV;

			// Update the selected child for the next iteration
			currentNodeIdx = firstChildIdx + (qU + qV * 2);
		}

		// Update to be returned u and v
		float sampledU = minU + r1 * (maxU - minU);
		float sampledV = minV + r2 * (maxV - minV);
		u = std::min(std::max(sampledU, 0.f), 0.999999f);
		v = std::min(std::max(sampledV, 0.f), 0.999999f);

		// Update to be returned PDF
		float leafArea = (maxU - minU) * (maxV - minV);
		float leafProbability = nodes[currentNodeIdx].weight / nodes[0].weight;
		pdf = leafProbability / leafArea;
	}

	// Method - Calculate the PDF of a given uv-point in Quad Tree
	float pdf(float u, float v) const {
		// Handle the case where node is empty
		if (nodes[0].weight <= 0.f) return 1.f;

		// Handle the case where u and v violate PSS
		if (u < 0.f || u >= 1.f || v < 0.f || v >= 1.f) return 0.f;

		// Define the bounding box for determining quadrant we'll work on
		float minU = 0.f, maxU = 1.f;
		float minV = 0.f, maxV = 1.f;
		int currentNodeIdx = 0;

		// Traverse through the tree
		for (int depth = 0; depth < maxDepth; depth++) {
			int firstChildIdx = (depth == 0) ? 1 : (5 + (currentNodeIdx - 1) * 4);
			
			float midU = 0.5f * (minU + maxU);
			float midV = 0.5f * (minV + maxV);

			int quadrant = (u >= midU ? 1 : 0) + (v >= midV ? 2 : 0);
			if (u >= midU) minU = midU; else maxU = midU;
			if (v >= midV) minV = midV; else maxV = midV;

			currentNodeIdx = firstChildIdx + quadrant;
			if (nodes[currentNodeIdx].weight <= 0.f) return 0.f;
		}
		float leafArea = (maxU - minU) * (maxV - minV);
		float leafProbability = nodes[currentNodeIdx].weight / nodes[0].weight;
		return leafProbability / leafArea;
	}
};