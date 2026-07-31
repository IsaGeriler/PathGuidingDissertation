#pragma once

struct QTreeBox {
	float minU, maxU;
	float minV, maxV;
	float area() const { return (maxU - minU) * (maxV - minV); }
};

// uv-coordinates
// Child Index 0: Bottom-Left  (0,0)
// Child Index 1: Bottom-Right (1,0)
// Child Index 2: Top-Left     (0,1)
// Child Index 3: Top-Right    (1,1)
struct QTreeNode {
	float totalWeight = 0.f;
	QTreeNode* children[4]{};
	bool isLeaf() const { return (children[0] == nullptr) && (children[1] == nullptr) && (children[2] == nullptr) && (children[3] == nullptr); }
};

class QTree {
private:
	void insertRecursive(QTreeNode* node, float u, float v, float luminance, QTreeBox currentBox, int depth) {
		// Add luminance to node->totalWeight
		node->totalWeight += luminance;

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
		if (node->children[index] == nullptr) node->children[index] = new QTreeNode();
		
		// Calculate the child's bounding box
		QTreeBox childrenBox;
		childrenBox.minU = (index % 2 == 1) ? midU : currentBox.minU;
		childrenBox.maxU = (index % 2 == 1) ? currentBox.maxU : midU;
		childrenBox.minV = (index % 2 == 1) ? midV : currentBox.minV;
		childrenBox.maxV = (index % 2 == 1) ? currentBox.maxV : midV;
		
		// Recursive call
		insertRecursive(node->children[index], u, v, luminance, childrenBox, depth + 1);
	}

	void sampleRecursive(QTreeNode* node, float r1, float r2, QTreeBox currentBox, int depth, float& u, float& v, float& pdf) {
		// If depth == maxDepth
		// -> calculate output u and v
		// -> calculate output pdf by totalWeight / root->totalWeight / box.area
		// -> return
		// Roulette with r1
		// Rescale r1
		// Calculate childBox
		// Recursive call
	}
	
	float pdfRecursive(QTreeNode* node, float u, float v, QTreeBox currentBox, int depth) {
		// Terminate the recursion when we reach the max depth of the tree
		// PDF = (currentNodeWeight / rootNodeWeight) / (boundingBoxArea)
		if (depth == maxDepth) return (node->totalWeight / root->totalWeight) / currentBox.area();

		// Find quadrant index for (u,v) using bounding box midpoints
		// Calculate mid-ponts
		float midU = (currentBox.maxU + currentBox.minU) * 0.5f;
		float midV = (currentBox.maxV + currentBox.minV) * 0.5f;

		// Obtain index by comparing the points with the mid-points
		int index = 0;
		if (u >= midU) index += 1;
		if (v >= midV) index += 2;

		// If the child node is null OR it's weight is 0, return 0
		if (node->children[index] == nullptr || node->children[index]->totalWeight == 0) return 0.f;

		// Calculate the child's bounding box
		QTreeBox childrenBox;
		childrenBox.minU = (index % 2 == 1) ? midU : currentBox.minU;
		childrenBox.maxU = (index % 2 == 1) ? currentBox.maxU : midU;
		childrenBox.minV = (index % 2 == 1) ? midV : currentBox.minV;
		childrenBox.maxV = (index % 2 == 1) ? currentBox.maxV : midV;

		// Recursive call
		pdfRecursive(node->children[index], u, v, childrenBox, depth + 1);
	}

	void deleteRecursive(QTreeNode* node) {
		if (node == nullptr) return;
		for (int i = 0; i < 4; i++) deleteRecursive(node->children[i]);
		delete node;
	}
public:
	// Attributes
	QTreeNode* root;
	int maxDepth;

	// Constructor & Destructor
	QTree(int _maxDepth = 6) {
		maxDepth = _maxDepth;
		root = new QTreeNode();
	}

	~QTree() {
		deleteRecursive(root);
	}

	// Methods
	void insert(float u, float v, float luminance) {
		if (luminance < 0 || u < 0.f || u >= 1.f || v < 0.f || v >= 1.f) return;
		QTreeBox rootBox{ 0.f, 1.f, 0.f, 1.f };
		insertRecursive(root, u, v, luminance, rootBox, 0);
	}

	void sample(float r1, float r2, float& u, float& v, float& pdf) {
		if (root->totalWeight <= 0.f) { u = r1; v = r2; pdf = 1; return; }
		QTreeBox rootBox{ 0.f, 1.f, 0.f, 1.f };
		sampleRecursive(root, r1, r2, rootBox, 0.f, u, v, pdf);
	}

	float pdf(float u, float v) {
		if (root->totalWeight <= 0.f) return 1.f;
		if (u < 0.f || u >= 1.f || v < 0.f || v >= 1.f) return 0.f;
		QTreeBox rootBox{ 0.f, 1.f, 0.f, 1.f };
		pdfRecursive(root, u, v, rootBox, 0);
	}
};