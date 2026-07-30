#pragma once

struct QTreeBox {
	float minU, minV;
	float maxU, maxV;
	float area() const { return (maxU - minU) * (maxV - minV); }
};

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
		if (depth == maxDepth) return;
		// Find quadrant index using bounding box midpoints
		// If node->children[idx] == nullptr, new QTReeNode
		// Calculate childBox
		// Recursive call
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
	
	void pdfRecursive(QTreeNode* node, float u, float v, QTreeBox currentBox, int depth) {
		// If depth == maxDepth
		// -> return totalWeight / root->totalWeight / box.area
		// Find quadrant for (u,v)
		// If child null OR it's weight 0 return 0
		// Calculate childBox
		// Recursive call
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