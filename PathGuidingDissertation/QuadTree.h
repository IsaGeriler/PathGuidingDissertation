#pragma once

#include <cmath>

struct Point {
	// Attributes
	float u, v;

	// Constructors
	Point() { u = 0.f; v = 0.f; }
	Point(float _u, float _v) { u = _u; v = _v; }
};

struct QuadTreeNode {
	// Attributes
	Point point;
	float data;

	// Constructors
	QuadTreeNode() { data = 0.f; }
	QuadTreeNode(Point _point, float _data) { point = _point; data = _data; }
};

class QuadTree {
private:
	// Attributes
	// Top-left and bottom-right boundaries
	Point topLeftPosition;
	Point bottomRightPosition;

	// Current node
	QuadTreeNode* node;

	// Childrens of the quad tree
	QuadTree* topLeftTree;
	QuadTree* topRightTree;
	QuadTree* bottomLeftTree;
	QuadTree* bottomRightTree;
public:
	// Constructors
	QuadTree() {
		topLeftPosition = Point(0.f, 0.f);
		bottomRightPosition = Point(0.f, 0.f);
		node = nullptr;
		topLeftTree = nullptr;
		topRightTree = nullptr;
		bottomLeftTree = nullptr;
		bottomRightTree = nullptr;
	}

	QuadTree(Point _topLeft, Point _bottomRight) {
		topLeftPosition = _topLeft;
		bottomRightPosition = _bottomRight;
		node = nullptr;
		topLeftTree = nullptr;
		topRightTree = nullptr;
		bottomLeftTree = nullptr;
		bottomRightTree = nullptr;
	}

	// Destructor
	~QuadTree() {
		if (node != nullptr) delete node;
		if (topLeftTree != nullptr) delete topLeftTree;
		if (topRightTree != nullptr) delete topRightTree;
		if (bottomLeftTree != nullptr) delete bottomLeftTree;
		if (bottomRightTree != nullptr) delete bottomRightTree;
	}

	// Methods
	void insert(QuadTreeNode* _node) {
		// Return if the passed node is null
		if (_node == nullptr) return;

		// We cannot insert if the point is not even in the bounds
		if (!isInBoundary(_node->point)) return;

		// Handle case where we cannot divide the quad further
		if (std::abs(topLeftPosition.u - topLeftPosition.u <= 1.f) && std::abs(topLeftPosition.v - topLeftPosition.v <= 1.f)) {
			if (node == nullptr) node = _node;
			return;
		}

		// Handle Left
		if (((topLeftPosition.u + bottomRightPosition.u) * 0.5f) >= _node->point.u) {
			// Top Left
			if (((topLeftPosition.v + bottomRightPosition.v) * 0.5f) >= _node->point.v) {
				// Only insert if the tree reference is nullptr
				if (topLeftTree == nullptr) {
					topLeftTree = new QuadTree(
						Point(topLeftPosition.u, topLeftPosition.v),
						Point((topLeftPosition.u + bottomRightPosition.u) * 0.5f, (topLeftPosition.v + bottomRightPosition.v) * 0.5f)
					);
					topLeftTree->insert(_node);
				}
			}
			// Bottom Left
			else {
				// Only insert if the tree reference is nullptr
				if (bottomLeftTree == nullptr) {
					bottomLeftTree = new QuadTree(
						Point(topLeftPosition.u, (topLeftPosition.v + bottomRightPosition.v) * 0.5f),
						Point((topLeftPosition.u + bottomRightPosition.u) * 0.5f, bottomRightPosition.v)
					);
					bottomLeftTree->insert(_node);
				}
			}
		}
		// Handle Right
		else {
			// Top Right
			if (((topLeftPosition.v + bottomRightPosition.v) * 0.5f) >= _node->point.v) {
				// Only insert if the tree reference is nullptr
				if (topRightTree == nullptr) {
					topRightTree = new QuadTree(
						Point((topLeftPosition.u + bottomRightPosition.u) * 0.5f, topLeftPosition.v),
						Point(bottomRightPosition.u, (topLeftPosition.v + bottomRightPosition.v) * 0.5f)
					);
					topRightTree->insert(_node);
				}
			}
			// Bottom Right
			else {
				// Only insert if the tree reference is nullptr
				if (bottomRightTree == nullptr) {
					bottomRightTree = new QuadTree(
						Point((topLeftPosition.u + bottomRightPosition.u) * 0.5f, (topLeftPosition.v + bottomRightPosition.v) * 0.5f),
						Point(bottomRightPosition.u, bottomRightPosition.v)
					);
					bottomRightTree->insert(_node);
				}
			}
		}
	}

	QuadTreeNode* search(Point _point) {
		// Return nullptr if the point is out of bounds
		if (!isInBoundary(_point)) return nullptr;

		// We cannot subdivide the quadrant further
		if (node != nullptr) return node;

		// Handle Left
		if (((topLeftPosition.u + bottomRightPosition.u) * 0.5f) >= _point.u) {
			// Top Left
			if (((topLeftPosition.v + bottomRightPosition.v) * 0.5f) >= _point.v) {
				if (topLeftTree == nullptr) return nullptr;
				return topLeftTree->search(_point);
			}
			// Bottom Left
			else {
				if (bottomLeftTree == nullptr) return nullptr;
				return bottomLeftTree->search(_point);
			}
		}
		// Handle Right
		else {
			// Top Right
			if (((topLeftPosition.v + bottomRightPosition.v) * 0.5f) >= _point.v) {
				if (topRightTree == nullptr) return nullptr;
				return topRightTree->search(_point);
			}
			// Bottom Right
			else {
				if (bottomRightTree == nullptr) return nullptr;
				return bottomRightTree->search(_point);
			}
		}
	}

	bool isInBoundary(Point _point) const {
		return (_point.u >= topLeftPosition.u) && (_point.u <= bottomRightPosition.u) &&
			   (_point.v >= topLeftPosition.v) && (_point.v <= bottomRightPosition.v);
	}
};