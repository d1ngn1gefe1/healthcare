#include <stdio.h>
#include <iostream>
using namespace std;

float dx = 0;
float dy = 1.04e3;
float dz = 3.52e3;

float unitVectors[4][4] = {{-1,0,0,0}, {0,0,-1,0}, {0,-1,0,0}, {0,0,0,1}};

float* rigidBodyMotion(float frontJoint[4], float dx, float dy, float dz, float unitVectors[4][4]) {
	float rotation[4][4];
	float projection[4][4];
	float t[4] = {dx, dy, dz, 0};
	float translation[4];
	float topJoint[4];

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			rotation[i][j] = unitVectors[i][j];
			projection[i][j] = rotation[i][j];
		}
	}

	for (int i = 0; i < 4; i++) {
		float sum = 0;
		for (int j = 0; j < 4; j++) {
			sum += -rotation[i][j] * t[j];
		}
		translation[i] = sum;
		projection[i][3] = translation[i];
	}

	for (int i = 0; i < 4; i++) {
		float sum = 0;
		for (int j = 0; j < 4; j++) {
			sum += projection[i][j] * frontJoint[j];
		}
		topJoint[i] = sum;
	}

	/*
	cout << "Rotation matrix: " << endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << rotation[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Translation vector: " << endl;
	for (int i = 0; i < 4; i++) {
		cout << translation[i] << " ";
	}
	cout << endl;

	cout << "Projection matrix: " << endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << projection[i][j] << " ";
		}
		cout << endl;
	}
	*/

	return topJoint;

}

// num is number of rows in front
void projection(float front[][5], float top[][3], int num) {
	for (int i = 0; i < num; i++) {
		float frontJoint[4] = {front[i][0], front[i][1], front[i][2], 1};
		float* coords = rigidBodyMotion(frontJoint,dx,dy,dz,unitVectors);
		
		for (int j = 0; j < 3; j++) {
			top[i][j] = *(coords + j);
		}
		
	}
}

int main() {
	float front[2][5] = {{-27.8104,411.304,3073.35,157.42,81.8357}, {-26.568,190.717,3092.72,157.55,102.415}};
	float top[2][3];
	projection(front,top,2);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			cout << top[i][j] << " ";
		}
		cout << endl;
	}

	return 0;
}