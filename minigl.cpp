/**
 * minigl.cpp
 * -------------------------------
 * Implement miniGL here.
 *
 * You may include minigl.h and any of  standard C++ libraries.
 * No or includes are permitted.  Or preprocessing directives
 * are also not permitted.  se requirements are strictly
 * enforced.  Be sure to run a test grading to make sure your file
 * passes  sanity tests.
 *
 *  behavior of  routines your are implenting is documented here:
 * https://www.opengl.org/sdk/docs/man2/
 * Note that you will only be implementing a subset of this.  In particular,
 * you only need to implement enough to pass  tests in  suite.
 */

#include "minigl.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <cstdio>
#include <list>
#include <stack>



#ifndef __vec__
#define __vec__

#include <cmath>
#include <iostream>
#include <cassert>

template<class T, int n> struct vec;
template<class T, int n> T dot(const vec<T, n>& u, const vec<T, n>& v);

template<class T, int n>
struct vec {
	T x[n];

	vec() {
		make_zero();
	}

	explicit vec(const T& a) {
		assert(n == 1);
		x[0] = a;
	}

	vec(const T& a, const T& b) {
		assert(n == 2);
		x[0] = a;
		x[1] = b;
	}

	vec(const T& a, const T& b, const T& c) {
		assert(n == 3);
		x[0] = a;
		x[1] = b;
		x[2] = c;
	}

	vec(const T& a, const T& b, const T& c, const T& d) {
		assert(n == 4);
		x[0] = a;
		x[1] = b;
		x[2] = c;
		x[3] = d;
	}

	template<class U>
	explicit vec(const vec<U, n>& v) {
		for (int i = 0; i < n; i++)
			x[i] = (T) v.x[i];
	}

	void make_zero() {
		for (int i = 0; i < n; i++)
			x[i] = 0;
	}

	vec& operator +=(const vec& v) {
		for (int i = 0; i < n; i++)
			x[i] += v.x[i];
		return *this;
	}

	vec& operator -=(const vec& v) {
		for (int i = 0; i < n; i++)
			x[i] -= v.x[i];
		return *this;
	}

	vec& operator *=(const vec& v) {
		for (int i = 0; i < n; i++)
			x[i] *= v.x[i];
		return *this;
	}

	vec& operator /=(const vec& v) {
		for (int i = 0; i < n; i++)
			x[i] /= v.x[i];
		return *this;
	}

	vec& operator *=(const T& c) {
		for (int i = 0; i < n; i++)
			x[i] *= c;
		return *this;
	}

	vec& operator /=(const T& c) {
		for (int i = 0; i < n; i++)
			x[i] /= c;
		return *this;
	}

	vec operator +() const {
		return *this;
	}

	vec operator -() const {
		vec r;
		for (int i = 0; i < n; i++)
			r[i] = -x[i];
		return r;
	}

	vec operator +(const vec& v) const {
		vec r;
		for (int i = 0; i < n; i++)
			r[i] = x[i] + v.x[i];
		return r;
	}

	vec operator -(const vec& v) const {
		vec r;
		for (int i = 0; i < n; i++)
			r[i] = x[i] - v.x[i];
		return r;
	}

	vec operator *(const vec& v) const {
		vec r;
		for (int i = 0; i < n; i++)
			r[i] = x[i] * v.x[i];
		return r;
	}

	vec operator /(const vec& v) const {
		vec r;
		for (int i = 0; i < n; i++)
			r[i] = x[i] / v.x[i];
		return r;
	}

	vec operator *(const T& c) const {
		vec r;
		for (int i = 0; i < n; i++)
			r[i] = x[i] * c;
		return r;
	}

	vec operator /(const T& c) const {
		vec r;
		for (int i = 0; i < n; i++)
			r[i] = x[i] / c;
		return r;
	}

	const T& operator[](int i) const {
		return x[i];
	}

	T& operator[](int i) {
		return x[i];
	}

	T magnitude_squared() const {
		return dot(*this, *this);
	}

	T magnitude() const {
		return sqrt(magnitude_squared());
	}

	// Be careful to handle  zero vector gracefully
	vec normalized() const {
		T mag = magnitude();
		if (mag)
			return *this / mag;
		vec r;
		r[0] = 1;
		return r;
	}
	;
};

template<class T, int n>
vec<T, n> operator *(const T& c, const vec<T, n>& v) {
	return v * c;
}

template<class T, int n>
T dot(const vec<T, n> & u, const vec<T, n> & v) {
	T r = 0;
	for (int i = 0; i < n; i++)
		r += u.x[i] * v.x[i];
	return r;
}

template<class T>
vec<T, 3> cross(const vec<T, 3> & u, const vec<T, 3> & v) {
	return vec<T, 3>(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
			u[0] * v[1] - u[1] * v[0]);
}

template<class T, int n>
std::ostream& operator <<(std::ostream& out, const vec<T, n> & u) {
	for (int i = 0; i < n; i++) {
		if (i)
			out << ' ';
		out << u[i];
	}
	return out;
}

template<class T, int n>
std::istream& operator >>(std::istream& in, vec<T, n> & u) {
	for (int i = 0; i < n; i++) {
		in >> u[i];
	}
	return in;
}

#endif

/*
 * End of vec.h
 */

using namespace std;

/**
 * Standard macro to report errors
 */
inline void MGL_ERROR(const char* description) {
	printf("%s\n", description);
	exit(1);
}

MGLbool started= false;
MGLpoly_mode poly;
typedef vec<MGLfloat, 3> vec3;
typedef vec<MGLfloat, 4> vec4;
vec3 Color(1, 1, 1);
MGLsize Width, Height;

class MGL_Ver {
public:
	MGL_Ver() :
			Vec(0, 0, 0, 1), w(1) {
	}
	;
	MGL_Ver(const MGL_Ver& b) :
			Vec(b.Vec), color(b.color), w(b.w) {
	}
	;
	MGL_Ver(MGLfloat x, MGLfloat y, MGLfloat z, const vec3& c) :
			Vec(x, y, z, 1), color(c), w(1) {
	}
	;

	vec3 GetVec3() {
		return vec3(Vec[0], Vec[1], Vec[2]);
	}

	vec4 Vec;
	vec3 color;
	MGLfloat w; //Useless before Norm_ViewPort;
};
vector<MGL_Ver> VertexChain_Buffer;
vector<MGL_Ver> VertexChain;

/**
 * Init:
 *
 *   ( a0  a4  a8  a12 )
 *   ( a1  a5  a9  a13 )
 *   ( a2  a6  a10 a14 )
 *   ( a3  a7  a11 a15 )
 *
 * where ai is  i'th entry of  array.
 */
class MGLMatrix {
public:
	MGLMatrix() {
		for (int i = 0; i < 16; ++i)
			m[i] = 0;
		m[0] = 1;
		m[5] = 1;
		m[10] = 1;
		m[15] = 1;
	}

	MGLMatrix(const MGLMatrix& o) {
		for (int i = 0; i < 16; ++i)
			m[i] = o.m[i];
	}

	MGLMatrix & operator =(const MGLMatrix & rhs) {
		for (int i = 0; i < 16; ++i)
			m[i] = rhs.m[i];
		return *this;
	}

	MGLfloat m[16];
};

MGLmatrix_mode Mode;
MGLMatrix MMatrix[MGL_PROJECTION + 1];
stack<MGLMatrix> PROJ_MatrixStack;
stack<MGLMatrix> MV_MatrixStack;

class MGL_FrameBuf {
public:

	MGL_FrameBuf(MGLsize width, MGLsize height) :
			width(width), height(height) {
		for (MGLsize i = 0; i < width; ++i) {
			for (MGLsize j = 0; j < height; ++j) {
				MGLpixel black = Make_Pixel(0, 0, 0);
				MGLfloat far_away = 2;
				Buf.push_back(black);
				zBuf.push_back(far_away);
			}
		}
	}
	;

	MGLpixel GetOne(MGLsize x, MGLsize y) {
		return Buf[pos(x, y)];
	}

	void SetOne(MGLsize x, MGLsize y, MGLfloat z, MGLpixel p) {
		if ((z < 1) && (z > -1))
			if (z < zBuf[pos(x, y)]) {
				Buf[pos(x, y)] = p;
				zBuf[pos(x, y)] = z;
			}
	}

	size_t pos(MGLsize x, MGLsize y) {
		if ((x >= width) || (y >= height) || (x < 0) || (y < 0)) {
			MGL_ERROR("FATAL: OUT OF RANGE!, Gen by MGL_FrameBuf::pos");
		}
		return y * width + x;
	}

	vector<MGLpixel> Buf;
	vector<MGLfloat> zBuf;
	MGLsize width, height;
};

void FillTri(MGL_FrameBuf &buffer, MGL_Ver t1, MGL_Ver t2, MGL_Ver t3) {
	vec3 v1 = t1.GetVec3();
	vec3 v2 = t2.GetVec3();
	vec3 v3 = t3.GetVec3();

	MGLfloat bottom = min(v1[1], min(v2[1], v3[1])), left = min(v1[0],
			min(v2[0], v3[0])), top = max(v1[1], max(v2[1], v3[1])), right =
			max(v1[0], max(v2[0], v3[0]));

	vec3 b_a(v2 - v1);
	vec3 c_a(v3 - v1);
	vec3 c_b(v3 - v2);
	vec3 a_c(v1 - v3);
	vec3 n(cross(b_a, c_a));
	MGLfloat n_square = n.magnitude_squared();
	n /= n_square;

	// 0~1 => 0~255
	t1.color *= 255;
	t2.color *= 255;
	t3.color *= 255;

	if (left < 0)
		left = 0.1;		//floor
	if (right >= Width)
		right = Width - 0.5;		//ceil
	if (bottom < 0)
		bottom = 0.1;
	if (top >= Height)
		top = Height - 0.5;

	for (int i = floor(left); i < ceil(right); ++i) {
		for (int j = floor(bottom); j < ceil(top); ++j) {
			vec3 p(i + 0.5, j + 0.5, 0);
			vec3 na(cross(c_b, p - v2));
			vec3 nb(cross(a_c, p - v3));

			MGLfloat alpha = dot(n, na);
			MGLfloat beta = dot(n, nb);
			MGLfloat gamma = 1 - alpha - beta;

			MGLfloat da = alpha / t1.w;
			MGLfloat db = beta / t2.w;
			MGLfloat dc = gamma / t3.w;
			MGLfloat k = da + db + dc;

			if ((alpha >= 0) && (beta >= 0) && (gamma >= 0)) {
				MGLfloat z = alpha * t1.Vec[2] + beta * t2.Vec[2]
						+ gamma * t3.Vec[2];
				alpha = da / k;
				beta = db / k;
				gamma = dc / k;
				vec3 p = alpha * t1.color + beta * t2.color + gamma * t3.color;
				buffer.SetOne(i, j, z,
						Make_Pixel(round(p[0]), round(p[1]), round(p[2])));
			}
		}
	}
}

MGL_Ver Norm_ViewPort(MGL_Ver input, MGLsize width, MGLsize height) {
	//Norm
	input.w = input.Vec[3];
	input.Vec /= input.Vec[3];
	//STEP 4 IN Vertex_Transformation
	input.Vec[0] = (input.Vec[0] * 0.5 + 0.5) * width;
	input.Vec[1] = (input.Vec[1] * 0.5 + 0.5) * height;
	//-1~1 => 0~1
	//input.Vec[2] = (1.0 + input.Vec[2]) * 0.5;
	return input;
}

MGL_Ver MulRatio(const MGL_Ver& A, const MGL_Ver& B, MGLfloat ratio) {
	MGL_Ver ret;
	ret.Vec = (1 - ratio) * A.Vec + ratio * B.Vec;
	ret.color = (1 - ratio) * A.color + ratio * B.color;
	return ret;
}

//clip against a pannel
//return new triangles
void ClipAgainst(vector<MGL_Ver>& input, vec4 normal, vec4 p) {
	vector<MGL_Ver> in(input);
	input.clear();

	for (size_t i = 0; i < in.size(); i += 3) {
		MGL_Ver A = in[i];
		MGL_Ver B = in[i + 1];
		MGL_Ver C = in[i + 2];

		MGLfloat fa = dot(normal, A.Vec - p);
		MGLfloat fb = dot(normal, B.Vec - p);
		MGLfloat fc = dot(normal, C.Vec - p);

		unsigned int In = 0;
		if (fa < 0)
			In = 1;
		if (fb < 0)
			In = In | 2;
		if (fc < 0)
			In = In | 4;

		MGLfloat sAB = dot(normal, B.Vec - A.Vec);
		if (sAB != 0) {
			sAB = dot(normal, p - A.Vec) / sAB;
		} else {
			sAB = -1;	//n treat as s<0
		}
		MGLfloat sAC = dot(normal, C.Vec - A.Vec);
		if (sAC != 0) {
			sAC = dot(normal, p - A.Vec) / sAC;
		} else {
			sAC = -1;	//n treat as s<0
		}
		MGLfloat sBC = dot(normal, C.Vec - B.Vec);
		if (sBC != 0) {
			sBC = dot(normal, p - B.Vec) / sBC;
		} else {
			sBC = -1;	//n treat as s<0
		}

		MGL_Ver O, Q;
		switch (In) {
		case 7:
			input.push_back(A);
			input.push_back(B);
			input.push_back(C);
			break;
		case 6:
			O = MulRatio(A, B, sAB);
			Q = MulRatio(A, C, sAC);
			input.push_back(O);
			input.push_back(B);
			input.push_back(C);
			input.push_back(C);
			input.push_back(Q);
			input.push_back(O);
			break;
		case 5:
			O = MulRatio(B, C, sBC);
			Q = MulRatio(B, A, 1 - sAB);
			input.push_back(O);
			input.push_back(C);
			input.push_back(A);
			input.push_back(A);
			input.push_back(Q);
			input.push_back(O);
			break;
		case 4:
			O = MulRatio(A, C, sAC);
			Q = MulRatio(B, C, sBC);
			input.push_back(C);
			input.push_back(O);
			input.push_back(Q);
			break;
		case 3:
			O = MulRatio(C, B, 1 - sBC);
			Q = MulRatio(C, A, 1 - sAC);
			input.push_back(O);
			input.push_back(B);
			input.push_back(A);
			input.push_back(A);
			input.push_back(Q);
			input.push_back(O);
			break;
		case 2:
			O = MulRatio(A, B, sAB);
			Q = MulRatio(C, B, 1 - sBC);
			input.push_back(O);
			input.push_back(B);
			input.push_back(Q);
			break;
		case 1:
			O = MulRatio(A, B, sAB);
			Q = MulRatio(A, C, sAC);
			input.push_back(A);
			input.push_back(O);
			input.push_back(Q);
			break;
		case 0:
		default:
			break;
		}

	}
}

void ClipAndPush(vector<MGL_Ver> & input, const MGL_Ver& A, const MGL_Ver& B,
		const MGL_Ver& C) {
	//TODO
	vector<MGL_Ver> tempVertex;
	tempVertex.push_back(A);
	tempVertex.push_back(B);
	tempVertex.push_back(C);

	vec4 SharedP;
	vec4 zwn(0,0,1,-1);//Normal of z=w pannel
	vec4 znwn(0,0,-1,-1);//Normal of z=-w pannel
//	vec4 ywn(0,0,1,1);//Normal of y=w pannel
//	vec4 ynwn(0,0,1,1);//Normal of y=-w pannel
//	vec4 xwn(0,0,1,1);//Normal of x=w pannel
//	vec4 xnwn(0,0,1,1);//Normal of x=-w pannel

	ClipAgainst(tempVertex,zwn,SharedP);
	ClipAgainst(tempVertex,znwn,SharedP);

	for (size_t i = 0; i < tempVertex.size(); ++i) {
		input.push_back(tempVertex[i]);
	}

}

void RasterizeTri(MGL_FrameBuf &buffer) {
	if (VertexChain.size() % 3 != 0)
		MGL_ERROR("FATAL: Wrong Vertex counts, Gen by RasterizeTri");

	for (size_t i = 0; i < VertexChain.size(); i += 3) {
		FillTri(buffer,
				Norm_ViewPort(VertexChain[i], buffer.width, buffer.height),
				Norm_ViewPort(VertexChain[i + 1], buffer.width, buffer.height),
				Norm_ViewPort(VertexChain[i + 2], buffer.width, buffer.height));
	}

}

vec4 Mat_Ver(MGLMatrix matrix, vec4 vertex) {
	vec4 ans;
	int n = 4, m = 4;
	for (int i = 0; i < n; i++) {
		ans[i] = 0;
		for (int k = 0; k < m; ++k) {
			ans[i] += matrix.m[k * n + i] * vertex[k];
		}
	}
	return ans;
}

/**
 * Read pixel data starting with  pixel at coordinates
 * (0, 0), up to (width,  height), into  array
 * pointed to by data.   boundaries are lower-inclusive,
 * that is, a call with width = height = 1 would just read
 *  pixel at (0, 0).
 *
 * Rasterization and z-buffering should be performed when
 * this function is called, so that  data array is filled
 * with  actual pixel values that should be displayed on
 *  two-dimensional screen.
 */
void mglReadPixels(MGLsize width, MGLsize height, MGLpixel *data) {
	if (started)
		MGL_ERROR("GL_INVALID_OPERATION, Gen by mglReadPixels");

	Width = width;
	Height = height;
	MGL_FrameBuf Buffer(width, height);
	RasterizeTri(Buffer);
	for (MGLsize i = 0; i < width; i++) {
		for (MGLsize j = 0; j < height; ++j) {
			//DANGEROUS~
			data[Buffer.pos(i, j)] = Buffer.GetOne(i, j);
		}
	}

}

/**
 * Start specifying  vertices for a group of primitives,
 * whose type is specified by  given mode.
 */
void mglBegin(MGLpoly_mode mode) {
	if (started)
		MGL_ERROR("GL_INVALID_OPERATION, Gen by mglBegin");

	started = true;
	poly = mode;
}

/**
 * Stop specifying  vertices for a group of primitives.
 */
void mglEnd() {
	if (!started)
		MGL_ERROR("GL_INVALID_OPERATION, Gen by mglEnd");

	VertexChain_Buffer.clear();
	started = false;
}

/**
 * Specify a two-dimensional vertex;  x- and y-coordinates
 * are explicitly specified, while  z-coordinate is assumed
 * to be zero.  Must appear between calls to mglBegin() and
 * mglEnd().
 */
void mglVertex2(MGLfloat x, MGLfloat y) {
	mglVertex3(x, y, 0);
}

/**
 * Specify a three-dimensional vertex.  Must appear between
 * calls to mglBegin() and mglEnd().
 */
void mglVertex3(MGLfloat x, MGLfloat y, MGLfloat z) {
	if (!started)
		MGL_ERROR("Wrong, Gen by mglVertex3");

	MGL_Ver newOne(x, y, z, Color);
	//https://www.khronos.org/opengl/wiki/Vertex_Transformation
	//Transform
	newOne.Vec = Mat_Ver(MMatrix[MGL_MODELVIEW], newOne.Vec);
	//Clip
	newOne.Vec = Mat_Ver(MMatrix[MGL_PROJECTION], newOne.Vec);

	//Normalize (devide w)
	//newOne.Vec = newOne.Vec / newOne.Vec[3];

	VertexChain_Buffer.push_back(newOne);

	if (poly == MGL_TRIANGLES) {
		if (VertexChain_Buffer.size() == 3) {
			ClipAndPush(VertexChain, VertexChain_Buffer[0],
					VertexChain_Buffer[1], VertexChain_Buffer[2]);
			VertexChain_Buffer.clear();
		}
	} else {
		if (VertexChain_Buffer.size() == 4) {
			//First, CCW
			ClipAndPush(VertexChain, VertexChain_Buffer[0],
					VertexChain_Buffer[1], VertexChain_Buffer[2]);
			//Second, CCW
			ClipAndPush(VertexChain, VertexChain_Buffer[0],
					VertexChain_Buffer[2], VertexChain_Buffer[3]);
			VertexChain_Buffer.clear();
		}
	}

}

/**
 * Set  current matrix mode (modelview or projection).
 */
void mglMatrixMode(MGLmatrix_mode mode) {
	Mode = mode;
}

/**
 * Push a copy of  current matrix onto  stack for 
 * current matrix mode.
 */
void mglPushMatrix() {
	if (started) {
		MGL_ERROR("GL_INVALID_OPERATION ,Gen by mglPushMatrix");
	}

	if (Mode == MGL_MODELVIEW) {
		MV_MatrixStack.push(MMatrix[MGL_MODELVIEW]);
	} else if (Mode == MGL_PROJECTION) {
		PROJ_MatrixStack.push(MMatrix[MGL_PROJECTION]);
	} else {
		MGL_ERROR("CRITICAL, gen by mglPushMatrix");
	}
}

/**
 * Pop  top matrix from  stack for  current matrix
 * mode.
 */
void mglPopMatrix() {
	if (started) {
		MGL_ERROR("GL_INVALID_OPERATION ,Gen by mglPopMatrix");
	}

	if (Mode == MGL_MODELVIEW) {
		if (MV_MatrixStack.size() < 1) {
			MGL_ERROR("GL_STACK_UNDERFLOW ,Gen by mglPopMatrix");
		} else {
			MMatrix[MGL_MODELVIEW] = MV_MatrixStack.top();
			MV_MatrixStack.pop();
		}
	} else if (Mode == MGL_PROJECTION) {
		if (PROJ_MatrixStack.size() < 1) {
			MGL_ERROR("GL_STACK_UNDERFLOW ,Gen by mglPopMatrix");
		} else {
			MMatrix[MGL_PROJECTION] = PROJ_MatrixStack.top();
			PROJ_MatrixStack.pop();
		}
	} else {
		MGL_ERROR("CRITICAL, gen by mglPopMatrix");
	}

}

/**
 * Replace  current matrix with  identity.
 */
void mglLoadIdentity() {
	for (int i = 0; i < 16; i++)
		MMatrix[Mode].m[i] = 0;
	MMatrix[Mode].m[0] = 1;
	MMatrix[Mode].m[5] = 1;
	MMatrix[Mode].m[10] = 1;
	MMatrix[Mode].m[15] = 1;
}

/**
 * Replace  current matrix with an arbitrary 4x4 matrix,
 * specified in column-major order.  That is,  matrix
 * is stored as:
 *
 *   ( a0  a4  a8  a12 )
 *   ( a1  a5  a9  a13 )
 *   ( a2  a6  a10 a14 )
 *   ( a3  a7  a11 a15 )
 *
 * where ai is  i'th entry of  array.
 */
void mglLoadMatrix(const MGLfloat *matrix) {
	if (started) {
		MGL_ERROR("GL_INVALID_OPERATION, Gen by mglLoadMatrix");
	}
	for (int i = 0; i < 16; ++i) {
		MMatrix[Mode].m[i] = matrix[i];
	}
}

/**
 * Multiply  current matrix by an arbitrary 4x4 matrix,
 * specified in column-major order.  That is,  matrix
 * is stored as:
 *
 *   ( a0  a4  a8  a12 )
 *   ( a1  a5  a9  a13 )
 *   ( a2  a6  a10 a14 )
 *   ( a3  a7  a11 a15 )
 *
 * where ai is  i'th entry of  array.
 */
void mglMultMatrix(const MGLfloat *matrix) {
	int n = 4, m = 4, p = 4;
	MGLMatrix old(MMatrix[Mode]);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			MMatrix[Mode].m[j * n + i] = 0;
			for (int k = 0; k < m; ++k) {
				MMatrix[Mode].m[j * n + i] += old.m[k * n + i]
						* matrix[j * p + k];
			}
		}
	}
}

/**
 * Multiply  current matrix by  translation matrix
 * for  translation vector given by (x, y, z).
 */
void mglTranslate(MGLfloat x, MGLfloat y, MGLfloat z) {
	if (started) {
		MGL_ERROR("GL_INVALID_OPERATION, Gen by mglTranslate");
	}

	MGLMatrix TranslateMatrix;
	TranslateMatrix.m[12] = x;
	TranslateMatrix.m[13] = y;
	TranslateMatrix.m[14] = z;
	mglMultMatrix(TranslateMatrix.m);

}

/**
 * Multiply  current matrix by  rotation matrix
 * for a rotation of (angle) degrees about  vector
 * from  origin to  point (x, y, z).
 */
void mglRotate(MGLfloat angle, MGLfloat x, MGLfloat y, MGLfloat z) {
	if (started) {
		MGL_ERROR("GL_INVALID_OPERATION, Gen by mglTranslate");
	}

	const MGLfloat pi = 3.1415926535897932384626433832795;
	vec3 t(vec3(x, y, z).normalized());
	x = t[0];
	y = t[1];
	z = t[2];
	MGLfloat c = cos(angle / 180.0 * pi);
	MGLfloat s = sin(angle / 180.0 * pi);

	MGLMatrix RotateMatrix;
	RotateMatrix.m[0] = x * x * (1 - c) + c;
	RotateMatrix.m[1] = y * x * (1 - c) + z * s;
	RotateMatrix.m[2] = x * z * (1 - c) - y * s;
	RotateMatrix.m[4] = x * y * (1 - c) - z * s;
	RotateMatrix.m[5] = y * y * (1 - c) + c;
	RotateMatrix.m[6] = y * z * (1 - c) + x * s;
	RotateMatrix.m[8] = x * z * (1 - c) + y * s;
	RotateMatrix.m[9] = y * z * (1 - c) - x * s;
	RotateMatrix.m[10] = z * z * (1 - c) + c;
	mglMultMatrix(RotateMatrix.m);

}

/**
 * Multiply  current matrix by  scale matrix
 * for  given scale factors.
 */
void mglScale(MGLfloat x, MGLfloat y, MGLfloat z) {
	if (started) {
		MGL_ERROR("GL_INVALID_OPERATION, Gen by mglScale");
	}

	MGLMatrix ScaleMatrix;
	ScaleMatrix.m[0] = x;
	ScaleMatrix.m[5] = y;
	ScaleMatrix.m[10] = z;
	mglMultMatrix(ScaleMatrix.m);

}

/**
 * Multiply  current matrix by  perspective matrix
 * with  given clipping plane coordinates.
 */
void mglFrustum(MGLfloat left, MGLfloat right, MGLfloat bottom, MGLfloat top,
		MGLfloat near, MGLfloat far) {

	//https://msdn.microsoft.com/en-us/library/windows/desktop/dd373537(v=vs.85).aspx
	MGLMatrix FrustumMatrix;
	FrustumMatrix.m[0] = (2.0 * near) / (right - left);
	FrustumMatrix.m[5] = (2.0 * near) / (top - bottom);
	FrustumMatrix.m[8] = (right + left) / (right - left);
	FrustumMatrix.m[9] = (top + bottom) / (top - bottom);
	FrustumMatrix.m[10] = (far + near) / (near - far);
	FrustumMatrix.m[11] = -1;
	FrustumMatrix.m[14] = (2.0 * far * near) / (near - far);
	FrustumMatrix.m[15] = 0;

	mglMultMatrix(FrustumMatrix.m);
}

/**
 * Multiply  current matrix by  orthographic matrix
 * with  given clipping plane coordinates.
 */
void mglOrtho(MGLfloat left, MGLfloat right, MGLfloat bottom, MGLfloat top,
		MGLfloat near, MGLfloat far) {
			if ((left == right) || (bottom == top) || (near == far)) {
		MGL_ERROR("GL_INVALID_VALUE, Gen by mglOrtho");
	}
	if (started) {
		MGL_ERROR("GL_INVALID_OPERATION");
	}

	MGLMatrix OrthoMatrix;
	OrthoMatrix.m[0] = 2.0 / (right - left);
	OrthoMatrix.m[5] = 2.0 / (top - bottom);
	OrthoMatrix.m[10] = (-2.0) / (far - near);
	OrthoMatrix.m[12] = (right + left) / (left - right);
	OrthoMatrix.m[13] = (top + bottom) / (bottom - top);
	OrthoMatrix.m[14] = (far + near) / (near - far);
	mglMultMatrix(OrthoMatrix.m);
			
}

/**
 * Set  current color for drawn shapes.
 */
void mglColor(MGLfloat red, MGLfloat green, MGLfloat blue) {
	Color[0] = red;
	Color[1] = green;
	Color[2] = blue;
}
