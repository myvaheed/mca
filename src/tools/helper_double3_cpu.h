/*
 * helper_double3_cpu.h
 *
 *  Created on: Sep 29, 2017
 *      Author: myvaheed
 */

#ifndef HELPER_DOUBLE3_CPU_H_
#define HELPER_DOUBLE3_CPU_H_

inline  double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline  double length(double3 v)
{
    return sqrt(dot(v, v));
}

inline  double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline  double3 make_double3(double s)
{
    return make_double3(s, s, s);
}
inline  double3 make_double3(double4 a)
{
    return make_double3(a.x, a.y, a.z);
}
inline double4 make_double4(double3 a, double w)
{
    return make_double4(a.x, a.y, a.z, w);
}
inline double3 operator-(double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

inline double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline void operator+=(double3 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline  void operator+=(double4 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline  double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline  void operator-=(double3 &a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline  double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
inline  double3 operator-(double b, double3 a)
{
    return make_double3(b - a.x, b - a.y, b - a.z);
}
inline  void operator-=(double3 &a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}
inline  double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline  void operator*=(double3 &a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline  double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline  double4 operator*(double4 a, double b)
{
    return make_double4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline  double3 operator*(double b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}
inline  void operator*=(double3 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}
inline  void operator*=(double4 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}
inline  double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline  void operator/=(double3 &a, double3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline  double3 operator/(double3 a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline  void operator/=(double3 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline  double3 operator/(double b, double3 a)
{
    return make_double3(b / a.x, b / a.y, b / a.z);
}

#endif /* HELPER_DOUBLE3_CPU_H_ */
