#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "types.h"

// Closest point on primitive
Vec3 closest_point_on_line(const Vec3& point, const Vec3& a, const Vec3& b);
Vec3 closest_point_on_triangle(const Vec3 &point, const Vec3 &a, const Vec3 &b, const Vec3 &c);
Vec3 closest_point_on_box(const Vec3 &point, const Vec3 &box_min, const Vec3 &box_max);

// Primitive intersections
bool box_sphere_intersection(const Vec3& box_min, const Vec3& box_max, const Vec3& sphere_center, const double& sphere_radius);
bool box_triangle_intersection(const Vec3& box_min, const Vec3& box_max, const Vec3& a,  const Vec3& b, const Vec3& c);

// Single triangle operations
Vec3 compute_triangle_normal(const Vec3& a, const Vec3& b, const Vec3& c);
Vec3 compute_triangle_center(const Vec3& a, const Vec3& b, const Vec3& c);
double compute_triangle_area(const Vec3& a, const Vec3& b, const Vec3& c);

// Triangle mesh operations
std::vector<Vec3> compute_triangle_mesh_normals(const std::vector<Vec3>& vertices, const std::vector<VecIds3>& elements);
std::vector<Vec3> compute_triangle_mesh_centers(const std::vector<Vec3>& vertices, const std::vector<VecIds3>& elements);
std::vector<double> compute_triangle_mesh_areas(const std::vector<Vec3>& vertices, const std::vector<VecIds3>& elements);

// Rotation matrices
Mat3 rotation_matrix_rad(const Vec3& angles);
Mat3 rotation_matrix_deg(const Vec3& angles);

#endif //GEOMETRY_H