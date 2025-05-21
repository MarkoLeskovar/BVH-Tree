#include "geometry.h"

#include <numbers>

#define EPS 1e-9

Vec3 UNIT_VEC_X(1.0, 0.0, 0.0);
Vec3 UNIT_VEC_Y(0.0, 1.0, 0.0);
Vec3 UNIT_VEC_Z(0.0, 0.0, 1.0);


Vec3 closest_point_on_line(const Vec3& point, const Vec3& a, const Vec3& b) {
    // Edges vectors
    const Vec3 ab = b - a;
    const Vec3 ap = point - a;
    // Distance alog a segment
    const double denominator = ab.dot(ab) + EPS;
    const double proj_dist = ap.dot(ab) / denominator;
    // Select correct line region
    if (proj_dist <= 0.0) { return a; }
    if (proj_dist >= 1.0) { return b; }
    return a + proj_dist * ab;
}


Vec3 closest_point_on_triangle(const Vec3 &point, const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    // Edge vectors
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = point - a;

    // Case 1 -> point a
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 and d2 <= 0.0) {
        return a;
    }

    // Case 2 -> point b
    const Vec3 bp = point - b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 and d4 <= d3) {
        return b;
    }

    // Case 3 -> point c
    const Vec3 cp = point - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 and d5 <= d6) {
        return c;
    }

    // Case 4 -> edge ab
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return a + v * ab;
    }

    // Case 5 -> edge ac
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0) {
        const double v = d2 / (d2 - d6);
        return a + v * ac;
    }

    // Case 6 -> edge bc
    const double va = d3 * d6 - d5 * d4;
    const double m = d4 - d3;
    if (va <= 0.0 and m >= 0.0 and d5 - d6 >= 0.0) {
        double const v = m / (m + (d5 - d6));
        return b + v * (c - b);
    }

    // Case 7 -> inside
    const double denominator = 1.0 / (va + vb + vc);
    const double v = vb * denominator;
    const double w = vc * denominator;
    return a + v * ab + w * ac;
}


Vec3 closest_point_on_box(const Vec3 &point, const Vec3 &box_min, const Vec3 &box_max) {

    // Inside-outside test
    const bool is_inside_x = box_min[0] <= point[0] and point[0] <= box_max[0];
    const bool is_inside_y = box_min[1] <= point[1] and point[1] <= box_max[1];
    const bool is_inside_z = box_min[2] <= point[2] and point[2] <= box_max[2];

    // Find the closest point inside the box
    if (is_inside_x and is_inside_y and is_inside_z) {
        const Vec3 t = (point - box_min).cwiseQuotient(box_max - box_min);
        const Vec3 temp = (t.array() - 0.5).abs();
        Eigen::Index axis;
        temp.maxCoeff(&axis);

        // Assign correct point
        Vec3 closest_point = point;
        if (t[axis] >= 0.5) {
            closest_point[axis] = box_max[axis];
        }
        else {
            closest_point[axis] = box_min[axis];
        }
        return closest_point;
    }

    // Outside the box
    return box_min.cwiseMax(point.cwiseMin(box_max));
}



bool box_sphere_intersection(const Vec3& box_min, const Vec3& box_max, const Vec3& sphere_center, const double& sphere_radius) {
    const double r2 = sphere_radius * sphere_radius;
    double d_min = 0.0;
    for (int i = 0; i < 3; ++i) {
        if (sphere_center[i] < box_min[i]) {
            d_min += std::pow(sphere_center[i] - box_min[i], 2);
        }
        else if (sphere_center[i] > box_max[i]) {
            d_min += std::pow(sphere_center[i] - box_max[i], 2);
        }
    }
    return d_min < r2;
}


bool separating_axis_theorem(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& box_extent, const Vec3& axis) {

    // Project points onto the provided axis
    const double p0 = a.dot(axis);
    const double p1 = b.dot(axis);
    const double p2 = c.dot(axis);

    // Length of the box projection ont a single axis
    const double r = (
        box_extent[0] * abs(UNIT_VEC_X.dot(axis)) +
        box_extent[1] * abs(UNIT_VEC_Y.dot(axis)) +
        box_extent[2] * abs(UNIT_VEC_Z.dot(axis)));

    // Find min and max of the projected triangle to create a line (p_min -> p_max)
    const double p_max = std::max(p0, std::max(p1, p2));
    const double p_min = std::min(p0, std::min(p1, p2));

    // Check if the line (0 -> r) overlaps the line (p_min -> p_max)
    return std::max(-p_max, p_min) > r;
}


bool box_triangle_intersection(const Vec3 &box_min, const Vec3 &box_max, const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    const Vec3 box_extent = 0.5 * (box_max - box_min).array();
    const Vec3 box_center = box_max - box_extent;

    // Center the triangle
    const Vec3 cent_a = a - box_center;
    const Vec3 cent_b = b - box_center;
    const Vec3 cent_c = c - box_center;

    // Triangle edge vectors
    Vec3 ab = cent_b - cent_a;
    Vec3 bc = cent_c - cent_b;
    Vec3 ca = cent_a - cent_c;

    // Normalize the edge vectors
    ab.normalize();
    bc.normalize();
    ca.normalize();

    // Cross ab, bc, and ca with (1, 0, 0)
    const Vec3 a00 = {0.0, -ab[2], ab[1]};
    const Vec3 a01 = {0.0, -bc[2], bc[1]};
    const Vec3 a02 = {0.0, -ca[2], ca[1]};

    //  Cross ab, bc, and ca with (0, 1, 0)
    const Vec3 a10 = {ab[2], 0.0, -ab[0]};
    const Vec3 a11 = {bc[2], 0.0, -bc[0]};
    const Vec3 a12 = {ca[2], 0.0, -ca[0]};

    // Cross ab, bc, and ca with (0, 0, 1)
    const Vec3 a20 = {-ab[1], ab[0], 0.0};
    const Vec3 a21 = {-bc[1], bc[0], 0.0};
    const Vec3 a22 = {-ca[1], ca[0], 0.0};

    // Check intersection with 9 axes, 3 AABB face normals and 1 triangle face normal
    const Vec3 axes[13] = {
        a00, a01, a02, a10, a11, a12, a20, a21, a22,
        UNIT_VEC_X, UNIT_VEC_Y, UNIT_VEC_Z, ab.cross(bc)
    };
    for (int i = 0; i < 13; ++i) {
        if (separating_axis_theorem(cent_a, cent_b, cent_c, box_extent, axes[i])) {
            return false;
        }
    }

    // There is an intersection
    return true;
}


Vec3 compute_triangle_normal(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    Vec3 normal = ab.cross(ac);
    normal.normalize();
    return normal;
}


Vec3 compute_triangle_center(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    const Vec3 center = a + b + c;
    return center.array() / 3.0;
}


double compute_triangle_area(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 normal = ab.cross(ac);
    return 0.5 * normal.norm();
}


std::vector<Vec3> compute_triangle_mesh_normals(const std::vector<Vec3> &vertices, const std::vector<VecIds3> &elements) {
    // Initialize output
    std::vector<Vec3> normals(elements.size());
    // Loop over all elements
    for (size_t i = 0; i < elements.size(); ++i) {
        // Get vertices
        const Vec3& v0 = vertices[elements[i][0]];
        const Vec3& v1 = vertices[elements[i][1]];
        const Vec3& v2 = vertices[elements[i][2]];
        // Compute normal
        const Vec3 normal = compute_triangle_normal(v0, v1, v2);
        normals[i] = normal;
    }
    return normals;
}


std::vector<Vec3> compute_triangle_mesh_centers(const std::vector<Vec3> &vertices, const std::vector<VecIds3> &elements) {
    // Initialize output
    std::vector<Vec3> centers(elements.size());
    // Loop over all elements
    for (size_t i = 0; i < elements.size(); ++i) {
        // Get vertices
        const Vec3& v0 = vertices[elements[i][0]];
        const Vec3& v1 = vertices[elements[i][1]];
        const Vec3& v2 = vertices[elements[i][2]];
        // Compute center
        const Vec3 center = compute_triangle_center(v0, v1, v2);
        centers[i] = center;
    }
    return centers;
}


std::vector<double> compute_triangle_mesh_areas(const std::vector<Vec3> &vertices, const std::vector<VecIds3> &elements) {
    // Initialize output
    std::vector<double> areas(elements.size());
    // Loop over all elements
    for (size_t i = 0; i < elements.size(); ++i) {
        // Get vertices
        const Vec3& v0 = vertices[elements[i][0]];
        const Vec3& v1 = vertices[elements[i][1]];
        const Vec3& v2 = vertices[elements[i][2]];
        // Compute area
        const double area = compute_triangle_area(v0, v1, v2);
        areas[i] = area;
    }
    return areas;
}


Mat3 rotation_matrix_rad(const Vec3 &angles) {
    const double cosA = std::cos(angles[0]);
    const double cosB = std::cos(angles[1]);
    const double cosC = std::cos(angles[2]);
    const double sinA = std::sin(angles[0]);
    const double sinB = std::sin(angles[1]);
    const double sinC = std::sin(angles[2]);
    Mat3 output;
    output << cosB*cosC, sinA*sinB*cosC - cosA*sinC, cosA*sinB*cosC + sinA*sinC,
              cosB*sinC, sinA*sinB*sinC + cosA*cosC, cosA*sinB*sinC - sinA*cosC,
              -sinB    , sinA*cosB                 , cosA*cosB;
    return output;
}

Mat3 rotation_matrix_deg(const Vec3 &angles) {
    constexpr double pi_divided_by_180 = std::numbers::pi / 180.0;
    Vec3 angles_rad;
    angles_rad[0] = angles[0] * pi_divided_by_180;
    angles_rad[1] = angles[1] * pi_divided_by_180;
    angles_rad[2] = angles[2] * pi_divided_by_180;
    return rotation_matrix_rad(angles_rad);
}
