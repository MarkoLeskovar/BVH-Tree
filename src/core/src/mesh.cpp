#include "types.h"
#include "mesh.h"
#include "geometry.h"
#include "query.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif


/*
O-----------------------------------------------------------------------------O
| TRIANGLE                                                                    |
O-----------------------------------------------------------------------------O
*/

Triangle::Triangle(Vec3 v0, Vec3 v1, Vec3 v2) :
    v0(std::move(v0)), v1(std::move(v1)), v2(std::move(v2)) {
}


Triangle::Triangle(const std::initializer_list<Vec3> &vertices) {
    if (vertices.size() != 3) {
        throw std::invalid_argument("Triangle constructor requires 3 vertices");
    }
    // Assign vertices
    auto it = vertices.begin();
    v0 = *it++;
    v1 = *it++;
    v2 = *it;
}


Vec3 Triangle::compute_normal() const {
    return compute_triangle_normal(v0, v1, v2);
}


Vec3 Triangle::compute_center() const {
    return compute_triangle_center(v0, v1, v2);
}


double Triangle::compute_area() const {
    return compute_triangle_area(v0, v1, v2);
}


/*
O-----------------------------------------------------------------------------O
| TRIANGLE MESH                                                               |
O-----------------------------------------------------------------------------O
*/

TriangleMesh::TriangleMesh(const std::vector<Vec3> &vertices, const std::vector<VecIds3> &faces) {
    this->m_vertices = vertices;
    this->m_faces = faces;
}


Triangle TriangleMesh::get_triangle(const size_t &id) const {
    const Vec3& v0 = m_vertices[m_faces[id][0]];
    const Vec3& v1 = m_vertices[m_faces[id][1]];
    const Vec3& v2 = m_vertices[m_faces[id][2]];
    return {v0, v1, v2};
}


std::vector<Vec3> TriangleMesh::compute_normals() const {
    return compute_triangle_mesh_normals(m_vertices, m_faces);
}


std::vector<Vec3> TriangleMesh::compute_centers() const {
    return compute_triangle_mesh_centers(m_vertices, m_faces);
}


std::vector<double> TriangleMesh::compute_areas() const {
    return compute_triangle_mesh_areas(m_vertices, m_faces);
}


void TriangleMesh::translate(const Vec3 &translation) {
    // Transform every point
    for (size_t i = 0; i < m_vertices.size(); i++) {
        m_vertices[i] += translation;
    }
}


void TriangleMesh::scale(const Vec3 &scale) {
    // Transform every point
    for (size_t i = 0; i < m_vertices.size(); i++) {
        m_vertices[i][0] *= scale[0];
        m_vertices[i][1] *= scale[1];
        m_vertices[i][2] *= scale[2];
    }
}


void TriangleMesh::rotate(const Vec3 &angles) {
    const Mat3 rotation = rotation_matrix_rad(angles);
    for (size_t i = 0; i < m_vertices.size(); i++) {
        m_vertices[i] = rotation * m_vertices[i];
    }
}


void TriangleMesh::transform(const Mat4 &transform) {
    // Extract rotation and translation parts
    const Mat3 rotation = transform.block<3, 3>(0, 0);
    const Vec3 translation = transform.block<3, 1>(0, 3);
    // Transform every point
    for (size_t i = 0; i < m_vertices.size(); i++) {
        m_vertices[i] = rotation * m_vertices[i] + translation;
    }
}


QueryResult TriangleMesh::query_closest_point(const Vec3 &point) const {

    // Initialize output
    QueryResult result;
    result.m_distance = std::numeric_limits<double>::max();

    // Find the closest face
    for (int i = 0; i < m_faces.size(); ++i) {

        // Get vertex ids
        const VecIds3 &face = m_faces[i];

        // Get vertices
        const Vec3 &v0 = m_vertices[face[0]];
        const Vec3 &v1 = m_vertices[face[1]];
        const Vec3 &v2 = m_vertices[face[2]];

        // Get distance to triangle
        const Vec3 &temp_point = closest_point_on_triangle(point, v0, v1, v2);
        const double temp_distance = (temp_point - point).array().square().sum();

        // Update the distance
        if (temp_distance < result.m_distance) {
            result.m_distance = temp_distance;
            result.m_point = temp_point;
            result.m_face_id = i;
        }
    }
    // Normalize the distance
    result.m_distance = std::sqrt(result.m_distance);

    // Return results
    return result;
}


std::vector<QueryResult> TriangleMesh::query_closest_points(const std::vector<Vec3> &points, int workers) const {
    std::vector<QueryResult> results(points.size());

    // Loop over all points
#ifdef USE_OPENMP
    omp_set_num_threads(workers);
#pragma omp parallel for
#endif
    for (long long int i = 0; i < points.size(); ++i) {
        results[i] = query_closest_point(points[i]);
    }
    return results;
}
