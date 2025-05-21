#ifndef MESH_H
#define MESH_H

#include "query.h"
#include "types.h"
#include <vector>


/*
O-----------------------------------------------------------------------------O
| CLASS - TRIANGLE                                                            |
O-----------------------------------------------------------------------------O
*/

class Triangle {
public:

    Vec3 v0;
    Vec3 v1;
    Vec3 v2;

    // Constructors
    Triangle() = default;
    Triangle(Vec3  v0, Vec3  v1, Vec3  v2);
    Triangle(const std::initializer_list<Vec3>& vertices);

    // Member functions
    [[nodiscard]] Vec3 compute_normal() const;
    [[nodiscard]] Vec3 compute_center() const;
    [[nodiscard]] double compute_area() const;
};


/*
O-----------------------------------------------------------------------------O
| CLASS - TRIANGLE MESH                                                       |
O-----------------------------------------------------------------------------O
*/

class TriangleMesh {

private:
    std::vector<Vec3> m_vertices;
    std::vector<VecIds3> m_faces;

public:
    // Constructors
    TriangleMesh() = delete;
    TriangleMesh(const std::vector<Vec3>& vertices, const std::vector<VecIds3>& faces);

    // Modifiable getters
    std::vector<Vec3>& vertices() {return m_vertices; }

    // Read-only getters
    [[nodiscard]] const std::vector<Vec3>& vertices() const { return m_vertices; }
    [[nodiscard]] const std::vector<VecIds3>& faces() const {return m_faces; }
    [[nodiscard]] size_t num_vertices() const { return m_vertices.size(); }
    [[nodiscard]] size_t num_faces() const { return m_faces.size(); }

    // Special getters
    [[nodiscard]] const Vec3& get_vertex(const size_t &id) const { return m_vertices[id]; }
    [[nodiscard]] const VecIds3& get_face(const size_t &id) const { return m_faces[id]; }
    [[nodiscard]] Triangle get_triangle(const size_t &id) const;

    // Member functions
    [[nodiscard]] std::vector<Vec3> compute_normals() const;
    [[nodiscard]] std::vector<Vec3> compute_centers() const;
    [[nodiscard]] std::vector<double> compute_areas() const;

    // In-place transformation
    void translate(const Vec3& translation);
    void scale(const Vec3& scale);
    void rotate(const Vec3& angles);
    void transform(const Mat4& transform);

    // Closest point queries
    [[nodiscard]] QueryResult query_closest_point(const Vec3& point) const;
    [[nodiscard]] std::vector<QueryResult> query_closest_points(const std::vector<Vec3>& points, int workers=1) const;

};

#endif //MESH_H