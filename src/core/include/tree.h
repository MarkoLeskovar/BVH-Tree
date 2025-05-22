#ifndef TREE_H
#define TREE_H

#include "types.h"
#include "query.h"
#include "mesh.h"

#include <vector>
#include <array>
#include <tuple>


/*
O-----------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX                                           |
O-----------------------------------------------------------------------------O
*/

class AABB {

private:
    Vec3 m_min;
    Vec3 m_max;

public:
    // Constructors
    AABB();
    explicit AABB(const std::vector<Vec3>& points);

    // Getters
    [[nodiscard]] Vec3 min() const { return m_min; }
    [[nodiscard]] Vec3 max() const { return m_max; }
    [[nodiscard]] Vec3 size() const { return m_max - m_min; }
    [[nodiscard]] Vec3 center() const { return m_min + 0.5 * (m_max - m_min); }
    [[nodiscard]] double diagonal() const { return (m_max - m_min).norm(); }
    [[nodiscard]] int largest_axis() const;

    // Member functions
    void reset();
    void grow(const Vec3& point);
    void merge(const AABB& other);
    void expand(double delta);
};


/*
O-----------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX - NODE                                    |
O-----------------------------------------------------------------------------O
*/


class AABBNode {
    friend class AABBTree;

private:
    AABB m_aabb;
//    int m_parent_id = -1;
    int m_child_id = -1;
    int m_depth = 0;
    int m_face_index = -1;
    int m_face_count = -1;

public:
    // Getters
    [[nodiscard]] const AABB& aabb() const { return m_aabb; }
//    [[nodiscard]] int parent_id() const { return m_parent_id; }
    [[nodiscard]] int depth() const { return m_depth; }
    [[nodiscard]] int face_index() const { return m_face_index; }
    [[nodiscard]] int face_count() const { return m_face_count; }

    // Special getters
    [[nodiscard]] int left_child_id() const { return m_child_id; }
    [[nodiscard]] int right_child_id() const { return m_child_id + 1; }
    [[nodiscard]] bool is_leaf() const {return m_child_id == -1; }
//    [[nodiscard]] bool is_root() const {return m_parent_id == -1; }
};


/*
O-----------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX - TREE                                    |
O-----------------------------------------------------------------------------O
*/

class AABBTree {

private:
    const TriangleMesh& m_mesh;
    std::vector<AABBNode> m_nodes;
    std::vector<size_t> m_face_ids;
    std::unique_ptr<KDTree> m_kdtree;
    int m_max_depth = 0;

public:
    // Constructors
    AABBTree() = delete;
    AABBTree(const TriangleMesh& mesh, int depth_lim, int split_lim);

    // Getters
    [[nodiscard]] int max_depth() const { return m_max_depth; }
    [[nodiscard]] const TriangleMesh& mesh() const { return m_mesh; }
    [[nodiscard]] const std::vector<size_t>& face_ids() const { return m_face_ids; }
    [[nodiscard]] const std::vector<AABBNode>& nodes() const { return m_nodes; }

    // Special getters
    [[nodiscard]] std::vector<AABBNode> get_leaf_nodes() const;
    [[nodiscard]] std::vector<AABBNode> get_nodes_at_depth(int depth) const;

    // Closest point queries -> TREE O(n*log(n)) approach
    [[nodiscard]] QueryResult query_closest_point(const Vec3& point) const;
    [[nodiscard]] std::vector<QueryResult> query_closest_points(const std::vector<Vec3>& points, int workers=1) const;

private:
    // Build trees
    void build_aabb_tree(const int &depth_lim, const int &split_lim);
    void build_kd_tree(int max_leaf_size=10);

    // Build AABB internals
    [[nodiscard]] AABB create_aabb(const size_t& face_index, const size_t& face_count) const;
    [[nodiscard]] std::array<AABBNode, 2> split_middle(const AABBNode& node, const std::vector<Vec3>& all_face_centers);

    // Queries
    [[nodiscard]] std::tuple<double, size_t> kdtree_query(const Vec3& point) const;
    [[nodiscard]] std::vector<size_t> ball_query(const Vec3& ball_center, double ball_radius) const;
    [[nodiscard]] QueryResult find_closest_face(const Vec3 &point, const std::vector<size_t>& node_ids) const;
};


#endif //TREE_H