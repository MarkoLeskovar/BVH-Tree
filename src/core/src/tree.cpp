#include <numeric>
#include <utility>

#include "types.h"
#include "tree.h"
#include "geometry.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

/*
O-----------------------------------------------------------------------------O
| AXIS ALIGNED BOUNDING BOX                                                   |
O-----------------------------------------------------------------------------O
*/

AABB::AABB() {
    reset();
}


AABB::AABB(const std::vector<Vec3> &points) {
    // Check size
    if (points.empty()) {
        throw std::invalid_argument("Cannot construct AABB from an empty point set.");
    }
    // Add points
    reset();
    for (auto & point : points) {
        grow(point);
    }
}


int AABB::largest_axis() const {
    int index;
    (m_max - m_min).maxCoeff(&index);
    return index;
}


void AABB::reset() {
    constexpr double min_val = std::numeric_limits<double>::lowest();
    constexpr double max_val = std::numeric_limits<double>::max();
    m_min = Vec3(max_val, max_val, max_val);
    m_max = Vec3(min_val, min_val, min_val);
}


void AABB::grow(const Vec3 &point) {
    m_min = m_min.cwiseMin(point);
    m_max = m_max.cwiseMax(point);
}


void AABB::merge(const AABB& other) {
    m_min = m_min.cwiseMin(other.m_min);
    m_max = m_max.cwiseMax(other.m_max);
}


void AABB::expand(const double delta) {
    const double half_delta = 0.5 * delta;
    m_min = m_min.array() - half_delta;
    m_max = m_max.array() + half_delta;
}


/*
O-----------------------------------------------------------------------------O
| IMPLEMENTATION - AXIS ALIGNED BOUNDING BOX - TREE                           |
O-----------------------------------------------------------------------------O
*/


AABBTree::AABBTree(const TriangleMesh &mesh, int depth_lim, int split_lim) : m_mesh(mesh) {
    build_kd_tree(10);
    build_aabb_tree(depth_lim, split_lim);
}


std::vector<AABBNode> AABBTree::get_leaf_nodes() const {
    std::vector<AABBNode> nodes;
    nodes.reserve(m_nodes.size());
    for (const auto& node : m_nodes) {
        if (node.is_leaf()) {
            nodes.push_back(node);
        }
    }
    nodes.shrink_to_fit();
    return nodes;
}


std::vector<AABBNode> AABBTree::get_nodes_at_depth(int depth) const {
    std::vector<AABBNode> nodes;
    nodes.reserve(m_nodes.size());
    for (const auto & node : m_nodes) {
        if (node.depth() == depth) {
            nodes.push_back(node);
        }
    }
    nodes.shrink_to_fit();
    return nodes;
}


QueryResult AABBTree::query_closest_point(const Vec3 &point) const {
    const double init_distance = std::get<0>(kdtree_query(point));
    const std::vector<size_t> node_ids = ball_query(point, init_distance);
    return find_closest_face(point, node_ids);
}


std::vector<QueryResult> AABBTree::query_closest_points(const std::vector<Vec3>& points, int workers) const {
    std::vector<QueryResult> results(points.size());
#ifdef USE_OPENMP
    omp_set_num_threads(workers);
    #pragma omp parallel for
#endif
    for (long long int i = 0; i < points.size(); ++i) {
        results[i] = query_closest_point(points[i]);
    }
    return results;
}


std::tuple<double, size_t> AABBTree::kdtree_query(const Vec3 &point) const {
    // Initialize the data
    size_t point_index = 0;
    double squared_distance = 0.0;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&point_index, &squared_distance);
    // Find the closest point
    m_kdtree->index->findNeighbors(resultSet, &point[0]);
    // Find the Euclidean distance
    const double distance = std::sqrt(squared_distance);
    // Return results
    return {distance, point_index};
}


void AABBTree::build_aabb_tree(const int &depth_lim, const int &split_lim) {
    m_max_depth = 0;
    const int max_nodes = std::pow(2, depth_lim) - 1;

    // Compute face centers
    const std::vector<Vec3> all_face_centers = m_mesh.compute_centers();

    // Initialize all face ids
    m_face_ids.clear();
    m_face_ids.resize(m_mesh.num_faces());
    std::iota(m_face_ids.begin(), m_face_ids.end(), 0);

    // Initialize tree nodes
    m_nodes.clear();
    m_nodes.reserve(max_nodes);

    // Initialize the root node
    AABBNode root_node;
    root_node.m_depth = 1;
    root_node.m_face_index = 0;
    root_node.m_face_count = m_mesh.num_faces();
    root_node.m_aabb = create_aabb(0, m_mesh.num_faces());

    // Add root node to the list
    m_nodes.push_back(root_node);

    // Initialize the stack
    std::vector<int> node_id_stack(depth_lim);

    // Iteration counters
    int list_size = 1;
    int stack_size = 1;

    // Add the nodes
    while (stack_size != 0) {
        stack_size -= 1;
        const int node_id = node_id_stack[stack_size];

        // Get the current data
        const int node_depth = m_nodes[node_id].m_depth;
        const int node_face_count = m_nodes[node_id].m_face_count;

        // Check splitting condition
        if (node_depth < depth_lim && node_face_count > split_lim) {

            // Update the node's child index
            m_nodes[node_id].m_child_id = list_size;

            // Update max depth
            const int new_depth = node_depth + 1;
            if (new_depth > m_max_depth) {
                m_max_depth = new_depth;
            }

            // Split the node into two children ([0] -> left child, [1] -> right child)
            std::array<AABBNode, 2> child_nodes = split_middle(m_nodes[node_id], all_face_centers);

            // Add parent ids
//            child_nodes[0].m_parent_id = node_id;
//            child_nodes[1].m_parent_id = node_id;

            // Add the nodes
            m_nodes.push_back(child_nodes[0]);
            m_nodes.push_back(child_nodes[1]);

            // Add to stack (process left node first)
            node_id_stack[stack_size] = list_size + 1;
            node_id_stack[stack_size + 1] = list_size;

            // Update counters
            list_size += 2;
            stack_size += 2;
        }
    }
    m_nodes.shrink_to_fit();
}

void AABBTree::build_kd_tree(int max_leaf_size) {
    m_kdtree = std::make_unique<KDTree>(3, m_mesh.vertices(), max_leaf_size);
}


QueryResult AABBTree::find_closest_face(const Vec3 &point, const std::vector<size_t>& node_ids) const {

    // Initialize output
    QueryResult result;
    result.m_distance = std::numeric_limits<double>::max();

    // Find the closest face
    for (int i = 0; i < node_ids.size(); ++i) {
        const AABBNode &node = m_nodes[node_ids[i]];

        // Loop over all triangles
        for (int j = node.m_face_index; j < node.m_face_index + node.m_face_count; ++j) {

            // Get vertex ids
            const size_t &face_id = m_face_ids[j];
            const VecIds3 &face = m_mesh.faces()[face_id];

            // Get vertices
            const Vec3 &v0 = m_mesh.vertices()[face[0]];
            const Vec3 &v1 = m_mesh.vertices()[face[1]];
            const Vec3 &v2 = m_mesh.vertices()[face[2]];

            // Get distance to triangle
            const Vec3 &temp_point = closest_point_on_triangle(point, v0, v1, v2);
            const double temp_distance = (temp_point - point).array().square().sum();

            // Update the distance
            if (temp_distance < result.m_distance) {
                result.m_distance = temp_distance;
                result.m_point = temp_point;
                result.m_face_id = face_id;
            }
        }
    }
    // Normalize the distance
    result.m_distance = std::sqrt(result.m_distance);

    // Return results
    return result;
}


AABB AABBTree::create_aabb(const size_t& face_index, const size_t& face_count) const {
    // Initialize the output
    AABB aabb;
    // Loop over all element indices
    for (size_t i = face_index; i < face_index + face_count; ++i) {
        const VecIds3 &element = m_mesh.get_face(m_face_ids[i]);
        // Loop over all triangle vertices
        for (int j = 0; j < 3; ++j) {
            const Vec3 &vertex = m_mesh.get_vertex(element[j]);
            aabb.grow(vertex);
        }
    }
    return aabb;
}


std::array<AABBNode, 2> AABBTree::split_middle(const AABBNode& node, const std::vector<Vec3>& all_face_centers) {

    // Get split axis (reuse aabb) - FASTER
    const int split_axis = node.m_aabb.largest_axis();

    // Get split axis (face centroid) - ORIGINAL
    // AABB aabb;
    // for (size_t i = node.m_face_index; i < node.m_face_index + node.m_face_count; ++i) {
    //     const size_t& face_id = m_face_ids[i];
    //     const Vec& face_center = all_face_centers[face_id];
    //     aabb.grow(face_center);
    // }
    // const int split_axis = aabb.largest_axis();

    // Get split position value
    double sum = 0.0;
    for (size_t i = node.m_face_index; i < node.m_face_index + node.m_face_count; ++i) {
        const size_t& face_id = m_face_ids[i];
        const Vec3& face_center = all_face_centers[face_id];
        const double center = face_center[split_axis];
        sum += center;
    }
    const double split_pos = sum / node.m_face_count;

    // Initialize sorting indices
    const int face_index_left = node.m_face_index;
    int face_count_left = 0;
    int face_index_right = node.m_face_index;
    int face_count_right = 0;

    // Partial sort the "m_all_face_ids"
    for (size_t i = node.m_face_index; i < node.m_face_index + node.m_face_count; ++i) {
        // If face is on the left
        if (all_face_centers[m_face_ids[i]][split_axis] < split_pos) {
            // Swap faces if index differs
            const int i_swap = face_index_left + face_count_left;
            if (i != i_swap) {
                std::swap(m_face_ids[i], m_face_ids[i_swap]);
            }
            // Update indices
            face_index_right += 1;
            face_count_left += 1;

        }
        // If face is on the right
        else {
            face_count_right += 1;
        }
    }

    // Create child nodes
    std::array<AABBNode, 2> child_nodes;

    // Left node (id = 0);
    child_nodes[0].m_depth = node.m_depth + 1;
    child_nodes[0].m_face_index = face_index_left;
    child_nodes[0].m_face_count = face_count_left;
    child_nodes[0].m_aabb = create_aabb(face_index_left, face_count_left);

    // Right node (id = 1);
    child_nodes[1].m_depth = node.m_depth + 1;
    child_nodes[1].m_face_index = face_index_right;
    child_nodes[1].m_face_count = face_count_right;
    child_nodes[1].m_aabb = create_aabb(face_index_right, face_count_right);

    // Return results
    return child_nodes;
}

std::vector<size_t> AABBTree::ball_query(const Vec3 &ball_center, const double ball_radius) const {

    // Initialize the stack
    std::vector<int> node_id_stack(m_max_depth);
    std::vector<size_t> node_id_list;
    node_id_list.reserve(m_nodes.size());

    // Add root node
    node_id_stack[0] = 0;

    // Stack counter
    int stack_size = 1;

    // Navigate the tree
    while (stack_size != 0) {
        stack_size -= 1;
        const int node_id = node_id_stack[stack_size];

        // Get the current data
       const AABBNode &node = m_nodes[node_id];

        // Check if we have a leaf node
        if (node.is_leaf()) {
            node_id_list.push_back(node_id);
        }
        else {
            // Check left child
            const AABBNode &left_child = m_nodes[node.left_child_id()];
            if (box_sphere_intersection(left_child.m_aabb.min(), left_child.m_aabb.max(), ball_center, ball_radius)) {
                node_id_stack[stack_size] = node.left_child_id();
                stack_size += 1;
            }
            // Check right child
            const AABBNode &right_child = m_nodes[node.right_child_id()];
            if (box_sphere_intersection(right_child.m_aabb.min(), right_child.m_aabb.max(), ball_center, ball_radius)) {
                node_id_stack[stack_size] = node.right_child_id();
                stack_size += 1;
            }
        }
    }
    // Return results
    node_id_list.shrink_to_fit();
    return node_id_list;
}

