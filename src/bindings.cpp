#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "types.h"
#include "query.h"
#include "mesh.h"
#include "tree.h"
#include "geometry.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


PYBIND11_MODULE(core, m) {
    m.doc() = R"pbdoc(
        Bounding Volume Hierarchy (BVH) Tree.        
    )pbdoc";


    py::class_<QueryResult>(m, "QueryResult")
        // Constructors
        .def(py::init<>())
        // Getters (property)
        .def_property_readonly("point", &QueryResult::point)
        .def_property_readonly("distance", &QueryResult::distance)
        .def_property_readonly("face_id", &QueryResult::face_id);


    py::class_<Triangle>(m, "Triangle")
        // Constructors
        .def(py::init<>())
        .def(py::init<Vec3, Vec3, Vec3>(), py::arg("v0"), py::arg("v1"), py::arg("v2"))
        .def(py::init<const std::initializer_list<Vec3>&>(), py::arg("vertices"))
        // Member variables
        .def_readwrite("v0", &Triangle::v0)
        .def_readwrite("v1", &Triangle::v1)
        .def_readwrite("v2", &Triangle::v2)
        // Member functions
        .def("compute_normal", &Triangle::compute_normal)
        .def("compute_center", &Triangle::compute_center)
        .def("compute_area", &Triangle::compute_area);


    py::class_<TriangleMesh>(m, "TriangleMesh_core")
        // Constructors
        .def(py::init<const std::vector<Vec3>&, const std::vector<VecIds3>&>(), py::arg("vertices"), py::arg("faces"))
        // Modifiable property
        .def_property("vertices",
            (std::vector<Vec3>& (TriangleMesh::*)()) &TriangleMesh::vertices,
            [](TriangleMesh& self, const std::vector<Vec3>& verts) { self.vertices() = verts; },
            py::return_value_policy::reference_internal)
        // Getters (property)
        .def_property_readonly("faces", &TriangleMesh::faces, py::return_value_policy::reference_internal)
        .def_property_readonly("num_vertices", &TriangleMesh::num_vertices)
        .def_property_readonly("num_faces", &TriangleMesh::num_faces)
        // Special getters
        .def("get_vertex", &TriangleMesh::get_vertex, py::arg("id"), py::return_value_policy::reference_internal)
        .def("get_face", &TriangleMesh::get_face, py::arg("id"), py::return_value_policy::reference_internal)
        .def("get_triangle", &TriangleMesh::get_triangle, py::arg("id"))
        // Member functions
        .def("compute_normals", &TriangleMesh::compute_normals)
        .def("compute_centers", &TriangleMesh::compute_centers)
        .def("compute_areas", &TriangleMesh::compute_areas)
        // In-place transformations
        .def("translate", &TriangleMesh::translate, py::arg("trans_vec"))
        .def("scale", &TriangleMesh::scale, py::arg("scale_vec"))
        .def("rotate", &TriangleMesh::rotate, py::arg("angles"))
        .def("transform", &TriangleMesh::transform, py::arg("trans_mat"))
        // Closest point queries
        .def("query_closest_point", &TriangleMesh::query_closest_point, py::arg("point"))
        .def("query_closest_points", &TriangleMesh::query_closest_points, py::arg("points"), py::arg("workers") = 1);


    py::class_<AABB>(m, "AABB")
        // Constructors
        .def(py::init<>())
        .def(py::init<const std::vector<Vec3>&>(), py::arg("points"))
        // Getters (property)
        .def_property_readonly("min", &AABB::min)
        .def_property_readonly("max", &AABB::max)
        .def_property_readonly("size", &AABB::size)
        .def_property_readonly("center", &AABB::center)
        .def_property_readonly("diagonal", &AABB::diagonal)
        .def_property_readonly("largest_axis", &AABB::largest_axis)
        // Member functions
        .def("reset", &AABB::reset)
        .def("grow", &AABB::grow, py::arg("point"))
        .def("merge", &AABB::merge, py::arg("other"))
        .def("expand", &AABB::expand, py::arg("delta"));


    py::class_<AABBNode>(m, "AABBNode")
        // Constructors
        .def(py::init<>())
        // Getters (property)
        .def_property_readonly("aabb", &AABBNode::aabb)
//        .def_property_readonly("parent_id", &AABBNode::parent_id)
        .def_property_readonly("depth", &AABBNode::depth)
        .def_property_readonly("face_index", &AABBNode::face_index)
        .def_property_readonly("face_count", &AABBNode::face_count)
        // Special getters
        .def("left_child_id", &AABBNode::left_child_id)
        .def("right_child_id", &AABBNode::right_child_id)
        .def("is_leaf", &AABBNode::is_leaf);
//        .def("is_root", &AABBNode::is_root)


    py::class_<AABBTree>(m, "AABBTree_core")
        // Constructors
        .def(py::init<const TriangleMesh&, int, int>(), py::arg("mesh"), py::arg("depth_lim"), py::arg("split_lim"))
        // Getters (property)
        .def_property_readonly("max_depth", &AABBTree::max_depth)
        .def_property_readonly("mesh", [](const AABBTree& self) -> const TriangleMesh& {return self.mesh();}, py::return_value_policy::reference_internal)
        .def_property_readonly("face_ids", [](const AABBTree& self) -> const std::vector<size_t>& {return self.face_ids();}, py::return_value_policy::reference_internal)
        .def_property_readonly("nodes", [](const AABBTree& self) -> const std::vector<AABBNode>& {return self.nodes();}, py::return_value_policy::reference_internal)
        // Special getters
        .def("get_leaf_nodes", &AABBTree::get_leaf_nodes)
        .def("get_nodes_at_depth", &AABBTree::get_nodes_at_depth)
        // Closest point queries
        .def("query_closest_point", &AABBTree::query_closest_point, py::arg("point"))
        .def("query_closest_points", &AABBTree::query_closest_points, py::arg("points"), py::arg("workers") = 1);


    // Closest point on primitive
    m.def("closest_point_on_line", &closest_point_on_line, "Closest point on line segment");
    m.def("closest_point_on_box", &closest_point_on_box, "Closest point on axis-aligned box");
    m.def("closest_point_on_triangle", &closest_point_on_triangle, "Closest point on triangle");


    // Primitive intersections
    m.def("box_sphere_intersection", &box_sphere_intersection, "Box-sphere intersection test");
    m.def("box_triangle_intersection", &box_triangle_intersection, "Box-triangle intersection test");


    // Add version info
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
