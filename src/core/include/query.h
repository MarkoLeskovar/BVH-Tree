#ifndef QUERY_H
#define QUERY_H

#include "types.h"

/*
O-----------------------------------------------------------------------------O
| CLASS - QUERY RESULT                                                        |
O-----------------------------------------------------------------------------O
*/

class QueryResult {
    friend class AABBTree;
    friend class TriangleMesh;

private:
    Vec3 m_point = {0.0, 0.0, 0.0};
    double m_distance = 0.0;
    size_t m_face_id = 0;

public:
    QueryResult() = default;
    [[nodiscard]] Vec3 const& point() const { return m_point; }
    [[nodiscard]] double distance() const { return m_distance; }
    [[nodiscard]] size_t face_id() const { return m_face_id; }
};

#endif //QUERY_H
