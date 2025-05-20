
__all__ = ['AABBNode']


class AABBNode:

    def __init__(self):
        self.box_min = None
        self.box_max = None
        self.parent_id = -1
        self.child_id = -1
        self.depth = 0
        self.face_index = -1
        self.face_count = -1

    @property
    def left_id(self) -> int:
        return self.child_id

    @property
    def right_id(self) -> int:
        return self.child_id + 1

    def is_leaf(self) -> bool:
        return self.child_id == -1

    def is_root(self) -> bool:
        return self.parent_id == -1
