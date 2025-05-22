import numpy as np
import pyvista as pv
from matplotlib import colormaps

from .class_tree import AABBTree
from .utils import nodes_to_pyvista_lines
from ..mesh.class_mesh import TriangleMesh


'''
O------------------------------------------------------------------------------O
| CLASS - AXIS ALIGNED BOUNDING BOX - TREE - DISPLAY                           |
O------------------------------------------------------------------------------O
'''

class AABBTreeDisplay:

    def __init__(self, aabb_tree: AABBTree):
        self._aabb_tree = aabb_tree
        # Initialize variables
        self._cmap = 'jet'
        self._line_width = 3
        self._actors_1 = []
        self._actors_2 = []
        self._active_actors = []
        self._previous_depth = 0
        self._previous_style = 0
        self._depth = 0

    def show(self, cmap='jet', line_width=3):
        self._cmap = str(cmap)
        self._line_width = int(line_width)

        # Select a colormap
        colors = colormaps[self._cmap](np.linspace(0.0, 1.0, self._aabb_tree.max_depth))

        # Add the surface mesh
        pl = pv.Plotter()
        pl.add_mesh(TriangleMesh(self._aabb_tree.faces, self._aabb_tree.vertices).to_pyvista_grid(), color='lightgray')

        # Add all the nodes
        for i in range(self._aabb_tree.max_depth):
            # Get nodes
            nodes = self._aabb_tree.get_nodes_at_depth(i + 1)
            pyvista_nodes = nodes_to_pyvista_lines(nodes)
            pyvista_nodes['color'] = np.repeat(np.linspace(0, 1, len(nodes)), 12)
            # Draw both versions of the nodes
            self._actors_1.append(pl.add_mesh(pyvista_nodes, color=colors[i], line_width=self._line_width))
            self._actors_2.append(pl.add_mesh(pyvista_nodes, scalars='color', cmap=self._cmap, line_width=self._line_width, show_scalar_bar=False))

        # Add slider widget
        pl.add_slider_widget(self._style_callback, rng=[1, 2], value=1, title='Display style',
                             interaction_event='always', pass_widget=True, style='modern', fmt='%.0f',
                             pointa=(0.05, 0.92), pointb=(0.45, 0.92))
        # Add slider widget
        pl.add_slider_widget(self._draw_callback, rng=[1, self._aabb_tree.max_depth], value=1, title='Tree depth',
                             interaction_event='always', pass_widget=True, style='modern', fmt='%.0f',
                             pointa=(0.55, 0.92), pointb=(0.95, 0.92))
        # Show everything
        pl.show()

    def _show_actors(self, actors):
        for i in range(self._depth):
            actors[i].SetVisibility(True)

    def _hide_actors(self, actors):
        for i in range(len(actors)):
            actors[i].SetVisibility(False)

    def _change_opacity(self, actors):
        for i in range(self._depth):
            ratio = (i + 1) / float(self._depth)
            actors[i].prop.opacity = ratio ** 1.5
            actors[i].prop.line_width = self._line_width * ratio ** 1.5

    def _style_callback(self, value, widget):
        # Snap slider to nearest integer value
        self._style = int(round(value))
        widget.GetSliderRepresentation().SetValue(self._style)
        # Check update condition
        if self._style != self._previous_style:
            self._previous_style = self._style
            # Select actors
            if self._style == 1:
                self._active_actors = self._actors_1
            elif self._style == 2:
                self._active_actors = self._actors_2
            # Delete old actors
            self._hide_actors(self._actors_1)
            self._hide_actors(self._actors_2)
            # Redraw the scene
            self._show_actors(self._active_actors)
            self._change_opacity(self._active_actors)

    def _draw_callback(self, value, widget):
        # Snap slider to nearest integer value
        self._depth = int(round(value))
        widget.GetSliderRepresentation().SetValue(self._depth)
        # Check update condition
        if self._depth != self._previous_depth:
            self._previous_depth = self._depth
            # Redraw the active actors
            self._hide_actors(self._active_actors)
            self._show_actors(self._active_actors)
            self._change_opacity(self._active_actors)
