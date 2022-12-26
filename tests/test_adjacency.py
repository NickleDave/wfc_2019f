"""Convert input data to adjacency information"""
from __future__ import annotations

import imageio  # type: ignore

from tests.conftest import Resources

from wfc import tiles
from wfc import patterns
from wfc import adjacency


def test_adjacency_extraction(resources: Resources) -> None:
    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.v2.imread(filename)
    tile_size = 1
    pattern_width = 2
    periodic = False
    _tile_catalog, tile_grid, _code_list, _unique_tiles = tiles.make_tile_catalog(img, tile_size)
    pattern_catalog, _pattern_weights, _pattern_list, pattern_grid = patterns.make_pattern_catalog(
        tile_grid, pattern_width, periodic
    )
    adjacency_relations = adjacency.adjacency_extraction(
        pattern_grid, pattern_catalog, direction_offsets
    )
    assert ((0, -1), -6150964001204120324, -4042134092912931260) in adjacency_relations
    assert ((-1, 0), -4042134092912931260, 3069048847358774683) in adjacency_relations
    assert ((1, 0), -3950451988873469076, -3950451988873469076) in adjacency_relations
    assert ((-1, 0), -3950451988873469076, -3950451988873469076) in adjacency_relations
    assert ((0, 1), -3950451988873469076, 3336256675067683735) in adjacency_relations
    assert (
        not ((0, -1), -3950451988873469076, -3950451988873469076) in adjacency_relations
    )
    assert (
        not ((0, 1), -3950451988873469076, -3950451988873469076) in adjacency_relations
    )
