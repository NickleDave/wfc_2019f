"""Breaks an image into consituant tiles."""
from __future__ import annotations

import imageio  # type: ignore
from tests.conftest import Resources
from wfc import tiles


def test_image_to_tile(resources: Resources) -> None:
    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.imread(filename)
    tiles = tiles.image_to_tiles(img, 1)
    assert tiles[2][2][0][0][0] == 255
    assert tiles[2][2][0][0][1] == 0


def test_make_tile_catalog(resources: Resources) -> None:
    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.imread(filename)
    print(img)
    tc, tg, cl, ut = tiles.make_tile_catalog(img, 1)
    print("tile catalog")
    print(tc)
    print("tile grid")
    print(tg)
    print("code list")
    print(cl)
    print("unique tiles")
    print(ut)
    assert ut[1][0] == 7
