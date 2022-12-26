from __future__ import annotations
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent / '..'

class Resources:
    def get_image(self, image: str) -> str:
        return PROJECT_ROOT / f"data/images/{image}"


@pytest.fixture(scope="session")
def resources() -> Resources:
    return Resources()