"""Sanity check that the package is importable and CI is wired up."""

import audiobook_engine


def test_package_importable():
    assert audiobook_engine is not None
