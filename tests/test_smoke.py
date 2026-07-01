"""Smoke tests — import-level checks that don't require the heavy AI stacks.

These verify the package wiring (structure, shared decoupling, config) without
importing torch/faiss/anthropic, so they run on any machine. Product-level
imports that need optional heavy deps are intentionally not exercised here.
"""
import importlib

import pytest


def test_package_imports():
    pkg = importlib.import_module("phansora")
    assert pkg.__version__


def test_config_loads():
    from phansora.config import settings

    assert settings.app_name
    assert isinstance(settings.cors_allow_origins, list)


def test_shared_utils_import():
    # Dependency-light helpers should import cleanly with no third-party stack.
    from phansora.shared.utils import chunking, naming

    assert callable(chunking.chunk_text)
    assert callable(naming.sanitize_stem)


def test_shared_does_not_import_products():
    """Guard the platform -> product dependency direction: no shared module may
    reference phansora.products."""
    import pathlib

    shared_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "phansora" / "shared"
    offenders = [
        str(p)
        for p in shared_dir.rglob("*.py")
        if "phansora.products" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"shared/ must not import products: {offenders}"


@pytest.mark.parametrize("module", [
    "phansora.shared.ai.anthropic",
])
def test_optional_ai_client_import(module):
    """shared.ai.anthropic imports the 'anthropic' SDK; skip if not installed."""
    pytest.importorskip("anthropic")
    pytest.importorskip("tenacity")
    importlib.import_module(module)
