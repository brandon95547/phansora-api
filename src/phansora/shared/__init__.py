"""Cross-cutting infrastructure shared by all Phansora products.

Submodules: ai, auth, billing, database, storage, queue, utils.
Nothing here may import from ``phansora.products`` — the dependency direction is
always product -> shared.
"""
