"""Cross-cutting infrastructure shared by all Phanoris products.

Submodules: ai, auth, billing, database, storage, queue, utils.
Nothing here may import from ``phanoris.products`` — the dependency direction is
always product -> shared.
"""
