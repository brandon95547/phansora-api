"""Cross-cutting infrastructure shared by all Phansora products.

Submodules: admin, ai, auth, conversions, net, utils (and contact, errors, gpu, paths).
Nothing here may import from ``phansora.products`` — the dependency direction is
always product -> shared.
"""
