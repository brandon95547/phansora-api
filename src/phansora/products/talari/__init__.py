"""Talari Book Creator — illustrated books (the product the app calls Talari Book Studio).

Two engines behind the routes in ``server.py``: illustrations, drawn by ComfyUI with
FLUX.1 Dev (comfy.py, styles.py), and text, written by DeepSeek (writing.py, prompts.py,
markup.py), with documents imported through Book Alchemy's parsers (sources.py). The
interface is the Node app's; this API keeps only the pictures it draws.
"""
