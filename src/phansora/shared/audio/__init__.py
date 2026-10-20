"""Speech and sound — one implementation for every product.

``whisper`` is faster-whisper: the one model cache in the process, and a transcribe that
runs as a guest of the shared GPU. Callers: SpokenVerse (/audio-to-txt, a cloned voice's
reference transcript), Narrava Studio (a narration's word timings, a song's lyrics).
Import from the modules (``phansora.shared.audio.whisper``), which is where tests patch.
"""
