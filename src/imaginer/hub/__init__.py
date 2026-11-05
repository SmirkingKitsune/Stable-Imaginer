"""Hugging Face Hub: finding model repos, reading their layout, and downloading from them.

* ``refs`` parses what people paste: repo ids, Hub URLs, ``hf://`` paths and llama.cpp's
  ``owner/name:QUANT``.
* ``layout`` sorts a repo's file list into what the engines load. Repos come in a few
  standard shapes: GGUF quants (flat, in per-quant folders, or split into shards) with their
  ``mmproj`` projectors; Diffusers pipelines (``model_index.json`` and a folder per
  component); ComfyUI repackages (``diffusion_models/``, ``vae/``, ``text_encoders/``);
  transformers checkpoints; and loose safetensors.
* ``client`` talks to the Hub's REST API, with the same token and endpoint the ``hf`` CLI uses.
* ``local`` finds copies already on disk, in the models folder or the Hugging Face cache.
* ``downloads`` fetches files in the background: resumable, checked against the Hub's
  SHA-256, and assigned to an engine's launch setting when done.
* ``service`` puts them together for the API and the CLI.
"""
