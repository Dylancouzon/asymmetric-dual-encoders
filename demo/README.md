# Constella space-travel demo

This demo searches a tiny, fictional travel guide. It makes the model family easy to see:
documents are encoded once with Stella, then Nano and Zero query the same Qdrant collection.

## Run it

Use Python 3.10 or newer in a fresh virtual environment:

```bash
python -m pip install \
  "fastembed @ git+https://github.com/Dylancouzon/fastembed.git@constella-research-preview" \
  qdrant-client
python demo/space_travel.py
```

The first run downloads the three published model artifacts. The document encoder is about
1.75 GB; later runs use the local Hugging Face cache.

The script runs six natural-language searches with each query encoder and exits with an error if
an unexpected destination ranks first. To run only one query encoder:

```bash
python demo/space_travel.py --model nano
python demo/space_travel.py --model zero
```

The `CONSTELLA_DOC_PATH`, `CONSTELLA_NANO_PATH`, and `CONSTELLA_ZERO_PATH` environment variables
can point to pre-downloaded model directories. Otherwise, FastEmbed downloads the models from
Hugging Face.

The destinations are intentionally playful rather than factual travel claims. The search results
are real model output; no result or score is hard-coded.
