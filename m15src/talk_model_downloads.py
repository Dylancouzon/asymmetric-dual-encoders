"""Hugging Face downloads (last 30 days and all time) of the 25 E8/E8x retrievers, to pick the talk's labelled field.

The slide shows the 10 most downloaded in the last 30 days, one version per model family (bge-base-en v1 is
dropped in favour of bge-base-en-v1.5). Download counts change daily; the receipt records the fetch time.

Run: .venv/bin/python m15src/talk_model_downloads.py
"""
import json

from common import REPO, receipt, sha_file, utc_now, write_result

FIELD = REPO / "results/m15_talk_flops_field.json"
OUT = REPO / "results/m15_talk_model_downloads.json"
OLDER_VERSION = {"bge-base-en-v1": "bge-base-en-v1.5", "e5-base-v1": "e5-base-v2"}


def main():
    from huggingface_hub import HfApi
    started, api = utc_now(), HfApi()
    rows = {}
    for name, r in json.loads(FIELD.read_text())["results"].items():
        info = api.model_info(r["repo"], expand=["downloads", "downloadsAllTime"])
        rows[name] = {"repo": r["repo"], "downloads_30d": info.downloads, "downloads_all_time": info.downloads_all_time}
    ranked = sorted((n for n in rows if n not in OLDER_VERSION), key=lambda n: -rows[n]["downloads_30d"])
    top = ranked[:10]
    print(top)
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory (popularity snapshot)", "fetched_utc": started,
                       "rule": "top 10 by 30-day downloads, one version per family", "top10": top, "models": rows,
                       "receipt": receipt(__file__, {"field": {"path": str(FIELD.relative_to(REPO)), "sha256": sha_file(FIELD)}},
                                          started, ("huggingface_hub",))})


if __name__ == "__main__":
    main()
