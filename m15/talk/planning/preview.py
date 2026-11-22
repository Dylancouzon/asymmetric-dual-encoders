"""Render deck slides to PNGs with headless Chrome: python3 preview.py id1 id2 ..."""
import re, subprocess, sys
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "deck"
OUT = Path(__file__).parent / "shots"; OUT.mkdir(exist_ok=True)
css = (D / "project/deck.css").read_text()
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for sid in sys.argv[1:]:
    sec = (D / f"project/slides/{sid}.html").read_text()
    sec = re.sub(r"<aside>.*?</aside>", "", sec, flags=re.S).replace(" hidden>", ">")
    sec = sec.replace('src="project/', f'src="file://{D}/project/')
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Mona+Sans:wght@200..900&family=JetBrains+Mono:wght@400;600&display=swap">
<style>*{{margin:0;box-sizing:border-box}} body{{width:1920px;height:1080px;overflow:hidden;background:#000}}
section{{width:1920px;height:1080px;position:relative;overflow:hidden}} p,h1,h2,h3{{margin:0}}
*{{animation-duration:0s!important;animation-delay:0s!important;transition:none!important}}
.fragment{{opacity:1!important;transform:none!important}}
{css}</style></head><body>{sec}</body></html>"""
    f = OUT / f"{sid}.html"; f.write_text(page)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--window-size=1920,1080",
                    "--virtual-time-budget=3000", f"--screenshot={OUT}/{sid}.png", f"file://{f}"], capture_output=True)
    print(OUT / f"{sid}.png")
