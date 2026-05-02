#!/usr/bin/env python3
"""
TeX Image Fixer — Iterative render → analyze → fix cycle.

Renders a .tex file to PDF/PNG, sends the image to a vision-capable LLM
(via OpenRouter) for analysis, and loops until the diagram looks correct
(no overlapping lines, clipped nodes, etc.).
"""

import argparse
import base64
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
VISION_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")
MAX_ITERATIONS = int(os.getenv("MAX_FIX_ITERATIONS", "8"))
RENDER_DPI = int(os.getenv("RENDER_DPI", "300"))

# ---------------------------------------------------------------------------
# TeX wrapping
# ---------------------------------------------------------------------------

TIKZ_LIBS = (
    "shapes.geometric, arrows.meta, positioning, fit, backgrounds, calc"
)

DOCUMENT_TEMPLATE = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{TIKZ_LIBS_PLACEHOLDER}
\begin{document}
TEX_CONTENT_PLACEHOLDER
\end{document}
"""


def wrap_tex(tex_content: str) -> str:
    """Wrap a tikzpicture fragment in a standalone document."""
    # If the content already has \documentclass, return as-is
    if r"\documentclass" in tex_content:
        return tex_content
    wrapped = DOCUMENT_TEMPLATE.replace("TIKZ_LIBS_PLACEHOLDER", TIKZ_LIBS)
    wrapped = wrapped.replace("TEX_CONTENT_PLACEHOLDER", tex_content)
    return wrapped


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_tex_to_pdf(tex_source: str, work_dir: Path) -> Path:
    """Render LaTeX source to PDF using tectonic. Returns path to PDF."""
    tex_path = work_dir / "input.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    result = subprocess.run(
        ["tectonic", "-k", "--outdir", str(work_dir), str(tex_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    pdf_path = work_dir / "input.pdf"
    if not pdf_path.exists():
        print(f"[ERROR] tectonic failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        sys.exit(1)

    return pdf_path


def pdf_to_png(pdf_path: Path, dpi: int = RENDER_DPI) -> Path:
    """Convert first page of PDF to high-res PNG."""
    png_path = pdf_path.with_suffix(".png")
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(png_path))
    doc.close()
    return png_path


MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))


def png_to_base64_data_uri(png_path: Path) -> str:
    """Read a PNG, resize if needed, and return a base64 data-URI string."""
    img = Image.open(png_path)
    # Resize to keep payload manageable for vision APIs
    if max(img.size) > MAX_IMAGE_DIM:
        ratio = MAX_IMAGE_DIM / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64}"
    print(f"  Image payload: {len(b64) // 1024} KB base64 ({img.width}x{img.height})")
    return data_uri


# ---------------------------------------------------------------------------
# LLM analysis & fix
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a LaTeX/TikZ diagram quality inspector and fixer.

You will be shown a rendered image of a LaTeX TikZ diagram. Your job is to:

1. ANALYZE the image for visual defects such as:
   - Overlapping text or nodes
   - Clipped or cut-off elements
   - Lines/arrows crossing through text
   - Misaligned nodes
   - Labels that are unreadable or too close together
   - Any other visual layout problems

2. If you find defects, produce a FIXED version of the original TeX code that
   resolves the issues. CRITICAL RULES FOR FIXES:

   a) NEVER make small incremental tweaks to the same parameters that failed before.
      If increasing `node distance` by 2mm didn't work last time, increasing it by 4mm
      won't either. You need a STRUCTURALLY DIFFERENT approach.

   b) PREFER COMPLETE RESTRUCTURING over parameter adjustment:
      - Replace relative positioning (`below=of X`) with ABSOLUTE coordinates
        using `\node at (x,y)` where you control exact placement
      - Use `\coordinate` to define anchor points for arrow routing
      - Split the diagram into clear horizontal tiers with explicit y-coordinates
      - Use `column sep` and `row sep` in matrix layouts if appropriate

   c) If nodes overlap vertically, DO NOT just increase `below=` distance.
      Instead, assign each tier a fixed y-coordinate (e.g., y=0, y=-3, y=-6, etc.)
      and place nodes at those coordinates.

   d) If group box labels overlap, move them OUTSIDE the box entirely using
      `label=above:` or place them as separate nodes.

   e) For arrow routing through tight spaces, use explicit `to[out=angle, in=angle]`
      or `|-` / `-|` operators with intermediate coordinates.

   f) When in doubt, spread things out MUCH more than seems necessary.

3. If the diagram looks correct with NO defects, respond with exactly:
   DIAGRAM_OK

Response format (when defects are found):
---ANALYSIS---
<brief description of defects found>
---FIXED_TEX---
<the complete corrected TeX code (just the tikzpicture, no \\documentclass)>
---END---

Response format (when no defects):
DIAGRAM_OK
"""


def analyze_and_fix(client: OpenAI, image_data_uri: str, current_tex: str,
                     iteration: int = 1, history: Optional[str] = None,
                     hints: Optional[str] = None) -> dict:
    """Send the rendered image + current tex to the vision model.

    Returns dict with keys: ok (bool), analysis (str), fixed_tex (str|None).
    """
    history_note = ""
    if history:
        history_note = (
            f"\n\n--- PREVIOUS ATTEMPTS (iteration {iteration}) ---\n"
            f"Previous fixes were INSUFFICIENT. The same defects persist.\n"
            f"You must make MUCH LARGER changes than before.\n\n"
            f"Previous analysis history:\n{history}"
        )

    hints_note = ""
    if hints:
        hints_note = (
            f"\n\n--- USER HINTS ---\n"
            f"The user has provided the following guidance for fixing the diagram. "
            f"You MUST follow these hints as constraints on your fix:\n"
            f"{hints}"
        )

    user_content = [
        {
            "type": "text",
            "text": (
                f"Here is the rendered image of the current TikZ diagram (iteration {iteration}). "
                "Analyze it for visual defects and fix the TeX if needed.\n\n"
                "Current TeX source code:\n\n"
                f"```latex\n{current_tex}\n```"
                + history_note
                + hints_note
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_data_uri},
        },
    ]

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=16000,
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if content is None:
        # Model returned no content — could be filtering, rate limit, or transient error.
        # Return a retry-able result so the loop can try again.
        print("  [WARN] Model returned empty content. Will retry.")
        return {"ok": False, "analysis": "Model returned no content (retry).", "fixed_tex": None, "retry": True}

    text = content.strip()

    if text == "DIAGRAM_OK" or text.endswith("\nDIAGRAM_OK"):
        return {"ok": True, "analysis": "No defects found.", "fixed_tex": None}

    # Parse structured response
    analysis_match = re.search(r"---ANALYSIS---\s*\n(.*?)---FIXED_TEX---", text, re.DOTALL)
    tex_match = re.search(r"---FIXED_TEX---\s*\n(.*?)---END---", text, re.DOTALL)

    analysis = analysis_match.group(1).strip() if analysis_match else "(could not parse analysis)"
    fixed_tex = tex_match.group(1).strip() if tex_match else None

    if fixed_tex is None:
        # Fallback: try to extract anything between ```latex ... ```
        code_match = re.search(r"```latex\n(.*?)```", text, re.DOTALL)
        if code_match:
            fixed_tex = code_match.group(1).strip()

    return {"ok": fixed_tex is None, "analysis": analysis, "fixed_tex": fixed_tex}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(tex_path: str, output_dir: Optional[str] = None, hints: Optional[str] = None):
    if not OPENROUTER_API_KEY:
        print("[ERROR] OPENROUTER_API_KEY not set. Check your .env file.")
        sys.exit(1)

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    tex_file = Path(tex_path).resolve()
    if not tex_file.exists():
        print(f"[ERROR] TeX file not found: {tex_file}")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else tex_file.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_tex = tex_file.read_text(encoding="utf-8")
    original_tex = current_tex
    history_lines = []

    print(f"TeX Fixer starting")
    print(f"  Input:   {tex_file}")
    print(f"  Output:  {out_dir}")
    print(f"  Model:   {VISION_MODEL}")
    print(f"  Max iters: {MAX_ITERATIONS}")
    print()

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"=== Iteration {iteration}/{MAX_ITERATIONS} ===")

        with tempfile.TemporaryDirectory(prefix="texfix_") as tmp:
            work = Path(tmp)

            # Wrap & render
            full_doc = wrap_tex(current_tex)
            pdf_path = render_tex_to_pdf(full_doc, work)
            png_path = pdf_to_png(pdf_path)

            # Save iteration artifacts
            iter_png = out_dir / f"iter_{iteration:02d}.png"
            iter_tex = out_dir / f"iter_{iteration:02d}.tex"
            shutil.copy2(png_path, iter_png)
            iter_tex.write_text(current_tex, encoding="utf-8")
            print(f"  Rendered: {iter_png}")

            # Build history string from previous analyses
            history = "\n".join(history_lines) if history_lines else None

            # Analyze (with retry for empty model responses)
            image_uri = png_to_base64_data_uri(png_path)
            max_retries = 3
            result = None
            for attempt in range(1, max_retries + 1):
                result = analyze_and_fix(client, image_uri, current_tex,
                                         iteration=iteration, history=history,
                                         hints=hints)
                if not result.get("retry"):
                    break
                if attempt < max_retries:
                    wait = attempt * 5
                    print(f"  Retry {attempt}/{max_retries} after {wait}s...")
                    time.sleep(wait)

            print(f"  Analysis: {result['analysis']}")

            # Record this analysis in history for future iterations
            history_lines.append(f"Iteration {iteration}: {result['analysis']}")

            if result["ok"]:
                print(f"\n✓ Diagram looks correct after {iteration} iteration(s)!")
                # Copy final as "final" artifacts
                final_png = out_dir / "final.png"
                final_tex = out_dir / "final.tex"
                shutil.copy2(iter_png, final_png)
                shutil.copy2(iter_tex, final_tex)
                print(f"  Final PNG: {final_png}")
                print(f"  Final TeX: {final_tex}")
                return

            if result["fixed_tex"] is None:
                print("  [WARN] Model did not provide fixed TeX. Stopping.")
                break

            current_tex = result["fixed_tex"]
            print(f"  Updated TeX ({len(current_tex)} chars)")

    # If we exhausted iterations
    print(f"\n✗ Max iterations ({MAX_ITERATIONS}) reached without convergence.")
    print(f"  Last version saved as iter_{MAX_ITERATIONS:02d}.tex / .png in {out_dir}")

    # Save whatever we have as final
    final_png = out_dir / "final.png"
    final_tex = out_dir / "final.tex"
    last_png = out_dir / f"iter_{MAX_ITERATIONS:02d}.png"
    last_tex = out_dir / f"iter_{MAX_ITERATIONS:02d}.tex"
    if last_png.exists():
        shutil.copy2(last_png, final_png)
    if last_tex.exists():
        shutil.copy2(last_tex, final_tex)


def main():
    parser = argparse.ArgumentParser(description="TeX Image Fixer — iterative render/analyze/fix cycle")
    parser.add_argument("tex_file", help="Path to the .tex file to fix")
    parser.add_argument("-o", "--output-dir", default=None, help="Directory for output artifacts")
    parser.add_argument("-H", "--hints", default=None, help="User hints for fixing (e.g. 'no overlapping lines, keep the size')")
    args = parser.parse_args()
    run(args.tex_file, args.output_dir, hints=args.hints)


if __name__ == "__main__":
    main()
