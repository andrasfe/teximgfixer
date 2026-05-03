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

ANALYST_MODEL = os.getenv("OPENROUTER_MODEL1", "google/gemma-4-31b-it")
FIXER_MODEL = os.getenv("OPENROUTER_MODEL3", "deepseek/deepseek-v4-pro")
JUDGE_MODEL = os.getenv("OPENROUTER_MODEL2", "qwen/qwen3.6-35b-a3b")

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
# Agent 1: Analyst — describes defects only, no fixing
# ---------------------------------------------------------------------------

ANALYST_PROMPT = """\
You are a LaTeX/TikZ diagram visual quality analyst. You ONLY describe defects — you NEVER fix them.

You will be shown a rendered image of a TikZ diagram. Produce a precise, structured defect report.

For each defect, specify:
- WHAT the defect is (overlap, clipping, crossing line, misalignment, etc.)
- WHERE it is (which nodes/labels/arrows are involved)
- SEVERITY (critical / moderate / minor)

If the diagram has NO visual defects, respond with exactly:
DIAGRAM_OK

Response format (when defects are found):
---DEFECTS---
1. [SEVERITY] <description of defect and which elements are involved>
2. [SEVERITY] <description>
...
---END---
"""


def run_analyst(client: OpenAI, image_data_uri: str, current_tex: str,
                iteration: int = 1, hints: Optional[str] = None) -> dict:
    """Run the Analyst agent: image → defect report.

    Returns dict: ok (bool), defects (str), retry (bool).
    """
    hints_note = ""
    if hints:
        hints_note = (
            f"\n\n--- USER HINTS ---\n"
            f"The user wants these constraints respected when fixing:\n{hints}"
        )

    user_content = [
        {
            "type": "text",
            "text": (
                f"Analyze this TikZ diagram (iteration {iteration}) for visual defects.\n\n"
                f"Current TeX source:\n```latex\n{current_tex}\n```"
                + hints_note
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_data_uri},
        },
    ]

    response = client.chat.completions.create(
        model=ANALYST_MODEL,
        messages=[
            {"role": "system", "content": ANALYST_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4000,
        temperature=0.2,
    )

    content = response.choices[0].message.content
    if content is None:
        print("  [ANALYST] Empty response — will retry.")
        return {"ok": False, "defects": "", "retry": True}

    text = content.strip()

    if text == "DIAGRAM_OK" or "DIAGRAM_OK" in text:
        return {"ok": True, "defects": "", "retry": False}

    # Extract defect report
    defects_match = re.search(r"---DEFECTS---\s*\n(.*?)---END---", text, re.DOTALL)
    defects = defects_match.group(1).strip() if defects_match else text

    return {"ok": False, "defects": defects, "retry": False}


# ---------------------------------------------------------------------------
# Agent 2: Fixer — takes defect report + TeX, produces fixed TeX
# ---------------------------------------------------------------------------

FIXER_PROMPT = """\
You are a LaTeX/TikZ diagram fixer. You receive a defect report and the current TeX source.
Your ONLY job is to produce a corrected version of the TeX that resolves ALL listed defects.

CRITICAL RULES:
a) NEVER make small incremental tweaks to the same parameters that failed before.
   If increasing `node distance` by 2mm didn't work, increasing by 4mm won't either.
   You need a STRUCTURALLY DIFFERENT approach.

b) PREFER COMPLETE RESTRUCTURING over parameter adjustment:
   - Replace relative positioning (`below=of X`) with ABSOLUTE coordinates using `\\node at (x,y)`
   - Use `\\coordinate` to define anchor points for arrow routing
   - Split the diagram into clear horizontal tiers with explicit y-coordinates
   - Use `column sep` and `row sep` in matrix layouts if appropriate

c) If nodes overlap vertically, assign each tier a fixed y-coordinate
   (e.g., y=0, y=-3, y=-6, etc.) and place nodes at those coordinates.

d) If group box labels overlap, move them OUTSIDE the box using `label=above:` or
   place them as separate nodes.

e) For arrow routing through tight spaces, use explicit `to[out=angle, in=angle]`
   or `|-` / `-|` operators with intermediate coordinates.

f) When in doubt, spread things out MUCH more than seems necessary.

Response format:
---FIXED_TEX---
<the complete corrected TeX code (just the tikzpicture, no \\documentclass)>
---END---
"""


def run_fixer(client: OpenAI, defects: str, current_tex: str,
              iteration: int = 1, history: Optional[str] = None,
              hints: Optional[str] = None) -> dict:
    """Run the Fixer agent: defect report + TeX → fixed TeX.

    Returns dict: fixed_tex (str|None), retry (bool).
    """
    history_note = ""
    if history:
        history_note = (
            f"\n\n--- PREVIOUS FIX HISTORY ---\n"
            f"Previous fixes FAILED to resolve the defects. You must take a DIFFERENT approach.\n"
            f"Do NOT repeat the same type of changes.\n\n"
            f"{history}"
        )

    hints_note = ""
    if hints:
        hints_note = (
            f"\n\n--- USER HINTS ---\n"
            f"You MUST follow these constraints:\n{hints}"
        )

    user_msg = (
        f"Defect report from analyst (iteration {iteration}):\n"
        f"---DEFECTS---\n{defects}\n---END---\n\n"
        f"Current TeX source:\n```latex\n{current_tex}\n```"
        + history_note
        + hints_note
    )

    response = client.chat.completions.create(
        model=FIXER_MODEL,
        messages=[
            {"role": "system", "content": FIXER_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=16000,
        temperature=0.4,
    )

    content = response.choices[0].message.content
    if content is None:
        print("  [FIXER] Empty response — will retry.")
        return {"fixed_tex": None, "retry": True}

    text = content.strip()

    # Extract fixed TeX
    tex_match = re.search(r"---FIXED_TEX---\s*\n(.*?)---END---", text, re.DOTALL)
    fixed_tex = tex_match.group(1).strip() if tex_match else None

    if fixed_tex is None:
        # Fallback: extract ```latex ... ```
        code_match = re.search(r"```latex\n(.*?)```", text, re.DOTALL)
        if code_match:
            fixed_tex = code_match.group(1).strip()

    return {"fixed_tex": fixed_tex, "retry": False}


# ---------------------------------------------------------------------------
# Agent 3: Judge — compares before/after images, accepts or rejects fix
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are a LaTeX/TikZ diagram quality judge. You compare a BEFORE and AFTER image of a diagram fix.

Your job:
1. Check if the defects listed in the defect report are resolved in the AFTER image.
2. Check if the fix introduced any NEW defects (regressions).
3. Decide: ACCEPT the fix, or REJECT it (revert to before).

Scoring:
- Rate defect resolution: IMPROVED / SAME / WORSE
- Rate new defects: NONE / MINOR / MAJOR
- Overall verdict: ACCEPT or REJECT

If the AFTER image has NO defects at all (original defects resolved and no new ones),
respond with DIAGRAM_OK instead.

Response format:
---VERDICT---
Resolution: <IMPROVED/SAME/WORSE>
New defects: <NONE/MINOR/MAJOR>
Verdict: <ACCEPT/REJECT>
Reasoning: <1-2 sentences>
---END---
"""


JUDGE_IMAGE_DIM = int(os.getenv("JUDGE_IMAGE_DIM", "1000"))


def png_to_judge_data_uri(png_path: Path) -> str:
    """Create a smaller data-URI for the Judge (dual-image payload)."""
    img = Image.open(png_path)
    if max(img.size) > JUDGE_IMAGE_DIM:
        ratio = JUDGE_IMAGE_DIM / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64}"
    print(f"  Judge image: {len(b64) // 1024} KB base64 ({img.width}x{img.height})")
    return data_uri


def run_judge(client: OpenAI, before_uri: str, after_uri: str,
              defects: str, hints: Optional[str] = None,
              use_fallback_model: bool = False) -> dict:
    """Run the Judge agent: before/after comparison → accept or reject.

    Returns dict: accepted (bool), verdict (str), diagram_ok (bool), retry (bool).
    """
    model = ANALYST_MODEL if use_fallback_model else JUDGE_MODEL

    hints_note = ""
    if hints:
        hints_note = f"\n\nUser constraints for the diagram:\n{hints}"

    user_content = [
        {
            "type": "text",
            "text": (
                "Compare BEFORE (original) and AFTER (fixed) versions of a TikZ diagram.\n\n"
                f"Defects that were supposed to be fixed:\n---DEFECTS---\n{defects}\n---END---\n"
                + hints_note
            ),
        },
        {
            "type": "text",
            "text": "BEFORE image:",
        },
        {
            "type": "image_url",
            "image_url": {"url": before_uri},
        },
        {
            "type": "text",
            "text": "AFTER image:",
        },
        {
            "type": "image_url",
            "image_url": {"url": after_uri},
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2000,
        temperature=0.2,
    )

    content = response.choices[0].message.content
    if content is None:
        print("  [JUDGE] Empty response — will retry.")
        return {"accepted": False, "verdict": "Judge returned no content", "diagram_ok": False, "retry": True}

    text = content.strip()

    if "DIAGRAM_OK" in text:
        return {"accepted": True, "verdict": "No defects remaining", "diagram_ok": True, "retry": False}

    # Parse verdict
    verdict_match = re.search(r"---VERDICT---\s*\n(.*?)---END---", text, re.DOTALL)
    verdict_text = verdict_match.group(1).strip() if verdict_match else text

    accepted = "Verdict: ACCEPT" in verdict_text
    diagram_ok = False

    return {"accepted": accepted, "verdict": verdict_text, "diagram_ok": diagram_ok, "retry": False}


# ---------------------------------------------------------------------------
# LLM call helper with retry
# ---------------------------------------------------------------------------

def llm_with_retry(fn, max_retries=3, **kwargs):
    """Call an agent function with retry on empty responses."""
    for attempt in range(1, max_retries + 1):
        result = fn(**kwargs)
        if not result.get("retry"):
            return result
        if attempt < max_retries:
            wait = attempt * 5
            print(f"  Retry {attempt}/{max_retries} after {wait}s...")
            time.sleep(wait)
    return result


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
    best_tex = current_tex  # Best accepted version so far
    history_lines = []
    fix_counter = 0  # Counts accepted fixes (for artifact naming)
    consecutive_rejects = 0

    print(f"TeX Fixer starting (multi-agent: Analyst → Fixer → Judge)")
    print(f"  Input:     {tex_file}")
    print(f"  Output:    {out_dir}")
    print(f"  Analyst:   {ANALYST_MODEL}")
    print(f"  Fixer:     {FIXER_MODEL}")
    print(f"  Judge:     {JUDGE_MODEL}")
    print(f"  Max iters: {MAX_ITERATIONS}")
    if hints:
        print(f"  Hints:     {hints}")
    print()

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"=== Iteration {iteration}/{MAX_ITERATIONS} ===")

        with tempfile.TemporaryDirectory(prefix="texfix_") as tmp:
            work = Path(tmp)

            # --- Step 1: Render current TeX ---
            full_doc = wrap_tex(current_tex)
            pdf_path = render_tex_to_pdf(full_doc, work)
            before_png = pdf_to_png(pdf_path)

            # Save before image
            iter_png = out_dir / f"iter_{iteration:02d}_before.png"
            iter_tex = out_dir / f"iter_{iteration:02d}_before.tex"
            shutil.copy2(before_png, iter_png)
            iter_tex.write_text(current_tex, encoding="utf-8")
            print(f"  Rendered (before): {iter_png}")

            before_uri = png_to_base64_data_uri(before_png)

            # --- Step 2: Analyst — identify defects ---
            print(f"  [ANALYST] Analyzing image...")
            analyst_result = llm_with_retry(
                run_analyst, client=client,
                image_data_uri=before_uri, current_tex=current_tex,
                iteration=iteration, hints=hints,
            )

            if analyst_result["ok"]:
                print(f"\n✓ Analyst: No defects found after {iteration} iteration(s)!")
                final_png = out_dir / "final.png"
                final_tex = out_dir / "final.tex"
                shutil.copy2(iter_png, final_png)
                shutil.copy2(iter_tex, final_tex)
                print(f"  Final PNG: {final_png}")
                print(f"  Final TeX: {final_tex}")
                return

            defects = analyst_result["defects"]
            print(f"  [ANALYST] Defects found:\n{defects[:300]}...")

            # Record in history
            history_lines.append(f"Iteration {iteration} defects: {defects[:200]}")
            history = "\n".join(history_lines) if history_lines else None

            # --- Step 3: Fixer — produce corrected TeX ---
            print(f"  [FIXER] Generating fix...")
            fixer_result = llm_with_retry(
                run_fixer, client=client,
                defects=defects, current_tex=current_tex,
                iteration=iteration, history=history, hints=hints,
            )

            if fixer_result["fixed_tex"] is None:
                print("  [FIXER] No fix produced. Skipping to next iteration.")
                continue

            proposed_tex = fixer_result["fixed_tex"]
            print(f"  [FIXER] Proposed fix ({len(proposed_tex)} chars)")

            # --- Step 4: Render the proposed fix ---
            try:
                proposed_doc = wrap_tex(proposed_tex)
                proposed_pdf = render_tex_to_pdf(proposed_doc, work)
                after_png = pdf_to_png(proposed_pdf)
            except Exception as e:
                print(f"  [ERROR] Proposed TeX failed to render: {e}")
                print("  Rejecting fix (compilation error).")
                consecutive_rejects += 1
                continue

            after_uri = png_to_base64_data_uri(after_png)

            # Save after image
            after_iter_png = out_dir / f"iter_{iteration:02d}_after.png"
            after_iter_tex = out_dir / f"iter_{iteration:02d}_after.tex"
            shutil.copy2(after_png, after_iter_png)
            after_iter_tex.write_text(proposed_tex, encoding="utf-8")
            print(f"  Rendered (after):  {after_iter_png}")

            # --- Step 5: Judge — accept or reject ---
            # Use smaller images for Judge (dual-image payload)
            judge_before_uri = png_to_judge_data_uri(before_png)
            judge_after_uri = png_to_judge_data_uri(after_png)

            print(f"  [JUDGE] Evaluating fix...")
            judge_result = llm_with_retry(
                run_judge, client=client,
                before_uri=judge_before_uri, after_uri=judge_after_uri,
                defects=defects, hints=hints,
            )

            # If Judge failed with primary model, try fallback (Analyst model)
            if judge_result.get("retry") or (not judge_result["accepted"] and "no content" in judge_result.get("verdict", "").lower()):
                print(f"  [JUDGE] Retrying with fallback model ({ANALYST_MODEL})...")
                judge_result = llm_with_retry(
                    run_judge, client=client,
                    before_uri=judge_before_uri, after_uri=judge_after_uri,
                    defects=defects, hints=hints,
                    use_fallback_model=True,
                )

            print(f"  [JUDGE] {judge_result['verdict'][:200]}")

            if judge_result["diagram_ok"]:
                print(f"\n✓ Judge: Diagram looks perfect after {iteration} iteration(s)!")
                final_png = out_dir / "final.png"
                final_tex = out_dir / "final.tex"
                shutil.copy2(after_iter_png, final_png)
                shutil.copy2(after_iter_tex, final_tex)
                print(f"  Final PNG: {final_png}")
                print(f"  Final TeX: {final_tex}")
                return

            if judge_result["accepted"]:
                current_tex = proposed_tex
                best_tex = proposed_tex
                fix_counter += 1
                consecutive_rejects = 0
                print(f"  ✓ Fix ACCEPTED ({fix_counter} accepted so far)")
            else:
                consecutive_rejects += 1
                print(f"  ✗ Fix REJECTED — reverting to previous version")
                # Keep current_tex unchanged (revert)
                history_lines.append(f"Iteration {iteration} judge: REJECTED — fix made things worse or didn't help")

                # If rejected 3 times in a row, reset to best known version
                if consecutive_rejects >= 3:
                    if current_tex != best_tex:
                        print(f"  [WARN] 3 consecutive rejections. Resetting to best known version.")
                        current_tex = best_tex
                        consecutive_rejects = 0

    # If we exhausted iterations
    print(f"\n✗ Max iterations ({MAX_ITERATIONS}) reached without convergence.")
    print(f"  Best version saved as final.tex / final.png in {out_dir}")

    # Save best version as final
    final_png = out_dir / "final.png"
    final_tex = out_dir / "final.tex"
    # Re-render best_tex for the final PNG
    with tempfile.TemporaryDirectory(prefix="texfix_") as tmp:
        work = Path(tmp)
        full_doc = wrap_tex(best_tex)
        pdf_path = render_tex_to_pdf(full_doc, work)
        png_path = pdf_to_png(pdf_path)
        shutil.copy2(png_path, final_png)
    final_tex.write_text(best_tex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="TeX Image Fixer — iterative render/analyze/fix cycle")
    parser.add_argument("tex_file", help="Path to the .tex file to fix")
    parser.add_argument("-o", "--output-dir", default=None, help="Directory for output artifacts")
    parser.add_argument("-H", "--hints", default=None, help="User hints for fixing (e.g. 'no overlapping lines, keep the size')")
    args = parser.parse_args()
    run(args.tex_file, args.output_dir, hints=args.hints)


if __name__ == "__main__":
    main()
