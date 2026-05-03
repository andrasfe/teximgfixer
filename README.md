# TeX Image Fixer

Multi-agent iterative fix cycle for LaTeX/TikZ diagrams. Three specialized LLM agents — **Analyst**, **Fixer**, and **Judge** — collaborate to render, diagnose, fix, and validate TikZ diagrams until they look correct.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires [tectonic](https://tectonic-typesetting.github.io/) for LaTeX rendering:

```bash
brew install tectonic
```

## Configuration

Create a `.env` file (already gitignored) with your OpenRouter credentials and three model assignments:

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL1=google/gemma-4-31b-it
OPENROUTER_MODEL2=qwen/qwen3.6-35b-a3b
OPENROUTER_MODEL3=deepseek/deepseek-v4-pro
```

- **MODEL1** → Analyst (describes defects only)
- **MODEL2** → Fixer (produces corrected TeX)
- **MODEL3** → Judge (accepts or rejects fixes)

Using different models for each role provides diverse perspectives and avoids shared blind spots.

Optional overrides:

| Variable | Default | Description |
|---|---|---|
| `MAX_FIX_ITERATIONS` | `8` | Max render/analyze/fix cycles |
| `RENDER_DPI` | `300` | DPI for PDF→PNG conversion |
| `MAX_IMAGE_DIM` | `1600` | Max image dimension (px) sent to API |

## Usage

```bash
.venv/bin/python tex_fixer.py <tex_file> [-o OUTPUT_DIR] [-H "hints"]
```

### Examples

Basic run:
```bash
.venv/bin/python tex_fixer.py example/architecture.tex -o output
```

With user hints:
```bash
.venv/bin/python tex_fixer.py example/architecture.tex -o output \
  -H "no overlapping lines, keep the size. move around boxes as needed."
```

## Output

Artifacts are saved per-iteration in the output directory:

- `iter_01_before.png` / `.tex` — rendered before fix attempt
- `iter_01_after.png` / `.tex` — rendered after fix attempt
- ...
- `final.png` / `final.tex` — converged result (or best accepted version)

## Multi-Agent Architecture

Each iteration follows a **5-step pipeline**:

1. **Render** — Wraps the TikZ fragment in a standalone document, compiles with `tectonic`, converts PDF→PNG
2. **Analyst** — Vision LLM examines the rendered image and produces a structured defect report (what, where, severity). Does NOT fix — this separation prevents biasing the diagnosis toward easy fixes.
3. **Fixer** — Receives the defect report + current TeX + fix history, produces corrected TeX. Different model from Analyst to avoid shared blind spots. Gets progressively stronger prompts if previous fixes failed.
4. **Re-render** — The proposed fix is compiled and rendered to PNG. If it fails to compile, the fix is auto-rejected.
5. **Judge** — Compares before and after images side-by-side. Decides ACCEPT or REJECT. Can detect regressions the Fixer missed. If rejected, the fix is reverted.

### Key features:
- **Fix rejection** — Bad fixes get rolled back instead of becoming the new baseline
- **Consecutive rejection reset** — After 3 rejections in a row, reverts to the best known version
- **Best-version tracking** — The final output is always the best accepted version, not the last attempted
- **Iteration history** — The Fixer sees all previous defect reports and judge verdicts to avoid repeating failed approaches
- **User hints** — Constraints from `-H` are injected into all three agents

## Project Structure

```
├── .env                # API keys + model config (gitignored)
├── .gitignore
├── .venv/              # Python virtualenv
├── requirements.txt
├── tex_fixer.py        # Main application
├── example/
│   └── architecture.tex
└── output/             # Generated artifacts
```
