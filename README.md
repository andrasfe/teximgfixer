# TeX Image Fixer

Iterative render → analyze → fix cycle for LaTeX/TikZ diagrams. Renders a `.tex` file, sends the image to a vision-capable LLM for analysis, and loops until the diagram looks correct (no overlapping lines, clipped nodes, etc.).

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

Create a `.env` file (already gitignored) with your OpenRouter credentials:

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemma-4-31b-it
```

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

- `iter_01.png` / `iter_01.tex` — first render + analysis
- `iter_02.png` / `iter_02.tex` — after first fix
- ...
- `final.png` / `final.tex` — converged result (or last iteration)

## How It Works

1. **Render** — Wraps the TikZ fragment in a standalone document, compiles with `tectonic`, converts PDF→PNG
2. **Analyze** — Sends the rendered image + current TeX source to a vision LLM via OpenRouter
3. **Fix** — If defects are found, the model produces corrected TeX; the cycle repeats
4. **Converge** — Stops when the model responds `DIAGRAM_OK` or max iterations reached

The model receives iteration history so it doesn't repeat failed approaches, and user hints are injected as constraints on every fix attempt.

## Project Structure

```
├── .env                # API keys (gitignored)
├── .gitignore
├── .venv/              # Python virtualenv
├── requirements.txt
├── tex_fixer.py        # Main application
├── example/
│   └── architecture.tex
└── output/             # Generated artifacts
```
