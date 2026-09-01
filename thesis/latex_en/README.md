# CryptoMamba-v thesis (LaTeX)

English seven-chapter thesis with English UI, chart, and diagram labels. The document separates:

- paper-reported aggregate results;
- locally aligned CM-v, S5-Full, and naive-persistence evaluation;
- original paper trading replay;
- corrected self-financing evaluation;
- five-screen Console behavior and local checkpoint inference.

## Source layout

```text
main.tex
figures/
  controlled_forecast.pdf
  params_vs_error.pdf
  corrected_trading.pdf
  tikz/paper_source_architecture.tex
  tikz/system_flow.tex
  ui/ui_evaluation.png
  ui/ui_predict.png
  ui/ui_trading.png
```

Numeric figures summarize the evaluation described in the thesis. UI screenshots were captured
from the local Console after exercising both checkpoint paths.

## Compile locally

XeLaTeX is required because the source uses `fontspec` and `polyglossia`.

```bash
cd /Users/hieutha/PycharmProject/thesis_draft/latex_en
tectonic -X compile main.tex --keep-logs
```

## Upload to Overleaf

Upload `final/cryptomamba_thesis_overleaf_en.zip` as a new project and select **XeLaTeX**. The delivery
builder includes only `main.tex`, this README, and files referenced under `figures/`; generated PDF,
auxiliary files, backups, and unused figures are excluded.

## Chapter map

1. Introduction
2. Background and related work
3. Research methodology
4. Experimental results
5. CMamba-T/S5-Full experiment
6. Explanatory system
7. Discussion and conclusion
