"""Summarize and plot the Reviewer 1 scheduler sensitivity experiment.

The script uses only the Python standard library. It writes a compact summary
CSV and a standalone PGFPlots source that can be compiled with any TeX Live
installation containing pgfplots.
"""

import argparse
import csv
from pathlib import Path


LABELS = {
    "step": "StepLR",
    "cosine": "Cosine annealing",
    "plateau": "ReduceLROnPlateau",
}


def read_history(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or {int(row["epoch"]) for row in rows} != set(range(100)):
        raise ValueError(f"expected one row for every epoch 0--99 in {path}")
    return rows


def write_summary(path, histories):
    fields = [
        "scheduler", "epoch_budget", "best_epoch", "best_val_psnr",
        "best_val_ssim", "final_val_psnr", "final_val_ssim", "final_lr",
        "psnr_change_last_10_epochs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, rows in histories.items():
            best = max(rows, key=lambda row: float(row["val_psnr"]))
            writer.writerow({
                "scheduler": LABELS[key],
                "epoch_budget": len(rows),
                "best_epoch": int(best["epoch"]) + 1,
                "best_val_psnr": f'{float(best["val_psnr"]):.6f}',
                "best_val_ssim": f'{float(best["val_ssim"]):.6f}',
                "final_val_psnr": f'{float(rows[-1]["val_psnr"]):.6f}',
                "final_val_ssim": f'{float(rows[-1]["val_ssim"]):.6f}',
                "final_lr": f'{float(rows[-1]["lr_used"]):.10g}',
                "psnr_change_last_10_epochs": (
                    f'{float(rows[-1]["val_psnr"])-float(rows[-11]["val_psnr"]):.6f}'
                ),
            })


def pgf_table(rows, column):
    return "\n".join(
        f'{int(row["epoch"])+1} {float(row[column]):.10g}' for row in rows
    )


def write_plot(path, histories):
    colors = {"step": "blue", "cosine": "orange", "plateau": "teal"}
    marks = {"step": "none", "cosine": "none", "plateau": "none"}
    panels = []
    for panel_index, (title, column, ylabel, extra) in enumerate([
        ("(a) Validation PSNR", "val_psnr", "PSNR (dB)", ""),
        ("(b) Validation SSIM", "val_ssim", "SSIM", ""),
        ("(c) Learning-rate schedule", "lr_used", "Learning rate", "ymode=log,"),
    ]):
        plots = []
        for key, rows in histories.items():
            legend_entry = f"\\addlegendentry{{{LABELS[key]}}}" if panel_index == 0 else ""
            plots.append(
                f"\\addplot+[color={colors[key]},mark={marks[key]},very thick] table[row sep=\\\\] {{%\n"
                f"epoch value\\\\\n{pgf_table(rows, column).replace(chr(10), chr(92)*2+chr(10))}\\\\\n}};\n"
                f"{legend_entry}"
            )
        legend_target = "legend to name=schedulerlegend," if panel_index == 0 else ""
        panels.append(
            "\\nextgroupplot[\n"
            f"title={{{title}}},xlabel={{Epoch}},ylabel={{{ylabel}}},{extra}\n"
            f"xmin=1,xmax=100,grid=major,{legend_target}]\n"
            + "\n".join(plots)
        )

    document = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=3 by 1,horizontal sep=1.35cm},
  width=0.34\textwidth,height=0.27\textwidth,
  tick label style={font=\small},label style={font=\small},
  title style={font=\small},legend style={font=\small},
]
""" + "\n".join(panels) + r"""
\end{groupplot}
\node at ($(group c2r1.south)+(0,-1.25cm)$) {\ref{schedulerlegend}};
\end{tikzpicture}
\end{document}
"""
    path.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    args = parser.parse_args()
    histories = {
        key: read_history(args.input_dir / f"{key}_history.csv")
        for key in ("step", "cosine", "plateau")
    }
    write_summary(args.input_dir / "summary.csv", histories)
    write_plot(args.input_dir / "scheduler_curves.tex", histories)


if __name__ == "__main__":
    main()
