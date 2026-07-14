#!/usr/bin/env python3
"""Build Overleaf-ready ICAIS English submission from official template."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_SRC = ROOT / "submission_template" / "icais"
BUNDLE = ROOT / "submission_en" / "icais_bundle"
FIG_SRC = ROOT / "figures"
TEX_SRC = ROOT / "submission_en" / "icais_submission.tex"

_LEGACY_TEX = r"""\documentclass{article}
\usepackage{icais_conference,times}
\input{math_commands.tex}
\usepackage{graphicx}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!70!black,citecolor=green!60!black,urlcolor=red!60!black}
\usepackage{url}

\title{Confidence Fallback: Selective Stopping for Continuous Latent Reasoning}

\author{Songyang Pang \\
Beijing Youth AI Academy \\
Experimental School Affiliated to Haidian Teachers Training College \\
\texttt{3478053618@qq.com} \\
Supervisor: Jingwen Fu \\
Beijing Zhongguancun Academy
}

% Youth Scientist Track: show real author (not double-blind anonymous)
\iclrfinalcopy

\begin{document}
\maketitle
% Clear running header set inside maketitle by ICAIS template
\fancyhead[L]{}
\fancyhead[R]{}
\fancyhead[C]{}
\renewcommand{\headrulewidth}{0pt}

\begin{abstract}
Continuous latent reasoning can maintain parallel search frontiers in latent space, yet deployment still requires per-instance step budgeting and selective recovery under uncertainty. We conduct 91 experiments on Coconut/ProsQA graph reasoning. Our prior analysis shows that optimal continuous-thought steps correlate with BFS depth ($r\approx0.543$); fixed-step \texttt{fixed\_3} reaches only 83.8\%, while structure routing improves to 93.6\%, but a single path cannot cover low-confidence cases. We propose \textbf{confidence\_fallback}: an M2 stop head outputs $\mathrm{prob}_0$ to gate main/fallback dual paths, invoking kNN backup search on only $\sim$7.2\% of low-confidence samples. On 419 questions, accuracy reaches \textbf{95.23\%} ($\tau{=}0.48$), +11.4\,pp over \texttt{fixed\_3}; across 53 OOD slices, \texttt{tri\_zone} with hybrid slice routing yields a weighted +2.08\,pp gain.
\end{abstract}

\textbf{Keywords:} confidence fallback; continuous latent reasoning; adaptive stopping; Coconut; ProsQA

\section{Introduction}
Chain-of-thought (CoT) \citep{wei2022cot} reasons step-by-step with discrete tokens, making per-instance step control difficult. Coconut \citep{hao2024coconut} performs continuous-thought reasoning in latent space; its hidden states can encode superposition over multiple search frontiers, enabling implicit parallel breadth-first search (BFS) \citep{zhu2025superposition}. On ProsQA graph reachability \citep{hao2024prosqa}, continuous thought outperforms discrete CoT. Prior work also studies latent-space reasoning and early stopping \citep{goyal2023thoughts,pfau2024dotbydot,zhou2023earlyexit}. However, most studies focus on training-time mechanisms or label-assisted step sweeping, leaving deployable stopping without ground-truth labels under-explored \citep{schuster2022calm}.

CALM \citep{schuster2022calm} adapts computation via output confidence, but Coconut deployment must jointly combine structure routing, stop discrimination, and backup search. On ProsQA we observe three findings: optimal steps align with graph depth; overthinking hurts accuracy (3-step 83.8\% vs.\ 5-step 74.0\%); structure routing reaches 93.6\% yet cannot rescue low-confidence main-path samples. We therefore propose \texttt{confidence\_fallback} and validate it against fixed-step, structure-routing, and online-stopping baselines under both in-distribution and out-of-distribution (OOD) settings.

\section{Problem Setting and Method}
\subsection{Problem Setting}
ProsQA \citep{hao2024prosqa} is a directed-graph reachability task: given a root and two candidate targets, the model must determine which target is reachable. Each instance requires choosing the number of continuous-thought steps $n$ (latent steps). Deployment constraints are: (1) no exhaustive sweep over steps 1--8 at inference time; (2) mean latent steps $\mathrm{mean\_n}\le 4.5$. All experiments use Coconut \texttt{checkpoint\_300} \citep{hao2024coconut}---the weight snapshot saved at training step 300 after ProsQA fine-tuning---evaluated on 419 in-distribution questions and 53 OOD slices.

\subsection{Confidence Fallback}
The method keeps an efficient main path via structure routing and gates whether to invoke fallback for uncertain samples (Figure~\ref{fig:flow}). (1) \textbf{Main path:} estimate depth $d$ by BFS, set $n_0{=}\mathrm{clamp}(d)$, and obtain $\mathrm{pred}_0$ in one forward pass. (2) \textbf{Confidence:} the M2 stop head outputs $\mathrm{prob}_0{=}\sigma(\mathrm{M2}(h_{n_0},n_0,x))$, estimating confidence in stopping at $n_0$. (3) \textbf{Gating:} if $\mathrm{prob}_0\ge\tau$, return $\mathrm{pred}_0$; otherwise trigger the kNN+M2 fallback path. (4) \textbf{Threshold:} $\tau{=}0.48$ is selected by sweeping $[0.42,0.55]$ on the validation set.

\begin{figure}[t]
\begin{center}
\includegraphics[width=0.92\linewidth]{figures/figure1_confidence_fallback_flow.png}
\end{center}
\caption{Inference flow of \texttt{confidence\_fallback}. kNN fallback is triggered when M2 confidence falls below $\tau$.}
\label{fig:flow}
\end{figure}

\noindent\textbf{Algorithm 1} \texttt{confidence\_fallback}\\
\textbf{Input:} $x,\tau$, Coconut, M2, kNN \quad \textbf{Output:} $y$\\
1: $n_0\leftarrow\max(\mathrm{min\_n},\min(\mathrm{BFS\_depth}(x),\mathrm{cap}))$; $\mathrm{pred}_0\leftarrow\mathrm{Coconut.forward}(x,n_0)$\\
2: $\mathrm{prob}_0\leftarrow\sigma(\mathrm{M2}(h_{n_0},n_0,x))$\\
3: if $\mathrm{prob}_0<\tau$ then $y\leftarrow\mathrm{KNN\_M2\_online\_stop}(x)$ else $y\leftarrow\mathrm{pred}_0$\\
4: return $y$

\subsection{Tri-Zone Gating and Cross-Distribution Deployment}
For in-distribution ProsQA we fix $\tau{=}0.48$. Under distribution shift, a single $\tau$ may over-trigger fallback. We introduce \texttt{tri\_zone} ($t_{\mathrm{low}}{=}0.40$, $t_{\mathrm{mid}}{=}0.48$): trust the main path if $\mathrm{prob}_0\ge0.48$; fallback if $\mathrm{prob}_0<0.40$; in the middle band, fallback only when the predicted answer flips. We further combine slice-specific rules in \texttt{hybrid\_slice\_router}: vulnerable OOD slices use skip/agreement strategies, while other slices default to \texttt{tri\_zone}.

\section{Discussion}
Expected accuracy decomposes as $p=p_0(1-f)+p_1 f$, where $p_0$ and $p_1$ are main-path and fallback-path accuracies and $f$ is the fallback rate. When $p_1>p_0$ and $f$ is constrained by $\tau$, accuracy improves at bounded computational cost. Structure routing raises $p_0$ to 93.6\%; \texttt{confidence\_fallback} further lifts $p$ to 95.23\% with $f{=}7.2\%$. Unlike \texttt{knn\_min3}, we trust high-confidence main paths and pay extra compute only when M2 is uncertain. Five-seed robustness ($\mu{=}93.89\%$) shows that $\tau{=}0.48$ is not a one-off hyperparameter choice; OOD gains (+2.08\,pp weighted; +7.44\,pp on the OOD subset) indicate practical utility under distribution shift.

\section{Experiments}
\subsection{Setup}
We compare five methods: \texttt{fixed\_3}, \texttt{auto\_route}, \texttt{structure\_d}, \texttt{knn\_min3}, and \texttt{confidence\_fallback}. Metrics include full-set accuracy, $\Delta$ relative to \texttt{fixed\_3}, and fallback rate. Cross-distribution results report weighted $\Delta$ over 53 OOD slices.

\subsection{Main Results}
Figure~\ref{fig:bar} and Table~\ref{tab:main} summarize results on 419 questions. \texttt{confidence\_fallback} achieves 95.23\% (highest), a 7.2\% fallback rate, and $\sim$93\% single-forward samples---+1.6\,pp over \texttt{structure\_d} and +11.4\,pp over \texttt{fixed\_3}.

\begin{figure}[t]
\begin{center}
\includegraphics[width=0.78\linewidth]{figures/figure2_main_results_bar.png}
\end{center}
\caption{Accuracy comparison on ProsQA (419 questions; dark blue bar: ours).}
\label{fig:bar}
\end{figure}

\begin{table}[t]
\caption{Main methods vs.\ baselines (ProsQA, 419 questions).}
\label{tab:main}
\begin{center}
\begin{tabular}{llcccc}
\multicolumn{1}{c}{\bf Method} & \multicolumn{1}{c}{\bf Type} & \multicolumn{1}{c}{\bf Acc.(\%)} & \multicolumn{1}{c}{\bf $\Delta$(pp)} & \multicolumn{1}{c}{\bf Fallback(\%)} & \multicolumn{1}{c}{\bf Mechanism} \\
\hline \\
fixed\_3 & Baseline & 83.8 & --- & --- & Fixed steps \\
auto\_route & Control & 93.1 & +9.3 & --- & BFS routing \\
structure\_d & Control & 93.6 & +9.8 & --- & Depth budget \\
knn\_min3 & Control & 92.6 & +8.8 & --- & kNN stopping \\
\textbf{confidence\_fallback} & \textbf{Ours} & \textbf{95.23} & \textbf{+11.4} & \textbf{7.2} & \textbf{Gated fallback} \\
\end{tabular}
\end{center}
\end{table}

\subsection{Analysis}
\texttt{fixed\_3} ignores instance heterogeneity and is limited on mixed-hop subsets. \texttt{auto\_route} and \texttt{structure\_d} fix step budgeting but cannot correct unreliable stopping despite reasonable step budgets. \texttt{knn\_min3} applies neighborhood-based stopping to all instances. \texttt{confidence\_fallback} retains the efficient \texttt{structure\_d} main path and invokes fallback on only $\sim$7.2\% low-confidence samples, achieving the best accuracy--compute trade-off. \texttt{tri\_zone} and hybrid slice routing further reduce false fallback triggers on OOD slices.

\section{Conclusion}
We present \texttt{confidence\_fallback}, an M2-gated main/fallback dual-path method that reaches 95.23\% on ProsQA (+11.4\,pp over \texttt{fixed\_3}). In-distribution deployment uses $\tau{=}0.48$; cross-distribution deployment combines \texttt{tri\_zone} and \texttt{hybrid\_slice\_router}. Its key advantage is \emph{selective correction} rather than global fixed steps or universal fine-grained stopping---a reusable recipe for deployable stopping in continuous latent reasoning.

\bibliography{icais_submission}
\bibliographystyle{icais_conference}

\end{document}
"""


def load_tex() -> str:
    if TEX_SRC.is_file():
        return TEX_SRC.read_text(encoding="utf-8")
    return _LEGACY_TEX


BIB = r"""@article{hao2024coconut,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Gu, Xin and Yang, Yihan and others},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
@inproceedings{wei2022cot,
  title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models},
  author={Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and others},
  booktitle={NeurIPS},
  year={2022}
}
@article{hao2024prosqa,
  title={ProsQA: Proof with Search Question-Answering},
  author={Hao, Shibo and others},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
@inproceedings{schuster2022calm,
  title={Confident Adaptive Language Modeling},
  author={Schuster, Tal and Fisch, Adam and Gupta, Jai and others},
  booktitle={NeurIPS},
  year={2022}
}
@article{zhu2025superposition,
  title={Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought},
  author={Zhu, Hanlin and Hao, Shibo and Hu, Zhiting and others},
  journal={arXiv preprint arXiv:2505.12514},
  year={2025}
}
@article{goyal2023thoughts,
  title={Thoughts Are All You Need: Exploiting Latent Space for LLM Reasoning},
  author={Goyal, Sachin and Li, Jiateng and Gu, Xin and others},
  journal={arXiv preprint arXiv:2311.01465},
  year={2023}
}
@article{pfau2024dotbydot,
  title={Let's Think Dot by Dot: Hidden Computation in Transformer Language Models},
  author={Pfau, James and Merrill, William and Bowman, Samuel},
  journal={arXiv preprint arXiv:2404.15758},
  year={2024}
}
@article{zhou2023earlyexit,
  title={Efficient Prompting via Dynamic Early Exiting in Large Language Models},
  author={Zhou, Wangchunshu and Xu, Canwen and Ge, Tao and others},
  journal={arXiv preprint arXiv:2310.07463},
  year={2023}
}
"""


def main():
    if not TEMPLATE_SRC.is_dir():
        raise SystemExit(f"Missing template dir: {TEMPLATE_SRC}. Upload icais.zip first.")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    sty_names = [
        "icais_conference.bst",
        "fancyhdr.sty",
        "natbib.sty",
        "math_commands.tex",
    ]
    for name in sty_names:
        shutil.copy2(TEMPLATE_SRC / name, BUNDLE / name)

    sty_text = (TEMPLATE_SRC / "icais_conference.sty").read_text(encoding="utf-8")
    sty_text = sty_text.replace(
        r"\lhead{Published as a conference paper at ICAIS}",
        r"\lhead{}",
    )
    (BUNDLE / "icais_conference.sty").write_text(sty_text, encoding="utf-8")

    fig_dst = BUNDLE / "figures"
    fig_dst.mkdir()
    tex_fig_src = ROOT / "submission_en" / "figures"
    for name in ["figure1_flow.tex", "figure2_bar.tex"]:
        shutil.copy2(tex_fig_src / name, fig_dst / name)
    for png in [
        "figure1_confidence_fallback_flow.png",
        "figure2_main_results_bar.png",
    ]:
        src = FIG_SRC / png
        if src.is_file():
            shutil.copy2(src, fig_dst / png)

    (BUNDLE / "icais_submission.tex").write_text(load_tex(), encoding="utf-8")
    (BUNDLE / "icais_submission.bib").write_text(BIB, encoding="utf-8")

    # Overleaf main file alias
    (BUNDLE / "main.tex").write_text(
        "% Upload this folder to Overleaf. Set main document to icais_submission.tex\n"
        "\\input{icais_submission.tex}\n",
        encoding="utf-8",
    )

    # Zip for easy download
    zip_path = Path(
        shutil.make_archive(str(ROOT / "submission_en" / "ICAIS2026_English_Overleaf"), "zip", BUNDLE)
    )
    print(f"Wrote {BUNDLE / 'icais_submission.tex'}")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
