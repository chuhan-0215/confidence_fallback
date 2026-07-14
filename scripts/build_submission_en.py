#!/usr/bin/env python3
"""Build English ICAIS submission (LaTeX) from optimized Chinese source."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submission_en"
FIG = ROOT / "figures"

PAPER_TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{times}

\title{Confidence Fallback: Selective Stopping for Continuous Latent Reasoning}
\author{Songyang Pang\\
\small Beijing Youth AI Academy / Experimental School Affiliated to Haidian Teachers Training College\\
\small Supervisor: Jingwen Fu\\
\small Beijing Zhongguancun Academy}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Continuous latent reasoning can maintain parallel search frontiers in latent space, yet deployment still requires per-instance step budgeting and selective recovery under uncertainty. We conduct 91 experiments on Coconut/ProsQA graph reasoning. Our prior analysis shows that optimal continuous-thought steps correlate with BFS depth ($r\approx0.543$); fixed-step \texttt{fixed\_3} reaches only 83.8\%, while structure routing improves to 93.6\%, but a single path cannot cover low-confidence cases. We propose \textbf{confidence\_fallback}: an M2 stop head outputs $\mathrm{prob}_0$ to gate main/fallback dual paths, invoking kNN backup search on only $\sim$7.2\% of low-confidence samples. On 419 questions, accuracy reaches \textbf{95.23\%} ($\tau{=}0.48$), +11.4\,pp over \texttt{fixed\_3}; across 53 OOD slices, \texttt{tri\_zone} with hybrid slice routing yields a weighted +2.08\,pp gain.
\end{abstract}

\noindent\textbf{Keywords:} confidence fallback; continuous latent reasoning; adaptive stopping; Coconut; ProsQA

\section{Introduction}
Chain-of-thought (CoT)~\cite{wei2022cot} reasons step-by-step with discrete tokens, making per-instance step control difficult. Coconut~\cite{hao2024coconut} performs continuous-thought reasoning in latent space; its hidden states can encode superposition over multiple search frontiers, enabling implicit parallel BFS~\cite{zhu2025superposition}. On ProsQA graph reachability~\cite{hao2024prosqa}, continuous thought outperforms discrete CoT. Prior work also studies latent-space reasoning and early stopping~\cite{goyal2023thoughts,pfau2024dotbydot,zhou2023earlyexit}. However, most studies focus on training-time mechanisms or label-assisted step sweeping, leaving deployable stopping without ground-truth labels under-explored~\cite{schuster2022calm}.

CALM~\cite{schuster2022calm} adapts computation via output confidence, but Coconut deployment must combine structure routing, stop discrimination, and backup search. On ProsQA we observe: optimal steps align with graph depth; overthinking hurts accuracy (3-step 83.8\% vs.\ 5-step 74.0\%); structure routing reaches 93.6\% yet cannot rescue low-confidence main-path samples. We therefore propose \texttt{confidence\_fallback} and validate it against fixed-step, structure-routing, and online-stopping baselines in both in-distribution and cross-distribution settings.

\section{Problem Setting and Method}
\subsection{Problem Setting}
ProsQA~\cite{hao2024prosqa} asks whether a target node is reachable from a root in a directed graph. Each instance requires choosing continuous-thought steps $n$ (latent steps). Deployment constraints: (1) no exhaustive sweep over steps 1--8; (2) mean latent steps $\mathrm{mean\_n}\le 4.5$. All experiments use Coconut \texttt{checkpoint\_300}~\cite{hao2024coconut}---weights saved at training step 300 after ProsQA fine-tuning---on 419 in-distribution questions and 53 OOD slices.

\subsection{confidence\_fallback}
The method keeps an efficient main path via structure routing and gates fallback for uncertain samples (Figure~\ref{fig:flow}). (1) \textbf{Main path:} estimate depth $d$ by BFS, set $n_0{=}\mathrm{clamp}(d)$, one forward pass yields $\mathrm{pred}_0$. (2) \textbf{Confidence:} M2 stop head outputs $\mathrm{prob}_0{=}\sigma(\mathrm{M2}(h_{n_0},n_0,x))$, the confidence of stopping at $n_0$. (3) \textbf{Gating:} if $\mathrm{prob}_0\ge\tau$, output $\mathrm{pred}_0$; else trigger kNN+M2 fallback. (4) \textbf{Threshold:} $\tau{=}0.48$ from validation sweep $0.42$--$0.55$.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{../figures/figure1_confidence_fallback_flow.png}
  \caption{Inference flow of \texttt{confidence\_fallback}. kNN fallback triggers when M2 confidence falls below $\tau$.}
  \label{fig:flow}
\end{figure}

\begin{quote}\small
\textbf{Algorithm 1} \texttt{confidence\_fallback}\\
\textbf{Input:} $x,\tau$, Coconut, M2, kNN \quad \textbf{Output:} $y$\\
1: $n_0\leftarrow\max(\mathrm{min\_n},\min(\mathrm{BFS\_depth}(x),\mathrm{cap}))$; $\mathrm{pred}_0\leftarrow\mathrm{Coconut.forward}(x,n_0)$\\
2: $\mathrm{prob}_0\leftarrow\sigma(\mathrm{M2}(h_{n_0},n_0,x))$\\
3: if $\mathrm{prob}_0<\tau$ then $y\leftarrow\mathrm{KNN\_M2\_online\_stop}(x)$ else $y\leftarrow\mathrm{pred}_0$\\
4: return $y$
\end{quote}

\subsection{Tri-zone Gating and Cross-distribution Deployment}
On in-distribution ProsQA we fix $\tau{=}0.48$. Under distribution shift, a single $\tau$ may over-trigger fallback. We introduce \texttt{tri\_zone} ($t_{\mathrm{low}}{=}0.40$, $t_{\mathrm{mid}}{=}0.48$): trust the main path if $\mathrm{prob}_0\ge0.48$; fallback if $\mathrm{prob}_0<0.40$; in the middle band, fallback only when the answer flips. Combined with slice rules as \texttt{hybrid\_slice\_router} for known hurt slices.

\section{Discussion}
Expected accuracy decomposes as $p=p_0(1-f)+p_1 f$, where $p_0,p_1$ are main/fallback accuracies and $f$ is fallback rate. When $p_1>p_0$ and $f$ is constrained by $\tau$, accuracy improves at bounded cost. Structure routing raises $p_0$ to 93.6\%; \texttt{confidence\_fallback} further lifts $p$ to 95.23\% with $f{=}7.2\%$. Unlike \texttt{knn\_min3}, we trust high-confidence main paths and pay extra compute only when M2 is uncertain. Five-seed robustness ($\mu{=}93.89\%$) shows $\tau{=}0.48$ is not a one-off tuning; OOD gains (+2.08\,pp weighted; +7.44\,pp on OOD subset) indicate practical cross-distribution utility.

\section{Experiments}
\subsection{Setup}
We compare five methods: \texttt{fixed\_3}, \texttt{auto\_route}, \texttt{structure\_d}, \texttt{knn\_min3}, and \texttt{confidence\_fallback}. Metrics: full-set accuracy, $\Delta$ vs.\ \texttt{fixed\_3}, and fallback rate. Cross-distribution results report weighted $\Delta$ over 53 OOD slices.

\subsection{Main Results}
Figure~\ref{fig:bar} and Table~\ref{tab:main} summarize 419 questions. \texttt{confidence\_fallback} achieves 95.23\% (highest), 7.2\% fallback, and $\sim$93\% single-forward samples---+1.6\,pp over \texttt{structure\_d} and +11.4\,pp over \texttt{fixed\_3}.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.82\linewidth]{../figures/figure2_main_results_bar.png}
  \caption{Accuracy comparison on ProsQA (419 questions; dark blue bar: ours).}
  \label{fig:bar}
\end{figure}

\begin{table}[t]
  \centering
  \caption{Main methods vs.\ baselines (ProsQA, 419 questions).}
  \label{tab:main}
  \small
  \begin{tabular}{llcccc}
    \toprule
    Method & Type & Acc.(\%) & $\Delta$(pp) & Fallback(\%) & Mechanism \\
    \midrule
    fixed\_3 & Baseline & 83.8 & --- & --- & Fixed steps \\
    auto\_route & Control & 93.1 & +9.3 & --- & BFS routing \\
    structure\_d & Control & 93.6 & +9.8 & --- & Depth budget \\
    knn\_min3 & Control & 92.6 & +8.8 & --- & kNN stop \\
    \textbf{confidence\_fallback} & \textbf{Ours} & \textbf{95.23} & \textbf{+11.4} & \textbf{7.2} & \textbf{Gated fallback} \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Analysis}
\texttt{fixed\_3} ignores instance heterogeneity. \texttt{auto\_route} and \texttt{structure\_d} fix step budgeting but not unstable stopping. \texttt{knn\_min3} applies neighborhood stopping to all instances. \texttt{confidence\_fallback} retains the efficient \texttt{structure\_d} main path and fallbacks on $\sim$7.2\% low-confidence samples, achieving the best accuracy--compute trade-off. \texttt{tri\_zone} and hybrid routing further reduce false triggers on OOD slices.

\section{Conclusion}
We present confidence fallback: M2-gated main/fallback dual paths reaching 95.23\% on ProsQA (+11.4\,pp over \texttt{fixed\_3}). In-distribution deployment uses $\tau{=}0.48$; cross-distribution deployment combines \texttt{tri\_zone} and hybrid slice routing. The key advantage is \emph{selective correction} rather than global fixed steps or universal fine stopping---a reusable recipe for deployable stopping in continuous latent reasoning.

\bibliographystyle{plain}
\bibliography{refs}

\end{document}
"""

BIB = r"""@article{hao2024coconut,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and others},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
@inproceedings{wei2022cot,
  title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models},
  author={Wei, Jason and others},
  booktitle={NeurIPS},
  year={2022}
}
@article{hao2024prosqa,
  title={ProsQA (Proof with Search Question-Answering)},
  author={Hao, Shibo and others},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
@inproceedings{schuster2022calm,
  title={Confident Adaptive Language Modeling},
  author={Schuster, Tal and others},
  booktitle={NeurIPS},
  year={2022}
}
@article{zhu2025superposition,
  title={Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought},
  author={Zhu, Hanlin and others},
  journal={arXiv preprint arXiv:2505.12514},
  year={2025}
}
@article{goyal2023thoughts,
  title={Thoughts Are All You Need: Exploiting Latent Space for LLM Reasoning},
  author={Goyal, Sachin and others},
  journal={arXiv preprint arXiv:2311.01465},
  year={2023}
}
@article{pfau2024dotbydot,
  title={Let's Think Dot by Dot: Hidden Computation in Transformer Language Models},
  author={Pfau, James and others},
  journal={arXiv preprint arXiv:2404.15758},
  year={2024}
}
@article{zhou2023earlyexit,
  title={Efficient Prompting via Dynamic Early Exiting in Large Language Models},
  author={Zhou, Wangchunshu and others},
  journal={arXiv preprint arXiv:2310.07463},
  year={2023}
}
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "paper.tex").write_text(PAPER_TEX, encoding="utf-8")
    (OUT_DIR / "refs.bib").write_text(BIB, encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'paper.tex'}")
    print(f"Wrote {OUT_DIR / 'refs.bib'}")

    # Try compile if pdflatex available
    import shutil
    import subprocess

    if shutil.which("pdflatex") and shutil.which("bibtex"):
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                cwd=OUT_DIR,
                check=False,
                capture_output=True,
            )
        subprocess.run(["bibtex", "paper"], cwd=OUT_DIR, check=False, capture_output=True)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
            cwd=OUT_DIR,
            check=False,
            capture_output=True,
        )
        if (OUT_DIR / "paper.pdf").exists():
            print(f"Wrote {OUT_DIR / 'paper.pdf'}")
    else:
        print("pdflatex not found; upload paper.tex to Overleaf with official template.")


if __name__ == "__main__":
    main()
