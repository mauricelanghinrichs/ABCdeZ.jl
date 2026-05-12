---
title: 'ABCdeZ.jl: Simulation-based Bayesian inference and model evidence estimation in Julia'
tags:
  - Julia
  - Bayesian inference
  - Approximate Bayesian Computation
  - simulation-based inference
  - likelihood-free inference
  - model comparison
  - model evidence
  - model selection
authors:
  - name: Maurice Langhinrichs
    orcid: 0000-0001-8529-5749
    affiliation: 1 # "1, 2" # (Multiple affiliations must be quoted)
  - name: Thomas Höfer
    orcid: 0000-0003-3560-8780
    affiliation: 1
  - name: Nils B. Becker
    orcid: 0000-0002-7490-6425
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Division of Theoretical Systems Biology, German Cancer Research Center, Heidelberg 69120, Germany
   index: 1
   ror: 04cdgtt98 # https://ror.org/04cdgtt98
date: 9 May 2026
bibliography: paper.bib

---

# Summary

Bayesian inference provides a statistical framework for understanding complex phenomena 
by fitting models to experimental data and identifying the most plausible explanations 
for the observations. Approximate Bayesian Computation (ABC) extends 
this framework to cases where the likelihood function is analytically intractable 
or computationally prohibitive, requiring only that models can be simulated. ABCdeZ.jl 
is a simulation-based Bayesian inference package for the Julia programming language, 
enabling parameter estimation and model comparison using Sequential Monte Carlo (SMC) 
methods. In particular, ABCdeZ.jl provides model evidence estimates for each model 
individually, enabling flexible and scalable model comparison workflows in which alternative models 
can be added or removed without recomputing previous inferences. ABCdeZ.jl is 
accompanied by comprehensive documentation, a full test suite, and accessible 
example applications.

# Statement of need

ABC methods are widely used for Bayesian inference in scientific applications 
where likelihood functions are unavailable or computationally prohibitive. 
There is currently no Julia package that provides independent model evidence 
estimates for individual models, enabling flexible and extensible model comparison 
workflows in which models can be added or removed without recomputing previous inferences.

ABCdeZ.jl was developed to address this gap while also supporting standard 
posterior parameter inference. In addition to model comparison and parameter inference, 
the package integrates features important for computationally intensive 
simulation-based workflows, including straightforward parallelisation and 
data blobs that enable reuse of simulation results associated with 
posterior particles after inference.

# State of the field

Several Approximate Bayesian Computation implementations exist in Julia, 
including ApproxBayes.jl and GpABC.jl [@tankhilevich_gpabc_2020],
which provide tools for parameter inference and, 
in some cases, model comparison via rejection, Markov Chain Monte Carlo, 
or SMC methods. In particular, approaches such as 
those implemented in GpABC.jl allow direct estimation of posterior model 
probabilities without intermediate model evidence calculations.

However, these implementations require all candidate models to be included 
within a single joint inference run. As a consequence, model comparison is 
tied to a fixed set of models and relies on parallel evaluation of all 
candidate models. Adding or removing models later on therefore requires recomputing 
the full inference procedure in order to obtain posterior model probabilities.

This coupling between inference and a fixed model set limits extensibility 
and makes iterative model development computationally inefficient.

# Software design

ABCdeZ.jl is an open-source, MIT-licensed software package for Approximate Bayesian 
Computation written in Julia [@bezanson_julia_2017] and hosted on 
GitHub (`https://github.com/mauricelanghinrichs/ABCdeZ.jl`), with automated continuous 
integration (CI) workflows for testing, documentation deployment, and
code coverage reporting. It is registered in the Julia General Registry 
and installable via `] add ABCdeZ`. The package provides a user-friendly API, 
comprehensive documentation, and minimal working examples for rapid onboarding. 
A reproducible example generating \autoref{fig:one} is included in the documentation 
and repository. The test suite (`] test ABCdeZ`) achieves 98% coverage 
(based on Codecov) within the CI workflow. The package builds on established 
Julia libraries, namely `Random`, `Distributions`, `StatsBase` and `FLoops`, 
while keeping external dependencies minimal to ensure ease of 
installation and long-term maintainability.

![Minimal example from the ABCdeZ.jl documentation showcasing parameter inference and model comparison. Two models were independently fitted to a dataset, updating posterior parameter distributions from their priors (a). The estimated model evidences were subsequently used to compute posterior model probabilities from an initially uniform model prior (b). Inference results obtained with ABCdeZ.jl are compared with the exact analytical distributions, which can be derived for this minimal example but are generally unavailable in realistic research applications.\label{fig:one}](fig1.png){ width=70% }

The core idea of ABCdeZ.jl is to infer model evidence estimates that 
are model-specific and independent of the overall model set. 
From these model evidences, posterior model probabilities and 
Bayes factors can be computed for arbitrary subsets of analysed models, 
allowing new models to be added without recomputing previous inferences.
To enable the estimation of model evidences alongside posterior samples, 
ABCdeZ.jl implements an ABC-SMC framework in which particle weights are tracked 
[@didelot_likelihood-free_2011; @del_moral_adaptive_2012], 
following weight formulations analogous to those used in 
likelihood-based SMC inference (e.g., @amaya_adaptive_2021). The algorithm uses 
differential evolution [@braak_markov_2006] for parameter proposals and 
stratified resampling [@douc_comparison_2005] to maintain particle diversity. 
It supports modular kernel definitions, including an indicator kernel (default) 
and continuous distance kernels. Lightweight data blobs enable attachment of auxiliary
information to particles throughout inference—such as simulation outputs, 
simulation-to-data distances, or other metadata. Thread-safe parallelisation via 
`FLoops` enables efficient multi-core execution.

# Research impact statement

ABCdeZ.jl was developed by the authors to address methodological 
demands arising from interdisciplinary research in systems biology. 
The software contributed substantially to two recent research studies 
[@frank_holistic_2024; @ikeda_early_2025], currently available as 
BioRxiv preprints and under review in peer-reviewed journals (as of May 2026).
The studies exemplify applications in the inference of lineage 
pathways during the development of multicellular organisms, where large 
numbers of distinct cell types and functional tissues emerge from common 
progenitors. Resolving such lineage pathways often requires systematic and 
scalable model comparison, since the number of possible lineage topologies 
grows combinatorially. \autoref{fig:two} illustrates such a workflow for 
one lineage topology with three terminal cell types (A, B, C).
In @frank_holistic_2024, a systematic investigation of 86 distinct models 
was performed, representing a scale of model comparison that would be 
computationally demanding with ABC workflows requiring joint inference 
across all candidate models. By enabling model-specific inference and model 
evidence estimation, ABCdeZ.jl supports iterative and extensible hypothesis 
testing in large model spaces.
As increasingly complex biological datasets become available, we 
anticipate growing demand for statistical inference tools that 
support scalable model comparison for simulation-based Bayesian 
inference. While motivated by systems biology applications, 
ABCdeZ.jl is a general-purpose framework and may therefore also 
prove useful in other research domains relying on 
likelihood-free inference.

![Representative workflow of ABCdeZ.jl adapted from a recent research application [@frank_holistic_2024]. ABCdeZ.jl enables inference of mechanistic processes underlying complex experimental data by combining generative forward simulations with large-scale and systematic model comparison. The framework estimates posterior parameter distributions and model evidences, from which posterior model probabilities can be derived to identify the most plausible explanatory processes underlying the observed data.\label{fig:two}](fig2.png){ width=90% }

# AI usage disclosure

Generative AI tools were used to improve grammar, readability 
and phrasing of the manuscript. Otherwise, no generative AI tools
were used in the development of the software, the writing of this 
manuscript, or the preparation of supporting materials.

# Acknowledgements

We thank Francesco Alemanno, author of the package KissABC.jl, 
now publicly archived, for the development of that package, 
parts of whose code base were used as a starting point for ABCdeZ.jl. 
In accordance with the original licensing terms, the corresponding 
shared license file is included in ABCdeZ.jl.

We thank all members of the Division of Theoretical Systems Biology at 
the German Cancer Research Center for testing and using ABCdeZ.jl, 
thereby providing valuable feedback for previous releases of ABCdeZ.jl.

# References