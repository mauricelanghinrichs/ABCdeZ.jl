---
title: 'ABCdeZ.jl: Simulation-based Bayesian inference and model evidence estimation in Julia'
tags:
  - Julia
  - Bayesian inference
  - Approximate Bayesian Computation
  - simulation-based inference
  - likelihood-free methods
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

Bayesian inference provides a statistical framework for understanding complex
phenomena by identifying the most plausible explanations for the observations.
Approximate Bayesian Computation (ABC) applies this framework to cases where
the likelihood function is unavailable, requiring only that models can be
simulated. ABCdeZ.jl is a general-purpose, simulation-based Bayesian
inference package for the Julia programming language, enabling parameter
estimation and model comparison using Sequential Monte Carlo (SMC) methods. In
particular, ABCdeZ.jl provides model evidence estimates for each considered
model individually, enabling flexible and scalable model comparison workflows
in which alternative models can be added or removed without recomputing
previous inferences. ABCdeZ.jl is accompanied by comprehensive documentation, 
an extensive test suite, and accessible example applications.

# Statement of need

Approximate Bayesian Computation (ABC) methods are widely used for Bayesian
parameter inference in scientific applications where likelihood functions are
unavailable or computationally prohibitive. ABC is still costly but
likelihood-free, being based purely on forward simulation of a generative
model. 

In addition to parameter inference for a given model, there is often a need for
ranking alternative models. In the Bayesian setting, this is achieved by
ranking the model evidences, also called marginal likelihoods. This model
comparison requires efficient methods to estimate the model evidence by forward
simulation. As research progresses, models of interest may be added or removed,
so that the evidence should ideally be estimated separately per model, without
the need for costly recomputation as the model set changes. There is currently
no Julia package that provides efficient likelihood-free individual model
evidence estimates.

ABCdeZ.jl was developed to address this gap while also supporting standard
posterior parameter inference. In addition to model comparison and parameter
inference, the package integrates features important for computationally
intensive simulation-based workflows, including straightforward parallelization 
and data blobs that enable reuse of partial simulation results
associated with posterior particles after inference.


# State of the field

Several Approximate Bayesian Computation implementations exist in Julia, 
providing tools for parameter inference, 
including [ApproxBayes.jl](https://github.com/marcjwilliams1/ApproxBayes.jl), 
[SimulationBasedInference.jl](https://github.com/bgroenks96/SimulationBasedInference.jl), 
[SimulatedAnnealingABC.jl](https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl) 
and [GpABC.jl](https://github.com/tanhevg/GpABC.jl) 
[@tankhilevich_gpabc_2020]. 
In some cases, model comparison is also implemented via rejection, Markov 
Chain Monte Carlo, or Sequential Monte Carlo (SMC) methods. In particular, 
approaches such as those implemented in ApproxBayes.jl or GpABC.jl directly 
estimate posterior model probabilities through model-switching Monte Carlo 
moves, foregoing the estimation of individual normalized evidences. 
A similar procedure is used by [pyABC](https://github.com/ICB-DCM/pyABC) 
[@schaelte2022pyabc], a popular ABC package in the Python programming 
language.

Crucially, these implementations require all candidate models to be included
within a single joint inference run. As a consequence, model comparison is tied
to a fixed set of models and relies on parallel evaluation of all candidate
models. Adding or removing models later on therefore requires recomputing the
full inference procedure in order to obtain posterior model probabilities.

This coupling between inference and a fixed model set makes iterative model
development computationally inefficient.


# Software design

ABCdeZ.jl is an open-source, MIT-licensed software package for Approximate
Bayesian Computation written in Julia [@bezanson_julia_2017] and hosted on
[GitHub](`https://github.com/mauricelanghinrichs/ABCdeZ.jl`). 
It is registered in the Julia General Registry and
installable via `] add ABCdeZ`. The package provides a user-friendly API,
comprehensive documentation and minimal working examples for rapid onboarding. 
For example, the code generating \autoref{fig:one} is included in
the documentation and repository. ABCdeZ.jl uses automated continuous
integration (CI) workflows for testing, documentation deployment, and test
coverage reporting. The test suite (`] test ABCdeZ`) achieves 98% coverage
(based on Codecov) within the CI workflow. The package depends only on widely 
used and established Julia libraries `Random`, `Distributions`,
`StatsBase` and `FLoops`, keeping external dependencies minimal to ensure ease
of installation and long-term maintainability.

![Minimal example from the ABCdeZ.jl documentation showcasing parameter inference and model comparison. Two models were independently fitted to a dataset, updating posterior parameter distributions from their priors (a). The estimated model evidences were subsequently used to compute posterior model probabilities from an initially uniform model prior (b). Inference results obtained with ABCdeZ.jl are compared with the exact analytical distributions, which can be derived for this minimal example but are generally unavailable in realistic research applications.\label{fig:one}](fig1.png){ width=70% }

The core idea of ABCdeZ.jl is to infer model evidence estimates per model,
independent of the overall model set. From these model evidences, posterior
model probabilities and Bayes factors can then be computed for arbitrary
subsets of analyzed models, allowing new models to be added without recomputing
previous inferences. 

To enable the estimation of model evidences,
ABCdeZ.jl implements an ABC-SMC framework in which particle weights are tracked
[@didelot_likelihood-free_2011; @del_moral_adaptive_2012], following weight
assignments analogous to those used in likelihood-based SMC inference (e.g.,
@amaya_adaptive_2021). As SMC algorithms handle complex multimodal 
parameter landscapes well [@neal_annealed_2001; @del_moral_sequential_2006], 
high-quality posterior parameter samples are obtained alongside 
the evidence estimation. The algorithm uses differential evolution
[@braak_markov_2006] for parameter proposals and stratified resampling
[@douc_comparison_2005] to maintain particle diversity. 

To enforce the sequential progress of the sampling distribution towards the
posterior, ABCdeZ.jl supports flexible distance constraint kernel definitions,
including an indicator kernel (default) for strict stepwise progress and
continuous distance kernels for softened progress constraints. Lightweight data
blobs enable attachment of auxiliary information to particles throughout
inference, including simulation outputs, simulation-to-data distances, or other
metadata. Efficiency is a core design goal, and care was taken to avoid 
unnecessary memory allocations within the ABC-SMC implementation. For the 
minimal example included in the documentation, inference completes in 
approximately 0.03 seconds, increasing to roughly 0.1 seconds for a 
ten-dimensional parameter prior. These results indicate that, for 
realistic applications, the computational cost is typically dominated 
by the user-defined model simulation and distance evaluation rather 
than by the ABCdeZ.jl framework itself. Thread-safe parallelization via 
`FLoops` enables efficient multi-core execution while accommodating 
fast in-place mutable operations for distance evaluations.

Effective application of ABC methods requires careful design of 
summary statistics and distance functions. ABCdeZ.jl therefore provides 
extensive documentation covering practical aspects of simulation-based 
inference workflows. In addition, the package emphasizes ease of use through 
a simple API centered around a single top-level inference function 
(`abcdesmc!`) and minimal working examples enabling straightforward 
adoption and reproducible workflows.


# Research impact statement

ABCdeZ.jl was developed by the authors to address methodological demands 
arising from interdisciplinary research in systems biology. The software 
contributed substantially to two recent research studies [@frank_holistic_2024;
@ikeda_early_2025], currently under review for publication (as of May 2026). 
The studies exemplify applications of ABCdeZ.jl to infer lineage pathways during 
tissue development in mammals, where large numbers of distinct cell types and 
functional tissues emerge from common progenitor cell types. Resolving such lineage 
pathways requires systematic and scalable model comparison, since the number of possible 
lineage topologies grows combinatorially. \autoref{fig:two} illustrates such a 
workflow for one lineage topology with three terminal cell types (A, B, C). In 
@frank_holistic_2024, a systematic investigation of 86 distinct models was 
performed, representing a scale of model comparison that would be 
computationally demanding with ABC workflows requiring joint 
inference across all candidate models. By enabling per-model evidence 
estimation, ABCdeZ.jl supported iterative and extensible hypothesis testing in 
this large model space. 

As increasingly complex biological datasets become available, we anticipate a
growing demand for statistical inference tools that support scalable model
comparison for simulation-based Bayesian inference. While motivated by systems
biology applications, ABCdeZ.jl is a general-purpose framework that may prove useful 
in other research domains relying on likelihood-free inference.

![Representative workflow of ABCdeZ.jl adapted from a recent research application [@frank_holistic_2024]. ABCdeZ.jl enables inference of mechanistic processes underlying complex experimental data by combining generative forward simulations with large-scale and systematic model comparison. The framework estimates posterior parameter distributions and model evidences, from which posterior model probabilities can be derived to identify the most plausible explanatory processes underlying the observed data.\label{fig:two}](fig2.png){ width=90% }


# AI usage disclosure

Generative AI tools were used to improve grammar, readability and phrasing of
the manuscript. No generative AI tools were used in the development of the
software, the writing of this manuscript, or the preparation of supporting
materials.


# Acknowledgements

We thank Francesco Alemanno, author of the package [KissABC.jl](https://github.com/francescoalemanno/KissABC.jl), 
now publicly archived, for helpful correspondence and for the development of that package, 
parts of whose code base were used as a starting point 
for ABCdeZ.jl. In accordance with the original licensing terms, 
the corresponding shared license file is included in
ABCdeZ.jl.

We thank all members of the Division of Theoretical Systems Biology at the
German Cancer Research Center for testing and using ABCdeZ.jl, thereby
providing valuable feedback for previous releases of ABCdeZ.jl.

# References
