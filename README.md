## ABCdeZ.jl

[![CI](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mauricelanghinrichs/ABCdeZ.jl/branch/main/graph/badge.svg?token=BZ86DWE65S)](https://codecov.io/gh/mauricelanghinrichs/ABCdeZ.jl)
[![Documentation](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/Documentation.yml/badge.svg?branch=main)](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/Documentation.yml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mauricelanghinrichs.github.io/ABCdeZ.jl/dev/)
<!--- ACTIVATE THIS ONCE READY: [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mauricelanghinrichs.github.io/ABCdeZ.jl/stable/) --->

Approximate Bayesian Computation (**ABC**) with differential evolution (**de**) moves and model evidence (**Z**) estimates.

ABCdeZ.jl offers Bayesian parameter estimation and model comparison/selection for inference problems with an intractable likelihood. Models only need to be simulated (to replace the likelihood requirement). Please visit the documentation ([dev](https://mauricelanghinrichs.github.io/ABCdeZ.jl/dev/)/stable) to get started.

<img src="docs/src/assets/abcdez_min_ex_post.png" width="539">

The documentation will go through a minimal example (code also found in `examples` folder above) computing the posterior samples (Figure above) and evidences for two different models. Model evidences can be used to derive posterior model probabilities (Figure below) or Bayes Factors.

<img src="docs/src/assets/abcdez_min_ex_model_sel.png" width="305">

ABCdeZ.jl was developed [@TSB](https://www.dkfz.de/en/modellierung-biologischer-systeme/) by [Maurice Langhinrichs](mailto:m.langhinrichs@icloud.com) and Nils Becker. This work is based on many people's previous achievements, particular some part of the code base was adapted from [KissABC.jl](https://github.com/francescoalemanno/KissABC.jl); please visit the documentation for a complete list of references.