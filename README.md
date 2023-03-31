# ABCdeZ.jl

[![CI](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mauricelanghinrichs/ABCdeZ.jl/branch/main/graph/badge.svg?token=BZ86DWE65S)](https://codecov.io/gh/mauricelanghinrichs/ABCdeZ.jl)
[![Documentation](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/Documentation.yml/badge.svg?branch=main)](https://github.com/mauricelanghinrichs/ABCdeZ.jl/actions/workflows/Documentation.yml)
<!--- ACTIVATE THIS ONCE READY: [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mauricelanghinrichs.github.io/ABCdeZ.jl/stable/) --->
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mauricelanghinrichs.github.io/ABCdeZ.jl/dev/)


# sync/document github
# - two example files for both alg.
# - add docs
# - describe briefly: demc greedy version (biased before 
#   completion but fast), desmc (accurate, a bit slower, but also 
#   evidence estimate (up to (final) kernel norm))
# - specialities: 
#    - link to varexternal usage (maybe my julia discourse)
#    - blobs feature
# - explain all the caveats of ABC-based evidence values 
#    - same ϵ target necessary (if not possible upper bound conservative 
#       estimate may be possible, or, use ϵs and logZs lists for finding 
#       last common ϵ to compare with)
#    - ABC (kernel instead of likelihood) and summary stats both introduce approximation errors
#    - summary stats need to be sufficient for model selection 
#       (it is not enough if summary stats are sufficient for 
#        each model's parameters!), link to paper and 
#        stackoverflow topic
# - uncertainy estimate for logZ: currently not available from 
#   a single run; users may run it multiple times to get it
# - basic references: didelot, geological paper, extensive review,
#                     summary stats paper, KissABC, LikelihoodFreeInference,
#                       DE moves (Cajo J. F. Ter Braak, Turner)
