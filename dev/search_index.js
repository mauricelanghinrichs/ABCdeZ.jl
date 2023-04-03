var documenterSearchIndex = {"docs":
[{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"CurrentModule = ABCdeZ","category":"page"},{"location":"#ABCdeZ.jl","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"","category":"page"},{"location":"#Introduction","page":"ABCdeZ.jl","title":"Introduction","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"two example files for both alg. ? or one extended minimal example (also changing   script in examples...)\ndescribe briefly: demc greedy version (biased before  completion but fast), desmc (accurate, a bit slower, but also  evidence estimate (up to (final) kernel norm))\nlink to all the caveats of ABC-based evidence values","category":"page"},{"location":"#Minimal-example","page":"ABCdeZ.jl","title":"Minimal example","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"<img src=\"./src/assets/abcdezminex_post.png\" width=\"539\">","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"<img src=\"./src/assets/abcdezminexmodelsel.png\" width=\"305\">","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"this code here should match the script in examples folder on github;  and then maybe on github just put the png image","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"current idea: use current example but add a second model  with a slightly larger Normal prior (and also change model / likelihood / simulation itself? maybe  variation in the Normal from 1.0 to something?); try to calculate  evidence also for the second model analytical; in the end make  a nice png with three plots (posterior model 1 and 2 (from the different priors) and a third  plot showing the model probabilities (or evidences?) between smc output and analytical also);  also as a final line compute posterior by abcdemc! version (just for the posterior,  not likelihood)","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"model comparison can be done with the model evidences obtained by  abcdesmc!; as the evidence values are (typically) off by the  normalisation constant, different models (each inferred from a separate  ABC run) have to be compared with logZs computed for the SAME ϵ  (then the unknown normalisation constants cancel and don't matter) (of course, next to the same data and distance method);\nfor example posterior model probabilities are computed like this  (for a uniform model prior)","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"    m1_evidence = evidence # evidence of model 1 from above\n    m2_evidence = 0.008 # from some other ABC run\n\n    m1_prior = 0.5 # model priors (uniform here)\n    m2_prior = 0.5\n\n    m1_prob = m1evid*m1_prior / (m1evid*m1_prior + m2evid*m2_prior) # posterior prob. model 1\n    m2_prob = m2evid*m2_prior / (m1evid*m1_prior + m2evid*m2_prior) # posterior prob. model 2","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"note: Prior\nMultidimensional prior distributions (continuous, discrete or mixed) can  be specified via the Factored() syntax (from  independent 1d marginals), e.g. prior2d = Factored(Normal(0, sqrt(10)),  DiscreteUniform(1, 10)).","category":"page"},{"location":"#Inference-by-abcdesmc!","page":"ABCdeZ.jl","title":"Inference by abcdesmc!","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"abcdesmc!","category":"page"},{"location":"#ABCdeZ.abcdesmc!","page":"ABCdeZ.jl","title":"ABCdeZ.abcdesmc!","text":"abcdesmc!(prior, dist!, ϵ_target, varexternal; <keyword arguments>)\n\nRun ABC with diffential evolution (de) moves in a sequential Monte Carlo setup (smc)  providing posterior samples and a model evidence estimate.\n\nThe particles have to be weighted (via r.Wns) for valid posterior samples.\n\nArguments\n\nprior: Distribution or Factored object specifying the parameter prior.\ndist!: distance function computing the distance (≥ 0.0) between model and data,    for given (θ, ve) input (θ parameters, ve external variables, see varexternal).\nϵ_target: final target distance (or more general, target width of the ABC kernel); algorithm    stops if ϵ_target or nsims_max is reached.\nvarexternal: external variables that are passed as second positional argument to dist!    and can be used to support the distance computation with fast in-place operations in    a thread-safe manner; objects in varexternal can be in-place mutated, even in parallel mode,    as each thread will receive its own copy of varexternal (if not needed input nothing).\nnparticles::Int=100: number of total particles to use for inference.\nα=0.95: used for adaptive choice of ϵ specifying the sequential target distributions; technically,    ϵ will be the α-quantile of current particle distances.\nδess=0.5: if the fractional effective sample size drops below δess, a stratified resampling step is performed.\nnsims_max::Int=10^7: maximal number of dist! evaluations (not counting initial samples from prior);    algorithm stops if ϵ_target or nsims_max is reached.\nKmcmc::Int=3: number of MCMC (Markov chain Monte Carlo) steps at each sequential    target distribution specified by current ϵ and ABC kernel type.\nABCk=ABCdeZ.Indicator0toϵ: ABC kernel to be specified by ϵ widths that receives distance values.\nfacc_min=0.25: if the fraction of accepted MCMC proposals drops below facc_min, diffential evolution    proposals are reduced by a factor of facc_tune.\nfacc_tune=0.95: factor to reduce the jump distance of the diffential evolution    proposals in the MCMC step (used if facc_min is reached).\nverbose::Bool=true: if set to true, enables verbosity (printout to REPL).\nverboseout::Bool=true: if set to true, algorithm returns a more detailed inference output.\nrng=Random.GLOBAL_RNG: an AbstractRNG object which is used by the inference.\nparallel::Bool=false: if set to true, threaded parallelism is enabled; dist! must be    thread-safe in such a case, e.g. by making use of varexternal (ve).\n\nExamples\n\njulia> using ABCdeZ, Distributions;\njulia> data = 5;\njulia> prior = Normal(0, sqrt(10));\njulia> model(θ) = rand(Normal(θ, 1));\njulia> dist!(θ, ve) = abs(model(θ)-data), nothing;\njulia> ϵ = 0.3;\njulia> r = abcdesmc!(prior, dist!, ϵ, nothing, nparticles=1000, parallel=true);\njulia> posterior = [t[1] for t in r.P[r.Wns .> 0.0]];\njulia> evidence = exp(r.logZ);\n\n\n\n\n\n","category":"function"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"warning: Posterior sample weights\nPosterior samples obtained by abcdesmc! have to be associated with their weights (r.Wns).  With an indicator ABC kernel (default) there are just two weights (i.e. alive and dead  particles) and the correct posterior samples are hence given by  posterior = [t[1] for t in r.P[r.Wns .> 0.0]] (for the first parameter here).","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"warning: Model evidence (off by a factor)\nThe model evidence estimates from the abcdesmc! method obtained by the default ABC  indicator kernel are off by a normalisation factor coming from an unnormalised  kernel (the one used in the final iteration). To do model selection / comparison  this means that evidence estimates for the set of models have to be done for the same  data (or summary statistics), distance function, ABC kernel and the same target  ϵ (which is ϵ_target if run not stopped by nsims_max). Then the (unknown)  normalisation factor is the same for all models and does not matter (cancels) for  Bayes factors or posterior model probabilities. See here for workarounds if ϵ is not the same.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"tip: Uncertainty of model evidence\nAs of now the abcdesmc! method does not provide a (numerical) uncertainty for the model evidence  estimate from a single run. It may be however very useful to check for this when doing  model comparison (as the resulting Bayes factors or posterior model probabilities are  uncertain as well). So, if the runtime permits, run the abcdesmc! method multiple times  and collect the resulting set of evidence values for mean/median and std estimates.","category":"page"},{"location":"#Inference-by-abcdemc!","page":"ABCdeZ.jl","title":"Inference by abcdemc!","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"abcdemc!","category":"page"},{"location":"#ABCdeZ.abcdemc!","page":"ABCdeZ.jl","title":"ABCdeZ.abcdemc!","text":"abcdemc!(prior, dist!, ϵ_target, varexternal; <keyword arguments>)\n\nRun ABC with diffential evolution (de) moves in a Markov chain Monte Carlo setup (mc)  providing posterior samples.\n\nAlgorithm needs to converge for an unbiased posterior estimate.\n\nArguments\n\nprior: Distribution or Factored object specifying the parameter prior.\ndist!: distance function computing the distance (≥ 0.0) between model and data,    for given (θ, ve) input (θ parameters, ve external variables, see varexternal).\nϵ_target: final target distance (or more general, target width of the ABC kernel); algorithm    equilibrates to final target distribution (approximate posterior) if ϵ_target is reached.\nvarexternal: external variables that are passed as second positional argument to dist!    and can be used to support the distance computation with fast in-place operations in    a thread-safe manner; objects in varexternal can be in-place mutated, even in parallel mode,    as each thread will receive its own copy of varexternal (if not needed input nothing).\nnparticles::Int=50: number of total particles to use for inference in each generation.\ngenerations::Int=20: number of generations (total iterations) to run the algorithm.\nverbose::Bool=true: if set to true, enables verbosity (printout to REPL).\nrng=Random.GLOBAL_RNG: an AbstractRNG object which is used by the inference.\nparallel::Bool=false: if set to true, threaded parallelism is enabled; dist! must be    thread-safe in such a case, e.g. by making use of varexternal (ve).\n\nExamples\n\njulia> using ABCdeZ, Distributions;\njulia> data = 5;\njulia> prior = Normal(0, sqrt(10));\njulia> model(θ) = rand(Normal(θ, 1));\njulia> dist!(θ, ve) = abs(model(θ)-data), nothing;\njulia> ϵ = 0.3;\njulia> r = abcdemc!(prior, dist!, ϵ, nothing, nparticles=1000, generations=300, parallel=true);\njulia> posterior = [t[1] for t in r.P];\n\n\n\n\n\n","category":"function"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"warning: Greediness of abcdemc!\nThe abcdemc! method implements a \"greedy\"/biased Metropolis-Hasting step in the  Markov chain. This allows a fast convergence, particularly well-suited for unimodal  problems. However, to obtain valid posterior estimates the algorithm needs to  converge (all particles below ϵ_target and r.reached_ϵ==true). Otherwise the samples will be  biased (closer to the MAP (maximum a posteriori probability) parameter values  with reduced variation).","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"note: Note\nIn contrast to the abcdesmc! method the resulting posterior samples of the abcdemc!  method are not associated with weights and can be used directly, i.e.  posterior = [t[1] for t in r.P] (for the first parameter here).","category":"page"},{"location":"#Distributions-and-Priors","page":"ABCdeZ.jl","title":"Distributions and Priors","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"Factored","category":"page"},{"location":"#ABCdeZ.Factored","page":"ABCdeZ.jl","title":"ABCdeZ.Factored","text":"Factored{N} <: Distribution{Multivariate, MixedSupport}\n\nA Distribution type that can be used to combine multiple UnivariateDistribution's (independently).\n\nExamples\n\njulia> prior = Factored(Normal(0, 1), Uniform(-1, 1))\nFactored{2}(\np: (Normal{Float64}(μ=0.0, σ=1.0), Uniform{Float64}(a=-1.0, b=1.0))\n)\n\n\n\n\n\n","category":"type"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"pdf","category":"page"},{"location":"#Distributions.pdf","page":"ABCdeZ.jl","title":"Distributions.pdf","text":"pdf(d::Factored, x)\n\nFunction to evaluate the pdf of a Factored distribution object.\n\n\n\n\n\n","category":"function"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"logpdf","category":"page"},{"location":"#Distributions.logpdf","page":"ABCdeZ.jl","title":"Distributions.logpdf","text":"logpdf(d::Factored, x)\n\nFunction to evaluate the logpdf of a Factored distribution object.\n\n\n\n\n\n","category":"function"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"rand","category":"page"},{"location":"#Base.rand","page":"ABCdeZ.jl","title":"Base.rand","text":"rand(rng::AbstractRNG, factoreddist::Factored)\n\nFunction to sample one element from a Factored object.\n\n\n\n\n\n","category":"function"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"length","category":"page"},{"location":"#Base.length","page":"ABCdeZ.jl","title":"Base.length","text":"length(p::Factored)\n\nReturns the number of distributions contained in p.\n\n\n\n\n\n","category":"function"},{"location":"#Various-notes","page":"ABCdeZ.jl","title":"Various notes","text":"","category":"section"},{"location":"#ABC-Approximations","page":"ABCdeZ.jl","title":"ABC - Approximations","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"eps, summary stats","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"ABC (kernel instead of likelihood) and summary stats both introduce approximation errors, maybe read the  two lines in Didelot again...","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"eps trade off","category":"page"},{"location":"#More-on-model-evidences","page":"ABCdeZ.jl","title":"More on model evidences","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"when using summary stats => sufficient for model selection (link stackoverflow post and paper)","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"summary stats need to be sufficient for model selection        (it is not enough if summary stats are sufficient for         each model's parameters!), link to paper and         stackoverflow topic","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"\\@ref(normalisation_factor) off by normalisation factor (with default kernel), does not  matter when comparing models for the same eps;  what if same eps not available / impractical?","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"same ϵ target necessary (if not possible upper bound conservative        estimate may be possible, or, use ϵs and logZs lists for finding        last common ϵ to compare with)","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"explain here what to do when same eps difficult (link goes here...)","category":"page"},{"location":"#Features-for-the-distance-methods","page":"ABCdeZ.jl","title":"Features for the distance methods","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"In the distance function in the minimal example (dist!(θ, ve) = abs(model(θ)-data), nothing)  ve are \"external variables\" (varexternal in abcdesmc!) that can  be used in the distance computation and mutated in-place, even in the parallel mode  (each thread will obtain its own copy for thread-safe parallel ABC runs). ve is passed as 4th positional argument to abcdesmc! (nothing in the  minimal example).\nIn the distance function in the minimal example (dist!(θ, ve) = abs(model(θ)-data), nothing)  the second return argument (nothing) can be used to store arbitrary data  (blobs) to each particle; these blobs will be associated with the final  posterior samples/particles in the end. For example blobs could record the  actual simulation output:\nfunction dist2!(θ, ve, constants, data)\n    # constants can be used to pass thread-safe constants that are NOT mutated;\n    # ve for in-place, mutatable variables\n\n    # distance method\n    simdata = model(θ)\n    blob = simdata\n    d = abs(simdata-data)\n    d, blob\nend\ndist2!(θ, ve) = dist2!(θ, ve, nothing, data)\n\nr = abcdesmc!(prior, dist2!, ϵ, nothing, \n                    nparticles=1000, parallel=true)\n\nposterior = [t[1] for t in r.P[r.Wns .> 0.0]]\nevidence = exp(r.logZ)\nblobs = r.blobs[r.Wns .> 0.0]","category":"page"},{"location":"#References","page":"ABCdeZ.jl","title":"References","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"ABCdeZ.jl was developed @TSB by  Maurice Langhinrichs and Nils Becker.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"Some part of the code was copied, adapted and/or inspired by KissABC.jl [1]. For example,    the Factored syntax was adopted, abcdemc! is based on ABCDE, abcdesmc! is loosely based on    smc. We thank the developers of the package.\nA very good theory background for the general approach of model evidences from single ABC runs    is given by Didelot et al. (2011) [2]. More details on algorithms (in the likelihood-context) is found in Llorente et al. (2020) [3].\nThe differential evolution moves are introduced in Ter Braak (2006) [4].\nAs done also in KissABC.jl [1], the implementations of the abcdemc! method are a simplified    version of the method in Turner et al. (2012) [5]. The algorithmic idea in abcdesmc! is mostly based on Amaya et al. (2021) [6],   next to KissABC.jl, particular the handling of weights and the adaptive differential evolution move tuning (Amaya et al. (2021) is    in the likelihood context, which we adapted to ABC).\nCloser read on sufficient summary statistics for model comparison in ABC is found in Marin et al. (2014) [7] and condensed in this    stackexchange post [8].\nStratified resampling (for abcdesmc!) is inspired by Douc et al. (2005) [9].","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[1]: KissABC (https://github.com/francescoalemanno/KissABC.jl)","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[2]: Didelot et al. \"Likelihood-free estimation of model evidence.\" Bayesian Anal. 6 (1) 49 - 76, 2011.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[3]: Llorente et al. \"Marginal likelihood computation for model selection and hypothesis testing: an extensive review\" arXiv:2005.08334 [stat.CO], 2020.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[4]: Ter Braak. \"A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces\" Statistics and Computing volume 16, pages 239–249, 2006.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[5]: Turner et al. \"Approximate Bayesian computation with differential evolution\"  Journal of Mathematical Psychology, 2012.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[6]: Amaya et al. \"Adaptive sequential Monte Carlo for posterior inference and model selection among complex geological priors\" arXiv:2105.02012 [physics.geo-ph], 2021.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[7]: Marin et al. \"Relevant statistics for Bayesian model choice\" J. R. Statist. Soc. B, 2014.","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[8]: https://stats.stackexchange.com/questions/26980/abc-model-selection","category":"page"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"[9]: Douc et al. \"Comparison of Resampling Schemes for Particle Filtering\" arXiv:cs/0507025 [cs.CE], 2005.","category":"page"},{"location":"#Index","page":"ABCdeZ.jl","title":"Index","text":"","category":"section"},{"location":"","page":"ABCdeZ.jl","title":"ABCdeZ.jl","text":"","category":"page"}]
}