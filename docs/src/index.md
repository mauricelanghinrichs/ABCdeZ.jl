
```@meta
CurrentModule = ABCdeZ
```

# ABCdeZ.jl

```@contents
```

## Introduction

- two example files for both alg. ? or one extended minimal example (also changing
    script in examples...)
- describe briefly: demc greedy version (biased before 
  completion but fast), desmc (accurate, a bit slower, but also 
  evidence estimate (up to (final) kernel norm))
- link to all the caveats of ABC-based evidence values

## Minimal example

this code here should match the script in examples folder on github; 
and then maybe on github just put the png image

current idea: use current example but add a second model 
with a slightly larger Normal prior (and also change model / likelihood / simulation itself? maybe 
variation in the Normal from 1.0 to something?); try to calculate 
evidence also for the second model analytical; in the end make 
a nice png with three plots (posterior model 1 and 2 (from the different priors) and a third 
plot showing the model probabilities (or evidences?) between smc output and analytical also); 
also as a final line compute posterior by abcdemc! version (just for the posterior, 
not likelihood)

- model comparison can be done with the model evidences obtained by 
  abcdesmc!; as the evidence values are (typically) off by the 
  normalisation constant, different models (each inferred from a separate 
  ABC run) have to be compared with logZs computed for the SAME ϵ 
  (then the unknown normalisation constants cancel and don't matter)
  (of course, next to the same data and distance method);

  for example posterior model probabilities are computed like this 
  (for a uniform model prior)


```julia
    m1_evidence = evidence # evidence of model 1 from above
    m2_evidence = 0.008 # from some other ABC run

    m1_prior = 0.5 # model priors (uniform here)
    m2_prior = 0.5

    m1_prob = m1evid*m1_prior / (m1evid*m1_prior + m2evid*m2_prior) # posterior prob. model 1
    m2_prob = m2evid*m2_prior / (m1evid*m1_prior + m2evid*m2_prior) # posterior prob. model 2
```

!!! note "Prior"

    Multidimensional prior distributions (continuous, discrete or mixed) can 
    be specified via the [`Factored()`](#Distributions-and-Priors) syntax (from 
    independent 1d marginals), e.g. `prior2d = Factored(Normal(0, sqrt(10)), 
    DiscreteUniform(1, 10))`.

## Inference by abcdesmc!

```@docs
abcdesmc!
```

!!! warning "Posterior sample weights"

    Posterior samples obtained by `abcdesmc!` have to be associated with their weights (`r.Wns`). 
    With an indicator ABC kernel (default) there are just two weights (i.e. alive and dead 
    particles) and the correct posterior samples are hence given by 
    `posterior = [t[1] for t in r.P[r.Wns .> 0.0]]` (for the first parameter here).

!!! warning "Model evidence (off by a factor)"

    The model evidence estimates from the `abcdesmc!` method obtained by the default ABC 
    indicator kernel are off by a normalisation factor coming from an unnormalised 
    kernel (in the final iteration). To do model selection / comparison 
    this means that evidence estimates for the set of models have to be done for the same 
    data (or summary statistics), distance function, ABC kernel *and* the same target 
    ϵ (which is `ϵ_target` if run not stopped by `nsims_max`). Then the (unknown) 
    normalisation factor is the same for all models and does not matter (cancels) for 
    Bayes factor or posterior model probabilities. See [here](#More-on-model-evidences)
    for workarounds if ϵ is not the same.

!!! tip "Uncertainty of model evidence"

    As of now the `abcdesmc!` method does not provide an uncertainty for the model evidence 
    estimate from a single run. It may be however very useful to check for this when doing 
    model comparison (as the resulting Bayes factors or posterior model probabilities are 
    uncertain as well). So, if the runtime permits, run the `abcdesmc!` method multiple times 
    and collect the resulting set of evidence values for `mean`/`median` and `std` estimates.

## Inference by abcdemc!

```@docs
abcdemc!
```

!!! warning "Greediness of abcdemc!"

    The `abcdemc!` method implements a "greedy"/biased Metropolis-Hasting step in the 
    Markov chain. This allows a fast convergence, particularly well-suited for unimodal 
    problems. However, to obtain valid posterior estimates the algorithm needs to 
    converge (all particles below `ϵ_target` and `r.reached_ϵ==true`). Otherwise the samples will be 
    biased (closer to the MAP (maximum a posteriori probability) parameter values 
    with reduced variation).


!!! note

    In contrast to the `abcdesmc!` method the resulting posterior samples of the `abcdemc!` 
    method are *not* associated with weights and can be used directly, i.e. 
    `posterior = [t[1] for t in r.P]` (for the first parameter here).

## Distributions and Priors

```@docs
Factored
```

```@docs
pdf
```

```@docs
logpdf
```

```@docs
rand
```

```@docs
length
```

## Various notes

#### ABC - Approximations

eps, summary stats

ABC (kernel instead of likelihood) and summary stats both introduce approximation errors, maybe read the 
two lines in Didelot again...

eps trade off

#### More on model evidences

when using summary stats => *sufficient* for model selection
(link stackoverflow post and paper)

summary stats need to be sufficient for model selection 
      (it is not enough if summary stats are sufficient for 
       each model's parameters!), link to paper and 
       stackoverflow topic


\@ref(normalisation_factor)
off by normalisation factor (with default kernel), does not 
matter when comparing models for the same eps; 
what if same eps not available / impractical?

same ϵ target necessary (if not possible upper bound conservative 
      estimate may be possible, or, use ϵs and logZs lists for finding 
      last common ϵ to compare with)

explain here what to do when same eps difficult (link goes here...)

#### Features for the distance methods

  - In the distance function in the [minimal example](#Minimal-example)
    (`dist!(θ, ve) = abs(model(θ)-data), nothing`) 
    `ve` are "external variables" (`varexternal` in `abcdesmc!`) that can 
    be used in the distance computation and mutated in-place, even in the parallel mode 
    (each thread base will obtain its own copy for thread-safe parallel ABC runs).
    `ve` is passed as 4th positional argument to `abcdesmc!` (`nothing` in the 
    minimal example).

  - In the distance function in the [minimal example](#Minimal-example)
    (`dist!(θ, ve) = abs(model(θ)-data), nothing`) 
    the second return argument (`nothing`) can be used to store arbitrary data 
    (`blobs`) to each particle; these `blobs` will be associated with the final 
    posterior samples/particles in the end. For example `blobs` could record the 
    actual simulation output:

    ```julia
    function dist2!(θ, ve, constants, data)
        # constants can be used to pass thread-safe constants that are NOT mutated;
        # ve for in-place, mutatable variables

        # distance method
        simdata = model(θ)
        blob = simdata
        d = abs(simdata-data)
        d, blob
    end
    dist2!(θ, ve) = dist2!(θ, ve, nothing, data)

    r = abcdesmc!(prior, dist2!, ϵ, nothing, 
                        nparticles=1000, parallel=true)

    posterior = [t[1] for t in r.P[r.Wns .> 0.0]]
    evidence = exp(r.logZ)
    blobs = r.blobs[r.Wns .> 0.0]
    ```

## References

- Some part of the code was copied, adapted and/or inspired by KissABC.jl [^1]. For example, 
    the `Factored` syntax was adopted, `abcdemc!` is based on `ABCDE`, `abcdesmc!` is inspired 
    by `smc`. We thank the developers of the package.
- A very good theory background for the general approach of model evidences from single ABC runs 
    is given by Didelot et al. (2011) [^2]. More details on algorithms (in the likelihood-context) is found in Llorente et al. (2020) [^3].
- The differential evolution moves are introduced in Ter Braak (2006) [^4].
- As done also in KissABC.jl [^1], the implementations of the `abcdemc!` method are a simplified 
    version of the method in Turner et al. (2012) [^5]. The algorithmic idea in `abcdesmc!` is mostly based on Amaya et al. (2021) [^6],
    next to KissABC.jl, particular the handling of weights and the adaptive differential evolution move tuning (Amaya et al. (2021) is 
    in the likelihood context, which we adapted to ABC).
- Closer read on sufficient summary statistics for model comparison in ABC is found in Marin et al. (2014) [^7] and condensed in this 
    stackexchange post [^8].
- Stratified resampling (for `abcdesmc!`) is inspired by Douc et al. (2005) [^9].

[^1]: 
    
    [KissABC (https://github.com/francescoalemanno/KissABC.jl)](https://github.com/francescoalemanno/KissABC.jl)

[^2]: 
    
    [Didelot et al. "Likelihood-free estimation of model evidence." Bayesian Anal. 6 (1) 49 - 76, 2011.](https://doi.org/10.1214/11-BA602)

[^3]: 
    
    [Llorente et al. "Marginal likelihood computation for model selection and hypothesis testing: an extensive review" arXiv:2005.08334 [stat.CO], 2020.](https://arxiv.org/abs/2005.08334)

[^4]:
    
    [Ter Braak. "A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces" Statistics and Computing volume 16, pages 239–249, 2006.](https://link.springer.com/article/10.1007/s11222-006-8769-1)

[^5]:

    [Turner et al. "Approximate Bayesian computation with differential evolution"  Journal of Mathematical Psychology, 2012.](https://compmem.org/assets/pubs/Turner.Sederberg.2012.pdf)

[^6]:
    
    [Amaya et al. "Adaptive sequential Monte Carlo for posterior inference and model selection among complex geological priors" arXiv:2105.02012 [physics.geo-ph], 2021.](https://arxiv.org/abs/2105.02012)

[^7]:
    
    [Marin et al. "Relevant statistics for Bayesian model choice" J. R. Statist. Soc. B, 2014.](https://www.jstor.org/stable/24774605)

[^8]:

    [https://stats.stackexchange.com/questions/26980/abc-model-selection](https://stats.stackexchange.com/questions/26980/abc-model-selection)

[^9]:
    
    [Douc et al. "Comparison of Resampling Schemes for Particle Filtering" arXiv:cs/0507025 [cs.CE], 2005.](https://arxiv.org/abs/cs/0507025)

## Index

```@index
```