
# implements an ABC DE SMC algorithm with evidence (Z) estimate

### helper methods
# effective sample size (~ number of particles)
# (with indicator kernel (all Wns 0.0 or some identical 
# value c, with sum(Wns)=1) get_ess(Wns)≈sum(alive))
get_ess(Wns) = 1.0/sum(Wns.^2)

# NOTE: maybe change resampling from multinomial to residual, stratified or 
# systematic resampling with lower variance; e.g., see KissABC 
# (only for indicator / alive-uniform weights, which I may want to generalise)
# idxalive = (1:nparticles)[alive]
# idx=repeat(idxalive,ceil(Int,nparticles/length(idxalive)))[1:nparticles])
function wsample_stratified!(rng, weights, inds) # ; wnorm=false
    # method for stratified resampling, writing sampling indices 
    # into inds (in-place) with n=length(weights)=length(inds)
    # resamples in total

    # NOTE: assume normalised weights here, i.e. sum(weights)≈1.0,
    # otherwise use wnorm=true to (re-)normalise
    # wnorm && (weights ./= sum(weights))

    # NOTE: a further check may be length(weights)==length(inds),
    # but not needed in our package context here

    # NOTE: this version works for continuous weights; of course 
    # with indicator kernel weights simplify to two possible 
    # values (0 and some c>0), so one could implement a faster 
    # version for indicator kernels only, but it should be fast enough

    # value increment per group/stratum (s)
    sval = 1/length(weights)

    # current sum of weights and resampling index
    wsum = 0.0
    i = 0

    # support range for uniform random variable
    unif0 = 0.0
    unif1 = 0.0

    # for each stratum si, draw random uniform r and take
    # the resampling index i of the weight that is "hit" by r
    for si in eachindex(weights)
        unif1 = unif0 + sval
        r = rand(rng, Uniform(unif0, unif1))
        while r > wsum
            i += 1
            wsum += weights[i]
        end
        unif0 = unif1
        inds[si] = i
    end
    inds
end

### smc methods
function abcdesmc_update_ws!(ws, alive, Δs, ϵ_k, ϵ_k_new, nparticles)
    # NOTE: the update step here is based on the ratio 
    # of unnormalised posterior distributions γ, i.e. prior x likelihood
    # (aka ABC kernel), for the same θ_{t-1} at the current (t-1)
    # and next target (t), i.e. γ_t(θ_{t-1}) / γ_{t-1}(θ_{t-1});
    # the prior ratio cancels and we need to compare the kernels only
    
    # more specifically, ϵ_k is the previous kernel γ_{t-1}(θ_{t-1}),
    # while logpdf(ϵ_k_new, Δs[i]) computes kernel at current (new) ϵ target

    # NOTE: that the kernels are unnormalised (evidence values 
    # computed up to a factor of first and final kernel normalisations)

    # NOTE: for this ws calculation we take the same simulation results 
    # from before and don't compute a second simulation for this particle 
    # to compare it to the new kernel ϵ_k_new; i.e. Δs[i] is the same 
    # here in both pdf evaluations

    for i in 1:nparticles
        if alive[i]
            # with an indicator kernel this will be 0 or 1
            ws[i] = exp(logpdf(ϵ_k_new, Δs[i]) - logpdf(ϵ_k, Δs[i]))
        end
    end
end

function abcdesmc_resample!(ess_inds, θs, logπ, Δs, Wns, alive, nparticles, rng, blobs)
    ### former multinomial sampling
    # resample with normalised weights, indices updated to ess_inds
    # wsample!(rng, 1:nparticles, Wns, ess_inds, replace=true)
    
    ### same-expectation, lower-variance stratified resampling
    # resample with normalised weights, indices updated to ess_inds
    wsample_stratified!(rng, Wns, ess_inds)

    # NOTE: need to create a copy/allocations here by array[ess_inds] 
    # (looping over indices is wrong, as particles may get overwritten)
    θs .= θs[ess_inds]
    logπ .= logπ[ess_inds]
    Δs .= Δs[ess_inds]
    blobs .= blobs[ess_inds]

    # all particles alive again (Wns>0.0) and equal weight
    Wns .= 1.0/nparticles
    alive .= true
end

function abcdesmc_swarm!(prior, dist!, varexternal, 
                        alive, θs, logπ, Δs, nθs, nlogπ, nΔs,
                        ϵ_k_new, γ0, γσ, nparticles, 
                        nsims, naccs, rng, ex, nblobs)
    @floop ex for i in 1:nparticles
        @init ve = deepcopy(varexternal)

        trng=rng
        ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)

        ### zero-weight particles can be neglected
        alive[i] || continue

        ### DE (diffential evolution) move
        # NOTE: DE move done within alive particles only 
        # (such a proposal kernel is still symmetric)
        a = i
        while a == i
            # a = rand(trng, 1:nparticles)
            a = wsample(trng, 1:nparticles, alive)
        end
        b = a
        while b == a || b == i
            # b = rand(trng, 1:nparticles)
            b = wsample(trng, 1:nparticles, alive)
        end
        # θp is a new Particle with new tuple values (.x)
        θp = op(+, θs[i], op(*, op(-, θs[a], θs[b]), γ0 * (1.0 + randn(trng)*γσ) ))
        ###

        ### MH step acceptance step to target current ϵ
        # DE move above is symmetric, hence min criterion 
        # simplifies to prior and ABC kernel ratios
        lπ = logpdf(prior, push_p(prior, θp.x))
        lπ < 0.0 && (isinf(lπ)) && continue

        dp, blob = dist!(push_p(prior, θp.x), ve)
        nsims[i] += 1

        w = (lπ - logπ[i] # prior ratio (in log space)
            + logpdf(ϵ_k_new, dp) - logpdf(ϵ_k_new, Δs[i])) # kernel ratio (in log space)
        # for indicator, logpdf(ϵ_kernel, Δs[i]) should be 0 (as we only look at alive)

        # if condition here the same as "log(rand(trng)) < min(0, w)"
        if 0.0 ≤ w || log(rand(trng)) < w
            nΔs[i] = dp
            nθs[i] = θp
            nlogπ[i] = lπ
            nblobs[i] = blob
            naccs[i] += 1
        end
    end
end

### main smc
"""
    abcdesmc!(prior, dist!, ϵ_target, varexternal; <keyword arguments>)

Run ABC with diffential evolution (de) moves in a Sequential Monte Carlo setup (smc) 
providing posterior samples and a model evidence estimate.

The particles have to be weighted (via `r.Wns`) for valid posterior samples.

# Arguments
- `prior`: `Distribution` or `Factored` object specifying the parameter prior.
- `dist!`: distance function computing the distance (`≥ 0.0`) between model and data, 
    for given `(θ, ve)` input (`θ` parameters, `ve` external variables, see `varexternal`).
- `ϵ_target`: final target distance (or more general, target width of the ABC kernel); algorithm 
    stops if `ϵ_target` or `nsims_max` is reached.
- `varexternal`: external variables that are passed as second positional argument to `dist!` 
    and can be used to support the distance computation with fast in-place operations in 
    a thread-safe manner; objects in `varexternal` can be in-place mutated, even in `parallel` mode, 
    as each thread will receive its own copy of `varexternal` (if not needed input `nothing`).
- `nparticles::Int=100`: number of total particles to use for inference.
- `α=0.95`: used for adaptive choice of ϵ specifying the sequential target distributions; technically, 
    ϵ will be the `α`-quantile of current particle distances.
- `δess=0.5`: if the fractional effective sample size drops below `δess`, a stratified resampling step is performed.
- `nsims_max::Int=10^7`: maximal number of `dist!` evaluations (not counting initial samples from prior); 
    algorithm stops if `ϵ_target` or `nsims_max` is reached.
- `Kmcmc::Int=3`: number of MCMC (Markov chain Monte Carlo) steps at each sequential 
    target distribution specified by current ϵ and ABC kernel type.
- `ABCk=ABCdeZ.Indicator0toϵ`: ABC kernel to be specified by ϵ widths that receives distance values.
- `facc_min=0.25`: if the fraction of accepted MCMC proposals drops below `facc_min`, diffential evolution 
    proposals are reduced by a factor of `facc_tune`.
- `facc_tune=0.95`: factor to reduce the jump distance of the diffential evolution 
    proposals in the MCMC step (used if `facc_min` is reached).
- `verbose::Bool=true`: if set to `true`, enables verbosity (printout to REPL).
- `verboseout::Bool=true`: if set to `true`, algorithm returns a more detailed inference output.
- `rng=Random.GLOBAL_RNG`: an AbstractRNG object which is used by the inference.
- `parallel::Bool=false`: if set to `true`, threaded parallelism is enabled; `dist!` must be 
    thread-safe in such a case, e.g. by making use of `varexternal` (`ve`).

# Examples
```julia-repl
julia> using ABCdeZ, Distributions;
julia> data = 5;
julia> prior = Normal(0, sqrt(10));
julia> model(θ) = rand(Normal(θ, 1));
julia> dist!(θ, ve) = abs(model(θ)-data), nothing;
julia> ϵ = 0.3;
julia> r = abcdesmc!(prior, dist!, ϵ, nothing, nparticles=1000, parallel=true);
julia> posterior = [t[1] for t in r.P[r.Wns .> 0.0]];
julia> evidence = exp(r.logZ);
```
"""
function abcdesmc!(prior, dist!, ϵ_target, varexternal;
                nparticles::Int=100, α=0.95, 
                δess=0.5, nsims_max::Int=10^7, Kmcmc::Int=3, 
                ABCk=Indicator0toϵ, facc_min=0.25, facc_tune=0.95,
                verbose::Bool=true, verboseout::Bool=true, 
                rng=Random.GLOBAL_RNG, parallel::Bool=false)
    
    ### initialisation
    0.0 ≤ α < 1.0 || error("α must be in 0 <= α < 1")
    0.0 ≤ δess ≤ 1.0 || error("δess must be in 0 <= δess <= 1")
    0.0 ≤ facc_min ≤ 1.0 || error("facc_min must be in 0 <= facc_min <= 1")
    0.0 ≤ facc_tune ≤ 1.0 || error("facc_tune must be in 0 <= facc_tune <= 1")
    0.0 ≤ ϵ_target || error("ϵ_target must be non-negative") # TODO/NOTE: like this or adaptive termination?
    1 ≤ Kmcmc || error("Kmcmc must be at least 1")
    1 ≤ nsims_max || error("nsims_max must be at least 1")

    nparticles_min = ceil(Int, 3 * length(prior) / (min(α, δess)))
    nparticles_min ≤ nparticles || error("nparticles must be at least $(nparticles_min)")

    parallel ? ex=ThreadedEx() : ex=SequentialEx()
    verbose && (@info("Running abcdesmc! with executor ($(Threads.nthreads()) threads available) ", typeof(ex)))
    verbose && (@info "Running abcdesmc! with" ϵ_target nparticles α δess nsims_max Kmcmc ABCk facc_min facc_tune rng parallel verboseout)

    # draw prior parameters for each particle, and calculate logprior values
    θs = [op(float, Particle(rand(rng, prior))) for i in 1:nparticles]
    logπ = [logpdf(prior, push_p(prior, θs[i].x)) for i in 1:nparticles]

    ve = deepcopy(varexternal)
    d1, blob1 = dist!(push_p(prior, θs[1].x), ve)
    Δs = fill(d1, nparticles)
    blobs = fill(blob1, nparticles)

    # this initialse a first round of particles from prior
    # with finite logprior and dist values; updates θs, logπ, Δs, blobs
    abcde_init!(prior, dist!, varexternal, θs, logπ, Δs, nparticles, rng, ex, blobs)

    # specify the initial kernel
    ϵ = Inf # current ϵ (ABC target distance)
    ϵ_k = ABCk(ϵ)

    # sample size measures for resampling option
    ess_min = nparticles * δess # minimal effective sample size of the population
    ess_inds = zeros(Int, nparticles) # placeholder for resampling indices
    
    # Z = 1.0 current evidence value (integral over prior)
    logZ = 0.0 

    # normalised weights and placeholder for weights product
    ws = ones(nparticles)
    Wns = ones(nparticles)./nparticles
    wprod = ones(nparticles)
    wnorm = 0.0
    alive = ones(Bool, nparticles)
    ess = 0.0

    # count total simulations and acceptances (without init phase)
    nsims = zeros(Int, nparticles)
    naccs = zeros(Int, nparticles)
    facc = 1.0

    # parameters for DE move
    γ0 = 2.38 / sqrt(2 * length(prior))
    γσ = 1e-5
    ###

    if verboseout
        ϵs = [ϵ]
        ranges_ϵ = [extrema(Δs)]
        logZs = [logZ]
        esss = [get_ess(Wns)]
        faccs = [facc]
        γ0s = [γ0]
    end

    iters = 0
    while true
        iters += 1

        # set a new ϵ target, including ABC kernel
        # NOTE: maybe also add option to force ϵ down by at least x%? 
        # may cause too low or only-zero weights however...
        ϵ = max(quantile(Δs[alive], α), ϵ_target)
        ϵ_k_new = ABCk(ϵ)

        # update target weights ws
        abcdesmc_update_ws!(ws, alive, Δs, ϵ_k, ϵ_k_new, nparticles)

        # update normalised weights and get norm for evidence
        wprod .= (Wns .* ws)
        wnorm = sum(wprod) # with indicator kernel, this is % of surviving particles
        Wns .= (wprod ./ wnorm)
        alive .= (Wns .> 0.0)

        # update evidence value 
        # (NOTE: log-space may make this estimate biased (Jensen ineq.), but ok...)
        logZ += log(wnorm)

        # reset naccs and tune proposal if it dropped below facc_min in previous step
        naccs .= 0
        facc < facc_min && (γ0 *= facc_tune)

        # resample if effective sample size too low
        ess = get_ess(Wns)
        ess < ess_min && (
            abcdesmc_resample!(ess_inds, θs, logπ, Δs, Wns, alive, nparticles, rng, blobs);
            ess = get_ess(Wns))

        # MCMC steps at current target density
        # NOTE: only iterate for alive particles, as Wns=0.0 
        # will not contribute to evidence and posterior samples anyway
        # NOTE: one could also break the Kmcmc loop if acceptance ratio high enough 
        # (so only do the "extra" work if getting good proposals becomes diffcult, 
        # see KissABC; however (also for Z estimate) it may be better in general 
        # to equilibrate better to current target distribution, so Kmcmc>1 in every 
        # step may be wanted in some cases)
        for __ in 1:Kmcmc
            nθs = identity.(θs) # vector of particles, where θs[i].x are parameters (as tuple)
            nΔs = identity.(Δs) # vector of floats with distance values (model/data)
            nlogπ = identity.(logπ) # vector of floats with log prior values of above particles
            nblobs = identity.(blobs) # blobs (some additional data) for each particle
            
            abcdesmc_swarm!(prior, dist!, varexternal, 
                            alive, θs, logπ, Δs, nθs, nlogπ, nΔs,
                            ϵ_k_new, γ0, γσ, nparticles, 
                            nsims, naccs, rng, ex, nblobs)

            θs = nθs
            Δs = nΔs
            logπ = nlogπ
            blobs = nblobs
        end

        # compute acceptance fraction of the last Kmcmc Markov steps 
        # among live particles
        facc = sum(naccs)/(sum(alive)*Kmcmc)

        # update kernel
        ϵ_k = ϵ_k_new

        if verboseout
            push!(ϵs, ϵ)
            push!(ranges_ϵ, extrema(Δs))
            push!(logZs, logZ)
            push!(esss, ess)
            push!(faccs, facc)
            push!(γ0s, γ0)
        end

        verbose && (@info "Finished run:" iteration = iters nsim = sum(nsims) ϵ = ϵ range_ϵ = extrema(Δs) ess = ess facc = facc logZ = logZ)

        # stopping criterion
        (ϵ ≤ ϵ_target || sum(nsims) ≥ nsims_max) && break
    end

    verbose && (@info "Final run:" iteration = iters nsim = sum(nsims) ϵ = ϵ range_ϵ = extrema(Δs) ess = ess facc = facc logZ = logZ)

    ### report results
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles]
    # l = length(prior)
    # P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    # length(P)==1 && (P=first(P))
    # NOTE: previous: P = P

    if verboseout
        (P = θs, Wns = Wns, C = Δs, ϵ = ϵ, logZ = logZ, blobs = blobs,
            ϵs = ϵs, ranges_ϵ = ranges_ϵ, logZs = logZs, esss = esss, faccs = faccs, γ0s = γ0s)
    else
        (P = θs, Wns = Wns, C = Δs, ϵ = ϵ, logZ = logZ, blobs = blobs)
    end
end

### some (outdated) NOTES:
# - nice would be to keep the DE MCMC move... seemed just 
#   to perform good before
# - unclear yet how then to correctly compute weights
# - KissABC smc can be maybe used as a template, there are currently no weights
#   explicitly because with current indicator kernel particles are just 
#   alive (1) or dead (0); maybe generalise this and instead of live/dead syntax,
#   simply do a check if weight > 0.0 (then it would still be fast for 
#   indicator kernel but also allows continuous kernels)
# - note: not so sure currently about the smc proposal gamma in KissABC SMC,
#   maybe take the one I used now in abcdemc!; 
# - multiple MCMC chains per ϵ step needed? implement?

# - something is currently weird with the ABC kernel... maybe start with 
#   an indicator, would it make sense? (think of very first iteration... 
#   with indicator kernel all particles would be accepted as it should be)
# - does it matter that ABC kernel is not scaled / normalised? should 
#   affect the scaling of the evidence; does it matter when we compare 
#   model evidences of different models??? (that maybe have a different 
#   amount of epsilon iterations etc.?!) => see Goodnotes notes and slides 
#   (only kernel normalisation of last ϵ kernel should matter, others 
#   cancel telescopically)
###