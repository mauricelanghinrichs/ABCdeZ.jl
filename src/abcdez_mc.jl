
# implements the ABC DE MCMC algorithm
# adapted from KissABC

function abcdemc_swarm!(prior, dist!, varexternal, θs, logπ, Δs, nθs, nlogπ, nΔs,
                    ϵ_pop, ϵ_target, γ0, γσ, nparticles, nsims, rng, ex, nblobs)
    @floop ex for i in 1:nparticles
        @init ve = deepcopy(varexternal)

        trng=rng
        ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)

        ### DE (diffential evolution) move
        # NOTE: as long as "if Δs[i] > ϵ" clause applies the proposal kernel 
        # is not symmetric as the if-clause selects a better particle to replace 
        # the original one in the DE move; non-symmetric proposal is fine, however 
        # it would need to enter the MH step below which it does not; the proposal 
        # becomes only symmetric if the if-clause is not entered anymore (a and b 
        # particles are symmetric), e.g. this happens when all particles are 
        # at some point below ϵ_target
        s = i
        ϵ = ifelse(Δs[i] <= ϵ_target, ϵ_target, ϵ_pop)
        if Δs[i] > ϵ
            # NOTE: Δs .<= Δs[i] is this data race safe? note that nΔs and Δs
            # are not referenced through the identity.() broadcast
            s=rand(trng, (1:nparticles)[Δs .<= Δs[i]])
        end
        a = s
        while a == s
            a = rand(trng, 1:nparticles)
        end
        b = a
        while b == a || b == s
            b = rand(trng, 1:nparticles)
        end
        # θp is a new Particle with new tuple values (.x) [see comment above]
        θp = op(+, θs[s], op(*, op(-, θs[a], θs[b]), γ0 * (1.0 + randn(trng)*γσ) ))

        ### MH (Metropolis–Hastings) acceptance step
        # NOTE: strictly ratios of prior, ABC kernel (likelihood if available) 
        # and proposal kernel needs to be considered in min{1, ratios} (non-log space);
        # only symmetric proposal kernel with q(θ'|θ)=q(θ|θ') would cancel (see DE move above);
        # ABC kernel can be simplified as below (out of min, into if clause) if simple indicator
        lπ = logpdf(prior, push_p(prior, θp.x))
        w_prior = lπ - logπ[i] # prior ratio (in log space)
        log(rand(trng)) > min(0, w_prior) && continue
        nsims[i] += 1
        dp, blob = dist!(push_p(prior, θp.x), ve)

        # NOTE: this "implements" the ABC kernel (indicator here) that is theoretically 
        # part of the min MH call above; at final equilibration (all particles below 
        # ϵ_target) the previous particle has already indicator kernel = 1 and we only 
        # (continue to (after MH above)) accept the new particle if it is below ϵ_target; 
        # so technically the above minimum could be extended by cases log(1/1) or log(0/1), 
        # so the minimum with prior ratio and this if-clause correctly implement the MH step 
        # (at equilibration, i.e., all particles below ϵ_target)
        if dp <= max(ϵ, Δs[i])
            nΔs[i] = dp
            nθs[i] = θp
            nlogπ[i] = lπ
            nblobs[i] = blob
        end
    end
end

"""
    abcdemc!(prior, dist!, ϵ_target, varexternal; <keyword arguments>)

Run ABC with diffential evolution (de) moves in a Markov chain Monte Carlo setup (mc) 
providing posterior samples.

Algorithm needs to converge for an unbiased posterior estimate.

# Arguments
- `prior`: `Distribution` or `Factored` object specifying the parameter prior.
- `dist!`: distance function computing the distance (`≥ 0.0`) between model and data, 
    for given `(θ, ve)` input (`θ` parameters, `ve` external variables, see `varexternal`).
- `ϵ_target`: final target distance (or more general, target width of the ABC kernel); algorithm 
    equilibrates to final target distribution (approximate posterior) if `ϵ_target` is reached.
- `varexternal`: external variables that are passed as second positional argument to `dist!` 
    and can be used to support the distance computation with fast in-place operations in 
    a thread-safe manner; objects in `varexternal` can be in-place mutated, even in `parallel` mode, 
    as each thread will receive its own copy of `varexternal` (if not needed input `nothing`).
- `nparticles::Int=50`: number of total particles to use for inference in each generation.
- `generations::Int=20`: number of generations (total iterations) to run the algorithm.
- `verbose::Bool=true`: if set to `true`, enables verbosity (printout to REPL).
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
julia> r = abcdemc!(prior, dist!, ϵ, nothing, nparticles=1000, generations=300, parallel=true);
julia> posterior = [t[1] for t in r.P];
```
"""
function abcdemc!(prior, dist!, ϵ_target, varexternal; 
                nparticles::Int=50, generations::Int=20,
                verbose=true, rng=Random.GLOBAL_RNG, parallel::Bool=false)
    
    ### initialisation
    α = 0.0 # 0.0 ≤ α < 1.0 || error("α must be in 0 <= α < 1")
    0.0 ≤ ϵ_target || error("ϵ_target must be non-negative")
    5 ≤ nparticles || error("nparticles must be at least 5")
    1 ≤ generations || error("generations must be at least 1")

    parallel ? ex=ThreadedEx() : ex=SequentialEx()
    verbose && (@info("Running abcdemc! with executor ($(Threads.nthreads()) threads available) ", typeof(ex)))
    verbose && (@info "Running abcdemc! with" ϵ_target nparticles generations α rng parallel)

    # draw prior parameters for each particle, and calculate logprior values
    θs = [op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
    logπ = [logpdf(prior, push_p(prior, θs[i].x)) for i = 1:nparticles]

    ve = deepcopy(varexternal)
    d1, blob1 = dist!(push_p(prior, θs[1].x), ve)
    Δs = fill(d1, nparticles)
    blobs = fill(blob1, nparticles)

    abcde_init!(prior, dist!, varexternal, θs, logπ, Δs, nparticles, rng, ex, blobs)
    ###

    ### actual ABC run
    nsims = zeros(Int, nparticles)
    γ0 = 2.38 / sqrt(2 * length(prior))
    γσ = 1e-5
    iters = 0
    complete = 1 - sum(Δs .> ϵ_target) / nparticles
    while iters<generations
        iters += 1
        # identity.() behaves like deepcopy(), i.e. == is true,
        # === is false (in general, except for immutables which will be true always),
        # so there are n=new object that can be mutated without data races

        nθs = identity.(θs) # vector of particles, where θs[i].x are parameters (as tuple)
        nΔs = identity.(Δs) # vector of floats with distance values (model/data)
        nlogπ = identity.(logπ) # vector of floats with log prior values of above particles
        nblobs = identity.(blobs) # blobs (some additional data) for each particle

        # returns minimal and maximal distance/cost
        ϵ_l, ϵ_h = extrema(Δs)
        ϵ_pop = max(ϵ_target, ϵ_l + α * (ϵ_h - ϵ_l))

        abcdemc_swarm!(prior, dist!, varexternal, θs, logπ, Δs, nθs, nlogπ, nΔs,
                            ϵ_pop, ϵ_target, γ0, γσ, nparticles, nsims, rng, ex, nblobs)

        θs = nθs
        Δs = nΔs
        logπ = nlogπ
        blobs = nblobs
        ncomplete = 1 - sum(Δs .> ϵ_target) / nparticles
        if verbose && (ncomplete != complete || complete >= (nparticles - 1) / nparticles)
            @info "Finished run:" completion = ncomplete nsim = sum(nsims) range_ϵ = extrema(Δs)
        end
        complete = ncomplete
    end

    conv = maximum(Δs) <= ϵ_target
    verbose && (@info "End:" completion = complete converged = conv nsim = sum(nsims) range_ϵ = extrema(Δs))
    
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles]
    # l = length(prior)
    # P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    # length(P)==1 && (P=first(P))
    # (P = P, C = Δs, reached_ϵ = conv, blobs = blobs)
    (P = θs, C = Δs, reached_ϵ = conv, blobs = blobs)
end

###### some (outdated) notes
# NOTE: on "γ0 = 2.38 / sqrt(2 * length(prior)), γσ = 1e-5";
# the values here for the DE proposal move come from 
# Cajo J. F. Ter Braak and also (with the actual implentation) from 
# Benjamin E. Nelson et al. (using γ = γ0 * (1+Z)) with a bit of 
# normal noise as Z~Normal(0,γσ)

### NOTE: this script is an addition to KissABC to implement
### multithreading support with in-place operations for the ABCDE method;
### for this we use FLoops and its @floop and @init macros

# TODO/NOTE: how to really check if multithreading runs correctly?
# maybe fix all random seeds

# IMPORTANT NOTE: in this code we use multithreading in the form of "filling pre-allocated
# output" (https://juliafolds.github.io/data-parallelism/tutorials/mutations/#filling_outputs)
# this can be unsafe and cause data races! (dict, sparsearrays, Bit vectors, views)
# but Arrays should be fine (and θs, logπ, Δs, nθs, nlogπ, nΔs are arrays)
# println(typeof(θs)) => Vector{KissABC.Particle{Tuple{Float64, Float64, Float64}}}
# println(typeof(logπ)) => Vector{Float64}
# println(typeof(Δs)) => Vector{Float64}

# NOTE: the op tuple operations (also on a Particle) seem to
# be zero-allocating (both immutable types!, no heap memory needed) and can be
# broadcasted (over multiple elements in a tuple or Particle.x); values inside
# the tuples are all plain data (int, float), so that this should be no problem
# for multithreading (although zero-allocation, nothing is changed in-place,
# just plain data calculation in stack memory); in the end written to nθs
# (which in each generation is created as deepcopy and creates allocations, by identity.());
# see also seems tests in polylox_env_ABC_HSC_test.jl
# so all things at θs[i], logπ[i], Δs[i] are immutable, such as Vector{SomeType}()
# with SomeType immutable as in my question (https://discourse.julialang.org/t/how-to-implement-multi-threading-with-external-in-place-mutable-variables/62610/3)

# NOTE: @reduce needed? if yes
# use solution like this https://discourse.julialang.org/t/using-floops-jl-to-update-array-counters/58805
# or this (histogram)? https://juliafolds.github.io/data-parallelism/tutorials/quick-introduction/

# NOTE: on push_p(prior, θs[i].x) (or push_p(prior, θp[i].x)) functionality;
# this makes sure that the tuple parameters θs[i].x of a particle θs[i] will 
# be converted to the same value domain as the prior distributions; only makes 
# a practical difference when DiscreteDistribution's are involved; e.g., 
# push_p will then convert (1.12131, 3.0) [= θs[i].x] to (1.12131, 3)
# NOTE (important): particles internally will have continuous values, even 
# if they are discrete (e.g. 3.123 will be kept), but the push_p/op float 
# calls (to 3 or 3.0) are always applied to the "outside", i.e. when 
# prior pdf or distance functions are called; also in the final reporting 
# of the samples!
######