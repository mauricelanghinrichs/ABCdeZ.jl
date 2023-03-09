
# implements the ABC DE MCMC algorithm from KissABC (my adaption)


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

# NOTE: on "γ0 = 2.38 / sqrt(2 * length(prior)), γσ = 1e-5";
# the values here for the DE proposal move come from 
# Cajo J. F. Ter Braak and also (with the actual implentation) from 
# Benjamin E. Nelson et al. (using γ = γ0 * (1+Z)) with a bit of 
# normal noise as Z~Normal(0,γσ)

function abcdemc_init!(prior, dist!, varexternal, θs, logπ, Δs, nparticles, rng, ex, blobs)
    # calculate cost/dist for each particle
    # (re-draw parameters if not finite)

    @floop ex for i = 1:nparticles
        # @init allows re-using mutable temporary objects within each base case/thread
        # TODO/NOTE: in ThreadedEx mode one might observe a high % of garbage collection
        # it was hard to check if this @init really works on my varexternal object...
        # (compared with ve = deepcopy(varexternal) alone allocations were similar,
        # but also in sequential mode, suggesting it is just not driving the allocations)
        # NOTE: varexternal is the tuple wrapper of all mutatable external
        # variables as cellstate, stats, ... (same in abcde_swarm!)
        # NOTE: θs, logπ, Δs are read out but never written to, (for this
        # nθs, nlogπ, nΔs are used), so this should be data race free
        @init ve = deepcopy(varexternal)
        trng=rng

        # NOTE: I checked that Threads.threadid() also works in an floop
        # NOTE: seems to have something with thread-safe random numbers, see
        # https://discourse.julialang.org/t/multithreading-and-random-number-generators/49777/8
        # NOTE/TODO: maybe this can be improved performance-wise (see FLoops docs
        # on random numbers)
        ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)

        if isfinite(logπ[i])
            d, blob = dist!(push_p(prior, θs[i].x), ve)
            Δs[i] = d
            blobs[i] = blob
        end
        while (!isfinite(Δs[i])) || (!isfinite(logπ[i]))
            θs[i] = op(float, Particle(rand(trng, prior)))
            logπ[i] = logpdf(prior, push_p(prior, θs[i].x))
            d, blob = dist!(push_p(prior, θs[i].x), ve)
            Δs[i] = d
            blobs[i] = blob
        end
    end
end

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

function abcdemc!(prior, dist!, ϵ_target, varexternal; nparticles=50, generations=20, α=0.0, 
                verbose=true, rng=Random.GLOBAL_RNG, ex=ThreadedEx())
    
    @info("Running abcde! with executor ", typeof(ex))

    ### initialisation
    0.0 ≤ α < 1.0 || error("α must be in 0 <= α < 1")
    0.0 ≤ ϵ_target || error("ϵ_target must be non-negative")
    5 ≤ nparticles || error("nparticles must be at least 5")

    # draw prior parameters for each particle
    θs =[op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
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

        abcde_swarm!(prior, dist!, varexternal, θs, logπ, Δs, nθs, nlogπ, nΔs,
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
    if verbose
        @info "End:" completion = complete converged = conv nsim = sum(nsims) range_ϵ = extrema(Δs)
    end
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles]
    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    length(P)==1 && (P=first(P))
    (P=P, C=Particles(Δs), reached_ϵ=conv, blobs=blobs)
end

