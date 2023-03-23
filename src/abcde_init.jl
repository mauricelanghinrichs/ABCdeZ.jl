
# @init allows re-using mutable temporary objects within each base case/thread
# TODO/NOTE: in ThreadedEx mode one might observe a high % of garbage collection
# it was hard to check if this @init really works on my varexternal object...
# (compared with ve = deepcopy(varexternal) alone allocations were similar,
# but also in sequential mode, suggesting it is just not driving the allocations)
# NOTE: varexternal is the tuple wrapper of all mutatable external
# variables as cellstate, stats, ... (same in abcde_swarm!)
# NOTE: θs, logπ, Δs are read out but never written to, (for this
# nθs, nlogπ, nΔs are used), so this should be data race free

# NOTE: I checked that Threads.threadid() also works in an floop
# NOTE: seems to have something with thread-safe random numbers, see
# https://discourse.julialang.org/t/multithreading-and-random-number-generators/49777/8
# NOTE/TODO: maybe this can be improved performance-wise (see FLoops docs
# on random numbers)

function abcde_init!(prior, dist!, varexternal, θs, logπ, Δs, nparticles, rng, ex, blobs)
    # calculate cost/dist for each particle
    # (re-draw parameters if not finite)

    @floop ex for i = 1:nparticles
        @init ve = deepcopy(varexternal)
        
        trng=rng
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
