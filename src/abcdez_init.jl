
function abcde_init!(prior, dist!, varexternal, θs, logπ, Δs, nparticles, rng, ex, blobs)
    # calculate cost/dist for each particle
    # (re-draw parameters if not finite)

    @floop ex for i in 1:nparticles
        @init ve = deepcopy(varexternal)

        if isfinite(logπ[i])
            d, blob = dist!(push_p(prior, θs[i].x), ve)
            Δs[i] = d
            blobs[i] = blob
        end
        while (!isfinite(Δs[i])) || (!isfinite(logπ[i]))
            θs[i] = op(float, Particle(rand(rng, prior)))
            logπ[i] = logpdf(prior, push_p(prior, θs[i].x))
            d, blob = dist!(push_p(prior, θs[i].x), ve)
            Δs[i] = d
            blobs[i] = blob
        end
    end
end

###### some (outdated) notes
# @init allows re-using mutable temporary objects within each base case/thread
# TODO/NOTE: in ThreadedEx mode one might observe a high % of garbage collection
# it was hard to check if this @init really works on my varexternal object...
# (compared with ve = deepcopy(varexternal) alone allocations were similar,
# but also in sequential mode, suggesting it is just not driving the allocations)
# NOTE: varexternal is the tuple wrapper of all mutatable external
# variables as cellstate, stats, ... (same in abcde_swarm!)
# NOTE: θs, logπ, Δs are read out but never written to, (for this
# nθs, nlogπ, nΔs are used), so this should be data race free