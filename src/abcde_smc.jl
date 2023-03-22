
# implements an ABC DE SMC algorithm with evidence (Z) estimate

# NOTES:
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
#   amount of epsilon iterations etc.?!)

function abcdesmc_step!(logγ, blobs, θs, prior, dist!, varexternal, ϵ_kernel, nparticles, rng, ex)

    # NOTE/TODO: random number not needed here?? maybe remove
    # NOTE/TODO: also add MH step here?

    @floop ex for i = 1:nparticles
        @init ve = deepcopy(varexternal)
        trng=rng
        ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)
        
        # parameters
        θ = push_p(prior, θs[i].x)

        # compute log-prior value
        logπ = logpdf(prior, θ)

        # compute distance value (between model / data)
        d, blob = dist!(θ, ve)
        blobs[i] = blob

        # compute log-likelihood (ABC kernel) value
        logk = logpdf(ϵ_kernel, d)

        # compute prior*likelihood (via ABC kernel) in log space
        logγ[i] = logπ + logk
    end
end

function abcdesmc_resample!(ess_inds, θs, logγ, Wns, nparticles, rng)
    # resample with normalised weights, indices updated to ess_inds
    wsample!(rng, 1:nparticles, Wns, ess_inds, replace=true)
            
    for i in 1:nparticles
        θs[i] = θs[ess_inds[i]]
        logγ[i] = logγ[ess_inds[i]]
    end

    Wns .= 1.0/nparticles
end

function abcdesmc_update_ws!(ws, logπ, logk, nlogπ, nlogk)
    
end

function abcdesmc!(prior, dist, ϵ_target, varexternal;
                nparticles::Int=100, δess=0.4,
                α=0.95,
                verbose=true, rng=Random.GLOBAL_RNG, ex=ThreadedEx(),
                testmode=false)
    
    ### initialisation
    @info("Running abcdemc! with executor ", typeof(ex))
    0.0 ≤ α < 1.0 || error("α must be in 0 <= α < 1")
    0.0 ≤ ϵ_target || error("ϵ_target must be non-negative") # TODO/NOTE: like this or adaptive termination?
    5 ≤ nparticles || error("nparticles must be at least 5") # TODO: maybe chance, see KissABC SMC

    ϵ_current = Inf # current ϵ (ABC target distance)
    ϵ_kernel = Normal(0.0, ϵ_current)

    ess_min = nparticles * δess # minimal effective sample size of the population
    ess_inds = zeros(Int, nparticles) # placeholder for resampling indices
    logZ = 0.0 # Z = 1.0 current evidence value

    # NOTE/TODO: preallocate blobs and logγ objects?
    ve = deepcopy(varexternal)
    d1, blob1 = dist!(push_p(prior, θs[1].x), ve)

    if testmode
        # maybe define some more vector to keep track of iterations for testing
    end

    # draw prior parameters for each particle
    θs = [op(float, Particle(rand(rng, prior))) for i in 1:nparticles]

    # at first iteration γ=prior*likelihood (likelihood via ABC kernel 
    # at ϵ=Inf) γ (unnormalised posterior) is identical to likelihood
    logγ = [logpdf(prior, push_p(prior, θs[i].x)) for i in 1:nparticles]
        
        

    # normalised weights and placeholder for weights product
    Wns = ones(nparticles)./nparticles
    wprod = ones(nparticles)
    wnorm = 0.0

    # 
    iters = 0
    while true
        iters += 1

        # set a new ϵ target, including ABC kernel
        ϵ_new = quantile(Δs, α)
        ϵ_kernel = Normal(0.0, ϵ_new) # TODO/NOTE: at this point correct??

        # update target weights
        # NOTE/TODO: does this make sense? the prior value (in there) 
        # may be actually the same, so we don't need them???
        ws .= nlogγ .- logγ

        # update normalised weights and get norm for evidence
        wprod .= (Wns .* ws)
        wnorm = sum(wprod)
        Wns .= (wprod./wnorm)

        # update evidence value
        # NOTE/TODO: maybe do this in log-mode?
        logZ += log(wnorm)

        


        # resample if effective sample size too low
        # TODO/NOTE: new or old particles here???
        ess(Wns)<ess_min && (abcdesmc_resample!(ess_inds, θs, logγ, Wns, nparticles, rng))

        # MCMC steps at current target density (as given by ϵ_current)
        


        ϵ_current = ϵ_new
    end

    ### report results
    # NOTE/TODO: need to return 
    # 1) weights (both types?), posterior samples may only make sense with weights!
    # 2) evidence estimate?
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles]
    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    length(P)==1 && (P=first(P))
    (P = P, C = Δs, ϵ = ϵ, blobs = blobs)
end

### helper methods
# function logsumexp(x::Vector{T}) where T
#     res = zero(T)
#     for i in eachindex(x)
#         res += exp(x[i])
#     end
#     log(res)
# end

# log_ess(logw) = -logsumexp(2 .* logw)
ess(Wns) = 1/sum(Wns.^2)

