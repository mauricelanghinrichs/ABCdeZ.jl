
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

function abcdesmc(prior, dist;
                rng::AbstractRNG = Random.GLOBAL_RNG,
                nparticles::Int = 100,
                alpha = 0.95)
    
    # initialisations

    # 

    
end
