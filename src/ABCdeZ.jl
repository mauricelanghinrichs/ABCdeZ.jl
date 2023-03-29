module ABCdeZ

    using Random
    using Distributions
    using StatsBase
    using FLoops

    include("abcde_priors.jl")
    export Factored

    include("abcde_types.jl")
    include("abcde_init.jl")

    include("abcde_mc.jl")
    export abcdemc!

    include("abcde_smc.jl")
    export abcdesmc!
    
end
