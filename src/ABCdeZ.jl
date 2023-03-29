module ABCdeZ

    using Random
    using Distributions
    using StatsBase
    using FLoops

    include("abcdez_priors.jl")
    export Factored

    include("abcdez_types.jl")
    include("abcdez_init.jl")

    include("abcdez_mc.jl")
    export abcdemc!

    include("abcdez_smc.jl")
    export abcdesmc!

end
