import Distributions.pdf, Distributions.logpdf, Random.rand, Base.length

struct MixedSupport <: ValueSupport end

"""
    Factored{N} <: Distribution{Multivariate, MixedSupport}

A `Distribution` type that can be used to combine multiple `UnivariateDistribution`'s (independently).

# Examples
```julia-repl
julia> prior = Factored(Normal(0, 1), Uniform(-1, 1))
Factored{2}(
p: (Normal{Float64}(μ=0.0, σ=1.0), Uniform{Float64}(a=-1.0, b=1.0))
)
```
"""
struct Factored{N} <: Distribution{Multivariate,MixedSupport}
    p::NTuple{N,UnivariateDistribution}
    Factored(args::UnivariateDistribution...) = new{length(args)}(args)
end
"""
    pdf(d::Factored, x)

Function to evaluate the pdf of a `Factored` distribution object.
"""
function pdf(d::Factored{N}, x) where {N}
    s = pdf(d.p[1], x[1])
    for i = 2:N
        s *= pdf(d.p[i], x[i])
    end
    s
end

"""
    logpdf(d::Factored, x)

Function to evaluate the logpdf of a `Factored` distribution object.
"""
function logpdf(d::Factored{N}, x) where {N}
    s = logpdf(d.p[1], x[1])
    for i = 2:N
        s += logpdf(d.p[i], x[i])
    end
    s
end

"""
    rand(rng::AbstractRNG, factoreddist::Factored)

Function to sample one element from a `Factored` object.
"""
rand(rng::AbstractRNG, factoreddist::Factored{N}) where {N} =
    ntuple(i -> rand(rng, factoreddist.p[i]), Val(N))

"""
    length(p::Factored)

Returns the number of distributions contained in `p`.
"""
length(p::Factored{N}) where {N} = N
