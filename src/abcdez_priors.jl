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
    for i in 2:N
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
    for i in 2:N
        s += logpdf(d.p[i], x[i])
    end
    s
end

"""
    rand(rng::AbstractRNG, factoreddist::Factored)

Function to draw one sample from a `Factored` distribution, returned as a tuple
with one value per contained (univariate) distribution.

# Examples
```julia-repl
julia> using ABCdeZ, Distributions;
julia> prior = Factored(Normal(0, 1), DiscreteUniform(1, 5));
julia> rand(prior)
(0.06423613735728075, 2)
```
"""
rand(rng::AbstractRNG, factoreddist::Factored{N}) where {N} =
    ntuple(i -> rand(rng, factoreddist.p[i]), Val(N))

"""
    length(p::Factored)

Returns the number of distributions contained in `p`.
"""
length(p::Factored{N}) where {N} = N
