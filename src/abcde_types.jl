
### particle definition and operations
struct Particle{Xt}
    x::Xt
    Particle(x::T) where {T} = new{T}(x)
end

op(f, a::Particle, b::Particle) = Particle(op(f, a.x, b.x))
op(f, a::Particle, b::Number) = Particle(op(f, a.x, b))
op(f, a::Number, b::Particle) = Particle(op(f, a, b.x))
op(f, a::Number, b::Number) = f(a, b)
op(f, a, b) = op.(Ref(f), a, b)

op(f, a::Particle) = Particle(op(f, a.x))
op(f, a::Number) = f(a)
op(f, a) = op.(Ref(f), a)

op(f, args...) = foldl((x, y) -> op(f, x, y), args)

push_p(density::Factored, p) = push_p.(density.p, p)
push_p(density::Distribution, p) = push_p.(Ref(density), p)
push_p(density::ContinuousDistribution, p::Number) = float(p)
push_p(density::DiscreteDistribution, p::Number) = round(Int, p)

### unnormalised indicator kernel
struct Indicator0toϵ <: ContinuousUnivariateDistribution
    ϵ::Float64

    function Indicator0toϵ(ϵ)
        ϵ ≥ 0.0 || error("Expected ϵ ≥ 0.0")
        new(ϵ)
    end
end
Distributions.insupport(d::Indicator0toϵ, x::Real) = 0.0 ≤ x ≤ d.ϵ ? true : false
Distributions.pdf(d::Indicator0toϵ, x::Real) = insupport(d, x) ? 1.0 : 0.0
Distributions.logpdf(d::Indicator0toϵ, x::Real) = insupport(d, x) ? 0.0 : -Inf
