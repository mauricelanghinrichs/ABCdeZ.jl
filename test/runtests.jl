
using ABCdeZ
using Distributions
using Random
using Test

Random.seed!(1)

isaround(θ, val) = (mean(θ)-std(θ) ≤ val ≤ mean(θ)+std(θ))

@testset "Factored" begin
    # tests copied and/or adapted from KissABC.jl
    d = Factored(Uniform(0, 1), Uniform(100, 101))
    @test all((0, 100) .<= rand(d) .<= (1, 101))
    @test pdf(d, (0.0, 0.0)) == 0.0
    @test pdf(d, (0.5, 100.5)) == 1.0
    @test logpdf(d, (0.5, 100.5)) == 0.0
    @test logpdf(d, (0.0, 0.0)) == -Inf
    @test length(d) == 2
    m = Factored(Uniform(0.00, 1.0), DiscreteUniform(1, 2))
    sample = rand(m)
    @test 0 < sample[1] < 1
    @test sample[2] == 1 || sample[2] == 2
    @test pdf(m, sample) == 0.5
    @test logpdf(m, sample) ≈ log(0.5)
end

@testset "Push" begin
    # tests copied and/or adapted from KissABC.jl
    push_p = ABCdeZ.push_p
    a /′ b = (typeof(a) == typeof(b)) && all(a .== b)
    @test push_p(Normal(), 1) /′ 1.0
    @test push_p(DiscreteUniform(), 1.0) /′ 1
    @test push_p(Factored(Normal(), DiscreteUniform()), (2, 1.0)) /′ (2.0, 1)
    @test push_p(product_distribution([Normal(), Normal()]), [2, 1]) /′ [2.0, 1.0]
end

@testset "Indicator kernel" begin
    d = ABCdeZ.Indicator0toϵ(0.1)
    @test d.ϵ == 0.1
    @test (pdf(d, 0.1), logpdf(d, 0.1)) == (1.0, 0.0)
    @test (pdf(d, -0.1), logpdf(d, -0.1)) == (0.0, -Inf)
    @test (pdf(d, 0.2), logpdf(d, 0.2)) == (0.0, -Inf)

    d = ABCdeZ.Indicator0toϵ(Inf)
    @test d.ϵ == Inf
    @test (pdf(d, 0.1), logpdf(d, 0.1)) == (1.0, 0.0)
    @test (pdf(d, -0.1), logpdf(d, -0.1)) == (0.0, -Inf)
    @test (pdf(d, 0.2), logpdf(d, 0.2)) == (1.0, 0.0)
end

@testset "1d Normal with Evidence (data=3)" begin
    xdata = 3 # 3, 5, 7
    dprior = Normal(0, sqrt(10))
    dposterior = Normal(10/11*xdata, sqrt(10/11))
    dmodel(μ) = Normal(μ, 1) # for likelihood
    evidence_exact = pdf(Normal(0, sqrt(11)), xdata)

    ϵ = 0.3
    # exact "unnormalised" evidence
    evidence_exact_indicator = evidence_exact * 2ϵ
    println("Z exact = ", evidence_exact_indicator)
    @test isapprox(0.047940112540007955, evidence_exact_indicator)

    ### ABC rejection and model evidence with unnormalised indicator kernel
    nsamples = Int(1e6)
    sprior = rand(dprior, nsamples)
    smodel = rand.(dmodel.(sprior))

    k_indicator = (abs.(smodel .- xdata) .< ϵ)

    saccept = k_indicator .> 0.0
    sposterior = sprior[saccept]

    # direct Monte Carlo evidence estimation (from prior samples)
    Zest = mean(k_indicator)
    println("Z rejection = ", Zest)
    @test evidence_exact_indicator * 0.9 ≤ Zest ≤ 1.1 * evidence_exact_indicator

    ### using abcdesmc! method for evidence and posterior
    dist!(θ, ve) = abs(rand(dmodel(θ)) - xdata), nothing

    r = abcdesmc!(dprior, dist!, ϵ, nothing,
                        nparticles=5000, verbose=false, 
                        verboseout=true, parallel=true)

    Zest = exp(r.logZ)
    println("Z abcdesmc! = ", Zest)
    @test evidence_exact_indicator * 0.9 ≤ Zest ≤ 1.1 * evidence_exact_indicator

     ### using abcdemc! method for posterior only
    rmc = abcdemc!(dprior, dist!, ϵ, nothing,
                        nparticles=5000, generations=500,
                        verbose=false, parallel=true)

    println("Posterior mean exact = ", mean(dposterior))
    println("Posterior mean rejection = ", mean(sposterior))
    println("Posterior mean abcdemc! = ", mean([t[1] for t in rmc.P]))
    println("Posterior mean abcdesmc! = ", mean([t[1] for t in r.P[r.Wns .> 0.0]]))

    @test isapprox(2.727272727272727, mean(dposterior))
    @test isaround(sposterior, mean(dposterior))
    @test isaround([t[1] for t in rmc.P], mean(dposterior))
    @test isaround([t[1] for t in r.P[r.Wns .> 0.0]], mean(dposterior))
end

@testset "1d Normal with Evidence (data=7)" begin
    xdata = 7 # 3, 5, 7
    dprior = Normal(0, sqrt(10))
    dposterior = Normal(10/11*xdata, sqrt(10/11))
    dmodel(μ) = Normal(μ, 1) # for likelihood
    evidence_exact = pdf(Normal(0, sqrt(11)), xdata)

    ϵ = 0.3
    # exact "unnormalised" evidence
    evidence_exact_indicator = evidence_exact * 2ϵ
    println("Z exact = ", evidence_exact_indicator)
    @test isapprox(0.007781668367620676, evidence_exact_indicator)

    ### ABC rejection and model evidence with unnormalised indicator kernel
    nsamples = Int(1e6)
    sprior = rand(dprior, nsamples)
    smodel = rand.(dmodel.(sprior))

    k_indicator = (abs.(smodel .- xdata) .< ϵ)

    saccept = k_indicator .> 0.0
    sposterior = sprior[saccept]

    # direct Monte Carlo evidence estimation (from prior samples)
    Zest = mean(k_indicator)
    println("Z rejection = ", Zest)
    @test evidence_exact_indicator * 0.9 ≤ Zest ≤ 1.1 * evidence_exact_indicator

    ### using abcdesmc! method for evidence and posterior
    dist!(θ, ve) = abs(rand(dmodel(θ)) - xdata), nothing

    r = abcdesmc!(dprior, dist!, ϵ, nothing,
                        nparticles=5000, verbose=false, 
                        verboseout=true, parallel=true)

    Zest = exp(r.logZ)
    println("Z abcdesmc! = ", Zest)
    @test evidence_exact_indicator * 0.9 ≤ Zest ≤ 1.1 * evidence_exact_indicator

     ### using abcdemc! method for posterior only
    rmc = abcdemc!(dprior, dist!, ϵ, nothing,
                        nparticles=5000, generations=500,
                        verbose=false, parallel=true)

    println("Posterior mean exact = ", mean(dposterior))
    println("Posterior mean rejection = ", mean(sposterior))
    println("Posterior mean abcdemc! = ", mean([t[1] for t in rmc.P]))
    println("Posterior mean abcdesmc! = ", mean([t[1] for t in r.P[r.Wns .> 0.0]]))

    @test isapprox(6.363636363636363, mean(dposterior))
    @test isaround(sposterior, mean(dposterior))
    @test isaround([t[1] for t in rmc.P], mean(dposterior))
    @test isaround([t[1] for t in r.P[r.Wns .> 0.0]], mean(dposterior))
end

@testset "Normal dist -> Dirac Delta inference, Parallel" begin
    # tests copied and/or adapted from KissABC.jl
    prior = Normal(1, 0.2)
    sim(μ) = μ * μ + 1
    dist!(θ, ve) = abs(sim(θ) - 1.5), nothing

    r = abcdemc!(prior, dist!, 0.1, nothing, verbose=false).P
    @test isaround([t[1] for t in r], 0.707)

    r = abcdesmc!(prior, dist!, 0.1, nothing, verbose=false)
    r = r.P[r.Wns .> 0.0]
    @test isaround([t[1] for t in r], 0.707)
end

@testset "Normal dist -> Dirac Delta inference, Parallel" begin
    # tests copied and/or adapted from KissABC.jl
    prior = Normal(1, 0.2)
    sim(μ) = μ * μ + 1
    dist!(θ, ve) = abs(sim(θ) - 1.5), nothing

    r = abcdemc!(prior, dist!, 0.1, nothing, verbose=false, parallel=true).P
    @test isaround([t[1] for t in r], 0.707)

    r = abcdesmc!(prior, dist!, 0.1, nothing, verbose=false, parallel=true)
    r = r.P[r.Wns .> 0.0]
    @test isaround([t[1] for t in r], 0.707)
end

@testset "Normal dist + Uniform Distr inference" begin
    # tests copied and/or adapted from KissABC.jl
    prior = Factored(Normal(1, 0.5), DiscreteUniform(1, 10))
    sim((n, du)) = (n * n + du) * (n + randn() * 0.01)
    dist!(θ, ve) = abs(sim(θ) - 5.5), nothing
    
    r = abcdemc!(prior, dist!, 0.01, nothing, nparticles=100, generations=1000, verbose=false).P
    @test isaround([t[1] for t in r], 1)
    @test isaround([t[2] for t in r], 5)

    r = abcdesmc!(prior, dist!, 0.01, nothing, nparticles=100, verbose=false)
    r = r.P[r.Wns .> 0.0]
    @test isaround([t[1] for t in r], 1)
    @test isaround([t[2] for t in r], 5)
end

function brownianrms((μ, σ), N, samples = 200)
    # tests copied and/or adapted from KissABC.jl
    t = 0:N
    #    rand()<1/20 && sleep(0.001)
    @.(sqrt(μ * μ * t * t + σ * σ * t)) .* (0.95 + 0.1 * rand())
end

@testset "Inference on drifted Wiener Process" begin
    # tests copied and/or adapted from KissABC.jl
    params = (0.5, 2.0)
    tdata = brownianrms(params, 30, 10000)
    prior = Factored(Uniform(0, 1), Uniform(0, 4))
    dist!(x, ve) = sum(abs, brownianrms(x, 30) .- tdata) / length(tdata), nothing

    r = abcdemc!(prior, dist!, 0.05, nothing, nparticles=500, generations=300, verbose=false).P
    @test isaround([t[1] for t in r], 0.5)
    @test isaround([t[2] for t in r], 2.0)

    r = abcdesmc!(prior, dist!, 0.05, nothing, nparticles=500, verbose=false)
    r = r.P[r.Wns .> 0.0]
    @test isaround([t[1] for t in r], 0.5)
    @test isaround([t[2] for t in r], 2.0)
end

@testset "Classical Mixture Model 0.1N+N" begin
    # tests copied and/or adapted from KissABC.jl
    st(res) =
        ((quantile(res, 0.1:0.1:0.9)-reverse(quantile(res, 0.1:0.1:0.9)))/2)[1+(end-1)÷2:end]
    st_n = [0.0,
            0.04680825481526908,
            0.1057221226763449,
            0.2682111969397526,
            0.8309228020477986]

    prior = Uniform(-10, 10)
    sim(μ) = μ + rand((randn() * 0.1, randn()))
    dist!(θ, ve) = abs(sim(θ) - 0.0), nothing

    rmc = abcdemc!(prior, dist!, 0.01, nothing, nparticles=2000, generations=1000, verbose=false).P
    rsmc = abcdesmc!(prior, dist!, 0.01, nothing, nparticles=2000, verbose=false)
    rsmc = rsmc.P[rsmc.Wns .> 0.0]

    testst(alg, r) = begin
        m = mean(abs, st(r) - st_n)
        println(":", alg, ": testing m = ", m)
        # @show r
        m < 0.1
    end

    @test testst("abcdemc!", [t[1] for t in rmc])
    @test testst("abcdesmc!", [t[1] for t in rsmc])
end

@testset "2d Problem" begin
    # tests copied and/or adapted from KissABC.jl
    prior = Factored(Normal(0, 5), Normal(0, 5))
    dist1!((x, y), ve) = 50 * (x + randn() * 0.01 - y^2)^2 + (y - 1 + randn() * 0.01)^2, nothing

    r = abcdemc!(prior, dist1!, 0.01, nothing, verbose=false, nparticles=500, generations=500, parallel=true).P
    @test isaround([t[1] for t in r], 1)
    @test isaround([t[2] for t in r], 1)

    r = abcdesmc!(prior, dist1!, 0.01, nothing, verbose=false, nparticles=500, parallel=true)
    r = r.P[r.Wns .> 0.0]
    @test isaround([t[1] for t in r], 1)
    @test isaround([t[2] for t in r], 1)

    dist2!((x, y), ve) = rand((50 * (x + randn() * 0.01 - y^2)^2 + (y - 1 + randn() * 0.01)^2,Inf)),nothing

    r = abcdemc!(prior, dist2!, 0.01, nothing, verbose=false, nparticles=500, generations=500, parallel=true).P
    @test isaround([t[1] for t in r], 1)
    @test isaround([t[2] for t in r], 1)

    r = abcdesmc!(prior, dist2!, 0.01, nothing, verbose=false, nparticles=500, parallel=true)
    r = r.P[r.Wns .> 0.0]
    @test isaround([t[1] for t in r], 1)
    @test isaround([t[2] for t in r], 1)
end
