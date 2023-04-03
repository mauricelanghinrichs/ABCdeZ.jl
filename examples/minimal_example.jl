
### minimal example for abcdesmc! method to obtain 
### posterior samples + model evidence estimates
### (NOTE: for abcdemc! method visit the documentation)

using ABCdeZ
using Distributions

### data
data = 3

### ABC target ϵ (maximum distance)
ϵ = 0.3

### model 1 inference
σ₁² = 10
prior1 = Normal(0, sqrt(σ₁²))

# model simulation (to replace likelihood)
model1(θ) = rand(Normal(θ, 1))

# distance function between model and data
dist1!(θ, ve) = abs(model1(θ)-data), nothing

### ABC run
# run the smc method for model 1
r1 = abcdesmc!(prior1, dist1!, ϵ, nothing, 
                    nparticles=1000, parallel=true)

### process results
# posterior parameters
# NOTE: when using abcdesmc!, parameters samples are
# associated with weights Wns (here just 0 vs. const>0)
posterior1 = [t[1] for t in r1.P[r1.Wns .> 0.0]]

# model evidence (logZ is the logarithmic evidence)
evidence1 = exp(r1.logZ)

### model 2 inference
### (on same data, same distance (abs()), same target ϵ)
σ₂² = 100
prior2 = Normal(0, sqrt(σ₂²))

# NOTE: here, model 2 differs from model 1 only in the prior;
# of course one could also adapt the following line (to implement, 
# what is usually the case, models with completely different 
# structure / parameters etc.)
model2(θ) = model1(θ)

dist2!(θ, ve) = abs(model2(θ)-data), nothing

r2 = abcdesmc!(prior2, dist2!, ϵ, nothing, 
                    nparticles=1000, parallel=true)

posterior2 = [t[1] for t in r2.P[r2.Wns .> 0.0]]
evidence2 = exp(r2.logZ)

### model probabilities
# model priors (uniform here)
mprior1 = 0.5
mprior2 = 0.5

# model posterior probabilities
mposterior1 = evidence1*mprior1 / (evidence1*mprior1 + evidence2*mprior2) # posterior prob. model 1
mposterior2 = evidence2*mprior2 / (evidence1*mprior1 + evidence2*mprior2) # posterior prob. model 2

### analytical comparison
# for this example we can calculate the posterior
# distributions and model evidences analytically
# (usually not possible; to compare with the ABC runs)
posterior1_exact = Normal(σ₁²/(σ₁²+1)*data, sqrt(σ₁²/(σ₁²+1)))
posterior2_exact = Normal(σ₂²/(σ₂²+1)*data, sqrt(σ₂²/(σ₂²+1)))

# abcdesmc! determines the model evidence up to a factor 
# (the normalisation constant of the ABC indicator kernel, 
# that, in this simple example, is knowned as 2ϵ)
evidence1_exact = pdf(Normal(0, sqrt(σ₁²+1)), data)
evidence1_expected = evidence1_exact * 2ϵ # ≈ evidence1 (above)

evidence2_exact = pdf(Normal(0, sqrt(σ₂²+1)), data)
evidence2_expected = evidence2_exact * 2ϵ # ≈ evidence2 (above)

mposterior1_exact = evidence1_exact*mprior1 / (evidence1_exact*mprior1 + evidence2_exact*mprior2)
mposterior2_exact = evidence2_exact*mprior2 / (evidence1_exact*mprior1 + evidence2_exact*mprior2)
# NOTE: the normalisation factor does not matter when comparing models 
# from ABC runs with the same ϵ target (as it would cancel in above lines)

### plots / visualise results
using Plots

θrange = collect(-10:0.01:10)
plot(θrange, pdf.(prior1, θrange), lw=3.0, c=:red, label="Prior", 
    grid=false, ylabel="Probability", xlabel="Parameter θ", title="Model 1",
    alpha=0.8, bg_legend=:transparent,
    fg_legend=:transparent, legend=:outertopright)
plot!(θrange, pdf.(posterior1_exact, θrange), lw=3.0, c=:deepskyblue, 
    label="Posterior (exact)", alpha=0.5)
histogram!(posterior1, normed=true, alpha=0.8, lw=0.0, c=:deepskyblue,
    linecolor=:deepskyblue, label="Posterior (ABCdeZ)")

plot(θrange, pdf.(prior2, θrange), lw=3.0, c=:red, label="Prior", 
    grid=false, ylabel="Probability", xlabel="Parameter θ", title="Model 2",
    alpha=0.8, bg_legend=:transparent,
    fg_legend=:transparent, legend=:outertopright)
plot!(θrange, pdf.(posterior2_exact, θrange), lw=3.0, c=:deepskyblue, 
    label="Posterior (exact)", alpha=0.5)
histogram!(posterior2, normed=true, alpha=0.8, lw=0.0, c=:deepskyblue,
    linecolor=:deepskyblue, label="Posterior (ABCdeZ)")

# NOTE: to obtain some idea about the (numerical) uncertainty of evidence 
# estimates from ABCdeZ, one may want to do 5 abcdesmc! runs each and 
# aggregate (which was also done for the images on github/docs)
scatter(["Model 1", "Model 2"], [mposterior1, mposterior2], ylims=(0, 1), 
    xlims=(0.2, 1.8), xrotation=45, ms=5.0, markerstrokewidth=0,
    c=:deepskyblue, alpha=0.8, grid=false, 
    ylabel="Probability", title="Model selection",
    label="Posterior (ABCdeZ)", legend_background_color=:transparent,
    fg_legend=:transparent, legend=:outertopright)
plot!([0.35, 0.65], [mposterior1_exact, mposterior1_exact], lw=2.0, c=:red,
    label="Posterior (exact)")
plot!([1.35, 1.65], [mposterior2_exact, mposterior2_exact], lw=2.0, c=:red,
    label=nothing)
