
### minimal example for abcdesmc! method to obtain 
### posterior samples + model evidence estimates

using ABCdeZ
using Distributions

### required input
# data and prior distribution
data = 5
prior = Normal(0, sqrt(10))

# model simulation (to replace likelihood)
model(θ) = rand(Normal(θ, 1))

# distance function between model and data
dist!(θ, ve) = abs(model(θ)-data), nothing

### ABC run
# target ϵ (maximum distance)
ϵ = 0.3

# run the smc method
r = abcdesmc!(prior, dist!, ϵ, nothing, 
                    nparticles=1000, parallel=true)

### process results
# posterior parameters
# NOTE: when using abcdesmc!, parameters samples are
# associated with weights Wns (here just 0 vs. const>0)
posterior = [t[1] for t in r.P[r.Wns .> 0.0]]

# model evidence (logZ is the logarithmic evidence)
evidence = exp(r.logZ)

### analytical comparison
# for this example we can calculate the posterior
# distribution and model evidence analytically
# (usually not possible; compared to the ABC run)
posterior_exact = Normal(10/11*data, sqrt(10/11))

# abcdesmc! determines the model evidence up to a factor 
# (the normalisation constant of the ABC indicator kernel, 
# that, in this simple example, is knowned as 2ϵ)
evidence_exact = pdf(Normal(0, sqrt(11)), data)
evidence_expected = evidence_exact * 2ϵ # ≈ evidence (above)

### plots
using Plots

θrange = collect(-15:0.01:15)
plot(θrange, pdf(prior, θrange))
plot!(θrange, pdf(posterior_exact, θrange))
histogram!(posterior, normed=true, alpha=0.5)
