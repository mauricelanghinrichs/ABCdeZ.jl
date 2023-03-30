
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

###### various notes:
# - multidimensional prior distribution (continuous, discrete or mixed) can 
#   be done via the Factored() syntax, e.g. this specifies a 2d prior:
prior2d = Factored(Normal(0, sqrt(10)), DiscreteUniform(1, 10))
rand(prior2d)

# - in the distance function above (dist!(θ, ve) = abs(model(θ)-data), nothing)
#   ve are "external variables" that can be used in the distance computation 
#   and mutated in-place, even in the parallel mode (each thread base will obtain 
#   its own copy for thread-safe parallel ABC runs);
#   ve is passed as 4th positional argument to abcdesmc! (here: "nothing")

# - in the distance function above (dist!(θ, ve) = abs(model(θ)-data), nothing) 
#   the second return argument ("nothing") can be used to store arbitrary data 
#   ("blobs") to each particle; these blobs will be associated with the final 
#   posterior samples/particles in the end

#   for example blobs could record the actual simulation output:
function dist2!(θ, ve, constants, data)
    # constants can be used to pass thread-safe constants that are NOT mutated;
    # ve for in-place, mutatable variables

    # distance method
    simdata = model(θ)
    blob = simdata
    d = abs(simdata-data)
    d, blob
end
dist2!(θ, ve) = dist2!(θ, ve, nothing, data)

r = abcdesmc!(prior, dist2!, ϵ, nothing, 
                    nparticles=1000, parallel=true)

posterior = [t[1] for t in r.P[r.Wns .> 0.0]]
evidence = exp(r.logZ)
blobs = r.blobs[r.Wns .> 0.0]

# - model comparison can be done with the model evidences obtained by 
#   abcdesmc!; as the evidence values are (typically) off by the 
#   normalisation constant, different models (each inferred from a separate 
#   ABC run) have to be compared with logZs computed for the SAME ϵ 
#   (then the unknown normalisation constants cancel and don't matter)
#   (of course, next to the same data and distance method);

#   for example posterior model probabilities are computed like this 
#   (for a uniform model prior)
m1_evidence = evidence # evidence of model 1 from above
m2_evidence = 0.008 # from some other ABC run

m1_prior = 0.5 # model priors (uniform here)
m2_prior = 0.5

m1_prob = m1evid*m1_prior / (m1evid*m1_prior + m2evid*m2_prior) # posterior prob. model 1
m2_prob = m2evid*m2_prior / (m1evid*m1_prior + m2evid*m2_prior) # posterior prob. model 2
