using Distributions

function rsimplex(n)
    x = rand(n)
    x / sum(x)
end

function mvbern(y, θ)
    prod(pdf(Bernoulli(p), y[i]) for (i, p) in enumerate(θ))
end

function marginal_pmf(y, θ, π)
    sum(π[k] * mvbern(y, θ[:, k]) for k in 1:length(π))
end

function log_lik(Y, θ, π)
    sum(log(marginal_pmf(Y[:, i], θ, π)) for i in 1:size(Y, 2))
end

function E_step(Y, θ, π)
    γ = zeros(Float64, size(y, 2), length(π))

    for k in 1:length(π)
        γ[:, k] = π[k] .* [mvbern(Y[:, i], θ[:, k]) for i in 1:size(Y, 2)]
    end

    γ ./ sum(γ, dims=2)
end

User
Here is my complete code implementing a BMM estimated using the EM algorithm.

```julia
using Distributions

function rsimplex(n)
    x = rand(n)
    x / sum(x)
end

function mvbern(y, θ)
    prod(pdf(Bernoulli(p), y[i]) for (i, p) in enumerate(θ))
end

function marginal_pmf(y, θ, π)
    sum(π[k] * mvbern(y, θ[:, k]) for k in 1:length(π))
end

function log_lik(Y, θ, π)
    sum(log(marginal_pmf(Y[:, i], θ, π)) for i in 1:size(Y, 2))
end

function E_step(Y, θ, π)
    γ = zeros(Float64, size(y, 2), length(π))

    for k in 1:length(π)
        γ[:, k] = π[k] .* [mvbern(Y[:, i], θ[:, k]) for i in 1:size(Y, 2)]
    end

    γ ./ sum(γ, dims=2)
end

function M_step(Y, γ)
    cluster_sums = sum(γ, dims = 1)

    θ = Y * γ
    for k in 1:size(θ, 2)
        θ[:, k] /=  cluster_sums[k]
    end

    π = cluster_probs / size(Y, 2)

    return (θ, π)
end

function EM(Y, K; max_iter=1e4, tol=1e-6)
    # Randomly initialize our parameters
    π = rsimplex(K)
    θ = rand(size(Y, 1), K)

    ll_old = ll_new = -Inf
    for i in 1:max_iter
        if i % 10 == 10
            print("Iteration: $(i)", i)
        end

        γ = E_step(Y, θ, π)
        θ, π = M_step(Y, γ)

        ll_new = log_lik(Y, θ, π)
        if abs(ll_new - ll_old) < tol
            break
        end

        ll_old = ll_new
    end

    return (θ, π, γ, ll_new)
end

N = 1_000
D = 3
K = 2

pi = rand(K)
normalize!(pi, 1)

theta = rand(D, K)
