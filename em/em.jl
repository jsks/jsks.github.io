using Distributions, Hungarian, LinearAlgebra, StatsBase

function initialize_parameters(K, D)
    π = rand(K)
    π /= sum(π)
    θ = 0.5 * ones(K, D) + 0.1 * rand(K, D)  # Avoiding extreme values
    return π, θ
end

function logsumexp(a)
    m = maximum(a)
    return m + log(sum(exp.(a .- m)))
end

function e_step(X, π, θ)
    N, D = size(X)
    K = length(π)
    γ = zeros(N, K)

    for i in 1:N
        for k in 1:K
            log_θ = log.(max.(eps(), θ[k, :]))
            log_1mθ = log.(max.(eps(), 1 .- θ[k, :]))
            γ[i, k] = log(π[k]) + sum(X[i, :] .* log_θ + (1 .- X[i, :]) .* log_1mθ)
        end
        γ[i, :] = exp.(γ[i, :] .- logsumexp(γ[i, :]))
    end

    return γ
end

function m_step(X, γ)
    N, D = size(X)
    K = size(γ, 2)

    Nk = sum(γ, dims=1)
    π = Nk ./ N
    θ = (γ' * X) ./ Nk'
    θ = clamp.(θ, eps(), 1-eps())  # Clipping to avoid extreme values

    return π[:], θ
end

function bernoulli_mixture_em(X, K; max_iter=10000, tol=1e-6)
    N, D = size(X)
    π, θ = initialize_parameters(K, D)

    prev_log_likelihood = -Inf
    for iter in 1:max_iter
        γ = e_step(X, π, θ)
        π, θ = m_step(X, γ)

        log_likelihood = sum([log(sum([π[k] * prod(θ[k, :] .^ X[i, :] .* (1 .- θ[k, :]) .^ (1 .- X[i, :]))
                                      for k in 1:K]))
                             for i in 1:N])
        println("Iteration $iter: Log Likelihood = $log_likelihood")

        if iter == 30 || abs(log_likelihood - prev_log_likelihood) < tol
            return θ, π, γ
        end
        prev_log_likelihood = log_likelihood
    end
end

function relabel(γ, labels)
    clusters = map(argmax, eachrow(γ))

    cm = zeros(Int, size(γ, 2), maximum(labels) + 1)
    for i in 1:length(clusters)
        cm[clusters[i], labels[i] + 1] += 1
    end

    assignment, _ = hungarian(maximum(cm) .- cm)
    assignment[clusters] .- 1
end


# Set parameters
N = 10_000  # Number of data points
D = 784   # Number of features (28x28 pixels)
K = 10    # Number of mixture components

# Step 1: Generate mixture probabilities
Random.seed!(123)  # For reproducibility
π = rand(K)
π /= sum(π)

# Step 2: Generate Bernoulli parameters for each mixture component
θ = rand(K, D)

# Step 3: Simulate data points and save the true cluster labels
data = zeros(N, D)
true_labels = zeros(Int, N)  # Array to store the true cluster labels

for i in 1:N
    # Choose a mixture component for each data point
    random_value = rand()
    k = findfirst(cumsum(π) .>= random_value)
    true_labels[i] = k  # Save the true cluster label

    # Generate data based on the Bernoulli parameters of the chosen component
    for j in 1:D
        data[i, j] = rand() < θ[k, j] ? 1.0 : 0.0
    end
end

fit = bernoulli_mixture_em(data, K)
mean(relabel(fit[3], true_labels) .== true_labels)
