using LinearAlgebra, LogExpFunctions, Octavian, StatsBase

function log_mvbernoulli_pmf(y, lp, lp1m)
    sum((pixel == 1) ? lp[i] : lp1m[i] for (i, pixel) in enumerate(y))
end

function log_mvbernoulli_pmf!(Y, ν, θ, π)
    lpi = log.(π)
    lp = log.(θ)
    lp1m = log.(1 .- θ)

    Threads.@threads for i in 1:size(Y, 2)
        @views for k in 1:length(lpi)
            @inbounds ν[i, k] = lpi[k] + log_mvbernoulli_pmf(Y[:, i], lp[:, k], lp1m[:, k])
        end
    end
end

function log_lik(ν)
    logsumexp(ν, dims=2) |> sum
end

function E_step!(Y, γ, ν)
    γ .= ν .- logsumexp(ν, dims=2)
    γ .= exp.(γ)
end

function M_step!(Y, γ, θ, π)
    cluster_sums = vec(sum(γ, dims=1))

    π .= cluster_sums ./ size(Y, 2)

    matmul!(θ, Y, γ)
    θ ./= cluster_sums'
    clamp!(θ, eps(), 1 - eps())
end

function EM(Y, K; max_iter=1_000, tol=1e-5)
    π = fill(1/K, K)
    θ = rand(Float64, size(Y, 1), K)

    γ = zeros(Float64, size(Y, 2), K)
    ν = copy(γ)
    log_mvbernoulli_pmf!(Y, ν, θ, π)

    log_lik_prev = log_lik_new = -Inf
    for i in 1:max_iter
        E_step!(Y, γ, ν)
        M_step!(Y, γ, θ, π)

        log_mvbernoulli_pmf!(Y, ν, θ, π)
        log_lik_new = log_lik(ν)

        log_lik_diff = abs(log_lik_new - log_lik_prev)
        println("Iteration $(i): log-likelihood = $(log_lik_new)")
        if log_lik_diff < tol
            return θ, π, γ
        end

        log_lik_prev = log_lik_new
    end

    error("Model failed to converge")
end

function label_mapping(clusters, labels)
    Dict(k => mode(labels[clusters .== k])
         for k in 1:maximum(clusters) if k in clusters)
end

using MLDatasets

pixels, labels = MNIST(split=:train)[:]
binned_pixels = collect(pixels .> 0.5)

model_input = reshape(binned_pixels, (28*28, 60_000))
θ, π, γ  = EM(model_input, 10)

clusters = map(argmax, eachrow(γ))
mapping = label_mapping(clusters, labels)
Z_hat = [mapping[i] for i in clusters]

mean(Z_hat .== labels)

###
# Plot predicted images
using Colors, Plots

plts = [Gray.(reshape(1 .- θ[:, i], 28, 28)') |> plot for i in 1:size(θ, 2)]
plot(plts..., layout = size(θ, 2), axis = false, ticks = false)
