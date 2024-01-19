using Distributions, LinearAlgebra, LogExpFunctions

function gaussians(μ, Σ)
    @views [MvNormal(μ[:, i], Σ[:, :, i]) for i in 1:size(μ, 2)]
end

function log_lik(Y, π, μ, Σ)
    d = MixtureModel(gaussians(μ, Σ), π)
    sum(logpdf(d, y) for y in eachcol(Y))
end

function E_step!(Y, γ, π, μ, Σ)
    dists = gaussians(μ, Σ)
    lpi = log.(π)

    for i in 1:size(Y, 2)
        lp = [lpi[k] + logpdf(dists[k], view(Y, :, i)) for k in 1:length(π)]
        denominator = logsumexp(lp)

        for k in 1:length(π)
            γ[i, k] = exp(lp[k] - denominator)
        end
    end
end

function M_step!(Y, γ, π, μ, Σ)
    cluster_sums = vec(sum(γ, dims=1))
    π .= cluster_sums ./ size(Y, 2)

    mul!(μ, Y, γ)
    μ ./= cluster_sums'

    @views for k in 1:length(π)
        Y_centered = Y .- μ[:, k]
        Σ_k = zeros(size(Y, 1), size(Y, 1))
        for i in 1:size(Y, 2)
            Σ_k += γ[i, k] * (Y_centered[:, i] * Y_centered[:, i]')
        end
        Σ[:, :, k] .= Σ_k / cluster_sums[k]
    end
end

function EM(Y, K; max_iter=10_000, tol=1e-5)
    π = fill(1/K, K)
    μ = Y[:, rand(1:size(Y, 2), K)]
    Σ = cat([Diagonal(ones(size(Y, 1))) for k in 1:K]..., dims=3)
    γ = zeros(size(Y, 2), K)

    log_lik_prev = log_lik_new = -Inf
    for i in 1:max_iter
        E_step!(Y, γ, π, μ, Σ)
        M_step!(Y, γ, π, μ, Σ)

        log_lik_new = log_lik(Y, π, μ, Σ)
        println("Iteration $i: Log Likelihood = $log_lik_new")

        if abs(log_lik_new - log_lik_prev) < tol
            return π, μ, Σ, γ
        end

        log_lik_prev = log_lik_new
    end

    error("Model failed to converge")
end

function predict(Y, results)
    dists = [MixtureModel(gaussians(μ, Σ), π) for (π, μ, Σ, _) in results]
    [argmax(logpdf(d, y) for d in dists) - 1 for y in eachcol(Y)]
end

###
# Real data
using Base.Threads, Colors, MLDatasets, MultivariateStats, Plots

pixels, labels = MNIST(split=:train)[:]
model_input = reshape(pixels, (28*28, 60_000))

###
# Dimensionality reduction
pca = fit(PCA, model_input, maxoutdim=50)
reduced_data = transform(pca, model_input)

###
# Parallelized run --- one model per digit
results = fetch.([@spawn EM(reduced_data[:, labels .== i], 4) for i in 0:9])

# Plotted marginal expectations --- ie, the image of the predicted
# 'average' digit per model
EY = map(((π, μ, _, _),) -> reconstruct(pca, μ * π), results)
plts = [Gray.(reshape(μ, 28, 28)') |> plot for μ in EY]
plot(plts..., layout = length(EY), axis = false, ticks = false)

###
# Classify by finding the model with the highest log-likelihood for
# the data
Z_hat = predict(reduced_data, results)
mean(Z_hat .== labels)

###
# Test predictive accuracy of our test data
test_pixels, test_labels = MNIST(split=:test)[:]
test_input = reshape(test_pixels, (28*28, 10_000))
test_reduced = transform(pca, test_input)

Z_test = predict(test_reduced, results)
mean(Z_test .== test_labels)
