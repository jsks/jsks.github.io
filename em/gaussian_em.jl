using Distributions, LinearAlgebra, LogExpFunctions

function log_lik(Y, π, μ, Σ)
    @views d = MixtureModel(map(i -> MvNormal(μ[:, i], Σ[:, :, i]), 1:length(π)), π)

    aux = Threads.Atomic{Float64}(0.0)
    Threads.@threads for i in 1:size(Y, 2)
        Threads.atomic_add!(aux, logpdf(d, view(Y, :, i)))
    end

    return aux[]
end

function E_step!(Y, γ, π, μ, Σ)
    lpi = log.(π)
    @views dist = [MvNormal(μ[:, k], Σ[:, :, k]) for k in 1:length(π)]

    Threads.@threads for i in 1:size(Y, 2)
        lp = [lpi[k] + logpdf(dist[k], view(Y, :, i)) for k in 1:length(π)]
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

function label_mapping(clusters, labels)
    Dict(k => mode(labels[clusters .== k])
         for k in 1:maximum(clusters) if k in clusters)
end

###
# Real data
using MLDatasets

pixels, labels = MNIST(split=:train)[:]
model_input = reshape(pixels, (28*28, 60_000))

fit = EM(model_input, 10)

using MultivariateStats
X = (model_input .- mean(model_input, dims=1)) ./ std(model_input, dims=1)
pca = fit(PCA, X, maxoutdim=50)

data = transform(pca, X)
ml = EM(data[:, labels .== 8], 5)


out = reconstruct(pca, ml[2])
co = (out .- minimum(out, dims=1)) ./ (maximum(out, dims=1) .- minimum(out, dims=1))
plts = [Gray.(reshape(1 .- co[:, i], 28, 28)') |> plot for i in 1:5]
plot(plts..., layout = size(co, 2), axis = false, ticks= false)


plot(Gray.(reshape(p, 28, 28)'))

###
# Simulated data
using CSV, DataFrames

labels = readlines("labels.txt")
df = CSV.read("test.csv", DataFrame)

labels = parse.(Int, labels)
data = Matrix(df)

fit = EM(data', 3)
γ = fit[4]

clusters = map(argmax, eachrow(γ))
mapping = label_mapping(clusters, labels)
Z_hat = [mapping[i] for i in clusters]

mean(Z_hat .== labels)
