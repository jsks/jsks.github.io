using Hungarian, LinearAlgebra, LogExpFunctions, Octavian, Printf, StatsBase

function log_mvbernoulli_pmf(y, lp, lp1m)
    sum((pixel == 1) ? lp[i] : lp1m[i] for (i, pixel) in enumerate(y))
end

function log_marginal_pmf(y, lp, lp1m, lpi)
    logsumexp(lpi[k] + log_mvbernoulli_pmf(y, view(lp, :, k), view(lp1m, :, k))
              for k in 1:length(π))
end

function log_lik(Y, θ, π)
    lpi = log.(π)
    lp = log.(θ)
    lp1m = log.(1 .- θ)

    #aux = Threads.Atomic{Float64}(0.0)
    #Threads.@threads for i in 1:size(Y, 2)
        #Threads.atomic_add!(aux, log_marginal_pmf(view(Y, :, i), lp, lp1m, lpi))
    #end

    sum(log_marginal_pmf(view(Y, :, i), lp, lp1m, lpi) for i in 1:size(Y, 2))
end

function E_step!(Y, γ, θ, π)
    lpi = log.(π)
    lp = log.(max.(eps(), θ))
    lp1m = log.(max.(eps(), 1 .- θ))

    for i in 1:size(Y, 2)
        @views log_resps = [lpi[k] + log_mvbernoulli_pmf(Y[:, i], lp[:, k], lp1m[:, k])
                            for k in 1:length(π)]
        denominator = logsumexp(log_resps)

        for k in 1:length(π)
            γ[i, k] = exp(log_resps[k] - denominator)
        end
    end
end

function M_step!(Y, γ, θ, π)
    cluster_sums = vec(sum(γ, dims=1))

    π .= cluster_sums ./ size(Y, 2)

    matmul!(θ, Y, γ)
    θ ./= cluster_sums'
    clamp!(θ, eps(), 1 - eps())
end

function EM(Y, K; max_iter=1_000, tol=1e-6)
    π = fill(1/K, K)
    θ = rand(Float64, size(Y, 1), K)
    γ = zeros(Float64, size(Y, 2), K)

    ll_old = ll_new = ll_diff = -Inf
    for i in 1:max_iter
        E_step!(Y, γ, θ, π)
        M_step!(Y, γ, θ, π)

        ll_new = log_lik(Y, θ, π)
        #@printf "Iteration %d: log-likelihood = %.7f\n" i ll_new
        println("Iteration $(i): log-likelihood = $(ll_new)")
        if abs(ll_new - ll_old) < tol
            return (θ, π, γ, ll_new)
        end

        ll_old = ll_new
    end

    error("Model failed to converge")
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

pixels, labels = MNIST(split=:train)[:]
binned_pixels = pixels .> 0.5

model_input = reshape(binned_pixels, (28*28, 60_000))
fit = EM(model_input, 10)

Z_hat = relabel(fit[3], labels)
mean(Z_hat .== labels)
