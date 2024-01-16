X = model_input'
Y = model_input

π = fill(1/K, K)
θ = rand(Float64, size(Y, 1), K)
γ = zeros(Float64, size(Y, 2), K)

N, D = size(X)
#new_pi, new_theta = initialize_parameters(K, D)
new_pi = deepcopy(π)
new_theta = deepcopy(θ)'

log_likelihood_prev = ll_old = -Inf
log_likelihood = ll_new = -Inf

for i in 1:5
    # Optimized algorithm
    E_step!(Y, γ, θ, π)
    M_step!(Y, γ, θ, π)

    new_gamma = e_step(X, new_pi, new_theta)
    new_pi, new_theta = m_step(X, new_gamma)

    log_likelihood = sum([log(sum([new_pi[k] * prod(new_theta[k, :] .^ X[i, :] .* (1 .- new_theta[k, :]) .^ (1 .- X[i, :]))
                                  for k in 1:K]))
                         for i in 1:N])

    ll_new = log_lik(Y, θ, π)

    log_likelihood_prev = log_likelihood
    ll_old = ll_new

end
