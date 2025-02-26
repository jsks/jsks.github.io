using Test

@testset "Multivariate T-Distribution" begin
    x = [-2, 3]
    μ = [1, 2]
    Σ = [4 2; 2 3]

    @test multi_t_logpdf(Multi_t(1, μ, Σ), x) ≈ -5.656174 atol=1e-6
    @test multi_t_logpdf(Multi_t(2, μ, Σ), x) ≈ -5.487495 atol=1e-6
    @test multi_t_logpdf(Multi_t(3, μ, Σ), x) ≈ -5.444195 atol=1e-6
    @test multi_t_logpdf(Multi_t(5, μ, Σ), x) ≈ -5.432462 atol=1e-6
    @test multi_t_logpdf(Multi_t(10, μ, Σ), x) ≈ -5.458544 atol=1e-6
end

@testset "Posterior Calculations" begin
    function naive(X, priors)
        n = size(X, 2)
        x_bar = vec(mean(X, dims=2))
        S = scattermat(Matrix(X'))

        μ_n = (μ_0 * κ_0 / (κ_0 + n)) + (n * x_bar / (κ_0 + n))
        κ_n = κ_0 + n
        ν_n = ν_0 + n

        Λ_n = Λ_0 + S + (κ_0 * n / (κ_0 + n)) * (x_bar - μ_0) * (x_bar - μ_0)'

        df = ν_n - size(X, 1) + 1
        Σ_n = Λ_n * ((κ_n + 1) / (κ_n * df))

        Multi_t(df, μ_n, Σ_n)
    end

    X = rand(3, 5)

    μ_0 = zeros(Float64, size(X, 1))
    κ_0 = 0.01
    ν_0 = size(X, 1) + 1
    Λ_0 = Matrix(I(size(X, 1)))

    priors = Priors(κ_0, ν_0, μ_0, Λ_0)

    td = naive(X, priors)
    cluster = Cluster(X, priors)

    @test td.ν ≈ cluster.td.ν atol=1e-6
    @test td.μ ≈ cluster.td.μ atol=1e-6
    @test LowerTriangular(td.L) ≈ LowerTriangular(cluster.td.L) atol=1e-6

    @test multi_t_logpdf(td, X) ≈ multi_t_logpdf(cluster.td, X) atol=1e-6

    remove!(cluster, X[:, 1])
    td = naive(X[:, 2:end], priors)

    @test td.ν ≈ cluster.td.ν atol=1e-6
    @test td.μ ≈ cluster.td.μ atol=1e-6
    @test LowerTriangular(td.L) ≈ LowerTriangular(cluster.td.L) atol=1e-6

    @test multi_t_logpdf(td, X) ≈ multi_t_logpdf(cluster.td, X) atol=1e-6

    X = hcat(X[:, 2:end], rand(3))
    td = naive(X, priors)
    add!(cluster, X[:, end])

    @test td.ν ≈ cluster.td.ν atol=1e-6
    @test td.μ ≈ cluster.td.μ atol=1e-6
    @test LowerTriangular(td.L) ≈ LowerTriangular(cluster.td.L) atol=1e-6

    @test multi_t_logpdf(td, X) ≈ multi_t_logpdf(cluster.td, X) atol=1e-6

    rebuild!(cluster, X)
    @test td.ν ≈ cluster.td.ν atol=1e-6
    @test td.μ ≈ cluster.td.μ atol=1e-6
    @test LowerTriangular(td.L) ≈ LowerTriangular(cluster.td.L) atol=1e-6

    @test multi_t_logpdf(td, X) ≈ multi_t_logpdf(cluster.td, X) atol=1e-6
end
