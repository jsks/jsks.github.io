#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(cmdstanr))
options(mc.cores = parallel::detectCores())

# Force cmdstanr to not check the version using a locally installed
# copy of CmdStan
assignInNamespace("cmdstan_version", function(...) "2.34.1", ns = "cmdstanr")

data <- list(N = 20, y = rbinom(20, 1, 0.3))

mod <- cmdstan_model(exe_file = "bernoulli")
fit <- mod$sample(data = data, refresh = 0)

fit$summary()
