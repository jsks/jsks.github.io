data {
  int K;
  array[K] int n_obs;
  array[K] int y;
}

parameters {
  real<lower=0> lambda;
}

model {
  lambda ~ gamma(25, 1);
  for (i in 1:K)
      target += n_obs[i] * poisson_lpmf(y[i] | lambda);
}
