data {
  int N;
  array[N] int y;
}

parameters {
  real<lower=0> lambda;
}

model {
  lambda ~ gamma(25, 1);
  y ~ poisson(lambda);
}
