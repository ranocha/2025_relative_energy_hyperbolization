# Numerical Experiments

This directory contains code to reproduce the numerical experiments described
in the manuscript.

This code is developed with Julia version 1.10.9. To reproduce the
results, start Julia in this directory and execute the following commands in
the Julia REPL to create the figures shown in the paper.

```julia
julia> include("code.jl")

julia> benjamin_bona_mahony_convergence()
  0.124158 seconds (122 allocations: 178.391 KiB)
  0.198803 seconds (9.95 k allocations: 18.820 MiB)
  0.215644 seconds (9.96 k allocations: 18.821 MiB, 1.67% gc time)
  0.212018 seconds (9.95 k allocations: 18.820 MiB)
  0.210132 seconds (9.95 k allocations: 18.820 MiB)
  0.210175 seconds (9.95 k allocations: 18.820 MiB)
  0.213002 seconds (9.97 k allocations: 18.821 MiB, 1.10% gc time)
  0.210528 seconds (9.95 k allocations: 18.820 MiB)
  0.206106 seconds (9.95 k allocations: 18.820 MiB)
  0.210032 seconds (9.95 k allocations: 18.820 MiB)
  0.208026 seconds (9.95 k allocations: 18.820 MiB)
  0.212702 seconds (9.97 k allocations: 18.821 MiB, 0.88% gc time)
  0.210496 seconds (9.95 k allocations: 18.820 MiB)
[ Info: Errors with respect to the numerical BBM solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   2.99e+00 │      NaN │
│ 1.00e-02 │   5.98e-01 │     0.70 │
│ 1.00e-03 │   1.85e-01 │     0.51 │
│ 1.00e-04 │   4.51e-02 │     0.61 │
│ 1.00e-05 │   5.55e-03 │     0.91 │
│ 1.00e-06 │   5.69e-04 │     0.99 │
│ 1.00e-07 │   5.70e-05 │     1.00 │
│ 1.00e-08 │   5.70e-06 │     1.00 │
│ 1.00e-09 │   5.70e-07 │     1.00 │
│ 1.00e-10 │   5.70e-08 │     1.00 │
│ 1.00e-11 │   5.68e-09 │     1.00 │
│ 1.00e-12 │   6.25e-10 │     0.96 │
└──────────┴────────────┴──────────┘

julia> korteweg_de_vries_convergence()
  0.300091 seconds (8.31 k allocations: 10.693 MiB)
  0.491029 seconds (24.04 k allocations: 32.472 MiB)
  0.499856 seconds (24.04 k allocations: 32.472 MiB)
  0.502270 seconds (24.05 k allocations: 32.472 MiB, 0.76% gc time)
  0.542212 seconds (24.04 k allocations: 32.481 MiB)
  0.556451 seconds (24.04 k allocations: 32.507 MiB)
  0.563421 seconds (24.04 k allocations: 32.507 MiB)
  0.560164 seconds (24.06 k allocations: 32.507 MiB, 0.48% gc time)
  0.555166 seconds (24.04 k allocations: 32.507 MiB)
  0.557523 seconds (24.04 k allocations: 32.507 MiB)
  0.566681 seconds (24.05 k allocations: 32.507 MiB, 0.55% gc time)
  0.556946 seconds (24.04 k allocations: 32.507 MiB)
  0.560264 seconds (24.04 k allocations: 32.507 MiB)
[ Info: Errors with respect to the numerical KdV solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   7.26e+00 │      NaN │
│ 1.00e-02 │   1.31e+00 │     0.74 │
│ 1.00e-03 │   1.29e-01 │     1.01 │
│ 1.00e-04 │   1.28e-02 │     1.00 │
│ 1.00e-05 │   1.28e-03 │     1.00 │
│ 1.00e-06 │   1.28e-04 │     1.00 │
│ 1.00e-07 │   1.28e-05 │     1.00 │
│ 1.00e-08 │   1.28e-06 │     1.00 │
│ 1.00e-09 │   1.28e-07 │     1.00 │
│ 1.00e-10 │   1.26e-08 │     1.01 │
│ 1.00e-11 │   1.01e-09 │     1.09 │
│ 1.00e-12 │   1.61e-10 │     0.80 │
└──────────┴────────────┴──────────┘

julia> korteweg_de_vries_burgers_convergence()
  0.179647 seconds (4.33 k allocations: 12.087 MiB)
  0.251279 seconds (20.04 k allocations: 32.662 MiB)
  0.258463 seconds (20.06 k allocations: 32.662 MiB, 2.43% gc time)
  0.252556 seconds (20.04 k allocations: 32.662 MiB)
  0.251340 seconds (20.04 k allocations: 32.662 MiB)
  0.255075 seconds (20.04 k allocations: 32.662 MiB)
  0.259854 seconds (20.07 k allocations: 32.662 MiB, 1.81% gc time)
  0.256254 seconds (20.04 k allocations: 32.662 MiB)
  0.254109 seconds (20.04 k allocations: 32.662 MiB)
  0.254969 seconds (20.06 k allocations: 32.662 MiB, 0.90% gc time)
  0.253129 seconds (20.04 k allocations: 32.662 MiB)
  0.254681 seconds (20.04 k allocations: 32.662 MiB)
  0.253774 seconds (20.04 k allocations: 32.662 MiB)
[ Info: Errors with respect to the numerical KdV-Burgers solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   3.74e-01 │      NaN │
│ 1.00e-02 │   3.79e-02 │     0.99 │
│ 1.00e-03 │   3.79e-03 │     1.00 │
│ 1.00e-04 │   3.79e-04 │     1.00 │
│ 1.00e-05 │   3.79e-05 │     1.00 │
│ 1.00e-06 │   3.79e-06 │     1.00 │
│ 1.00e-07 │   3.79e-07 │     1.00 │
│ 1.00e-08 │   3.79e-08 │     1.00 │
│ 1.00e-09 │   3.79e-09 │     1.00 │
│ 1.00e-10 │   3.76e-10 │     1.00 │
│ 1.00e-11 │   3.52e-11 │     1.03 │
│ 1.00e-12 │   3.38e-12 │     1.02 │
└──────────┴────────────┴──────────┘

julia> gardner_convergence()
tspan = (0.0, 83.33333333333334)
  1.167533 seconds (867.04 k allocations: 42.959 MiB)
  1.443493 seconds (629.48 k allocations: 40.614 MiB)
  1.440133 seconds (629.48 k allocations: 40.614 MiB)
  1.457383 seconds (629.50 k allocations: 40.615 MiB, 0.64% gc time)
  1.466320 seconds (629.48 k allocations: 40.614 MiB)
  1.435834 seconds (629.48 k allocations: 40.614 MiB)
  1.450428 seconds (629.50 k allocations: 40.615 MiB, 0.42% gc time)
  1.443170 seconds (629.48 k allocations: 40.614 MiB)
  1.451107 seconds (629.48 k allocations: 40.614 MiB)
  1.441170 seconds (629.48 k allocations: 40.614 MiB)
  1.454393 seconds (629.51 k allocations: 40.615 MiB, 0.39% gc time)
  1.452042 seconds (629.48 k allocations: 40.614 MiB)
  1.451014 seconds (629.48 k allocations: 40.614 MiB)
[ Info: Errors with respect to the numerical Gardner solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   3.84e+00 │      NaN │
│ 1.00e-02 │   2.57e+00 │     0.17 │
│ 1.00e-03 │   3.37e-01 │     0.88 │
│ 1.00e-04 │   3.32e-02 │     1.01 │
│ 1.00e-05 │   3.32e-03 │     1.00 │
│ 1.00e-06 │   3.32e-04 │     1.00 │
│ 1.00e-07 │   3.32e-05 │     1.00 │
│ 1.00e-08 │   3.32e-06 │     1.00 │
│ 1.00e-09 │   3.31e-07 │     1.00 │
│ 1.00e-10 │   3.26e-08 │     1.01 │
│ 1.00e-11 │   2.72e-09 │     1.08 │
│ 1.00e-12 │   2.70e-10 │     1.00 │
└──────────┴────────────┴──────────┘

julia> kawahara_convergence()
tspan = (0.0, 657.2222222222222)
  0.125015 seconds (26.61 k allocations: 1.619 MiB)
  0.297981 seconds (28.99 k allocations: 3.425 MiB)
  0.298844 seconds (28.99 k allocations: 3.425 MiB)
  0.298248 seconds (28.99 k allocations: 3.425 MiB)
  0.302537 seconds (28.99 k allocations: 3.714 MiB)
  0.387275 seconds (28.99 k allocations: 4.230 MiB)
  0.393962 seconds (28.99 k allocations: 4.164 MiB)
  0.387828 seconds (28.99 k allocations: 4.164 MiB)
  0.390558 seconds (28.99 k allocations: 4.164 MiB)
  0.386054 seconds (28.99 k allocations: 4.164 MiB)
  0.390323 seconds (28.99 k allocations: 4.164 MiB)
  0.396498 seconds (28.99 k allocations: 4.164 MiB)
  0.389815 seconds (28.99 k allocations: 4.164 MiB)
[ Info: Errors with respect to the numerical Kawahara solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   3.55e-01 │      NaN │
│ 1.00e-02 │   3.68e-02 │     0.98 │
│ 1.00e-03 │   4.02e-03 │     0.96 │
│ 1.00e-04 │   5.26e-04 │     0.88 │
│ 1.00e-05 │   5.39e-05 │     0.99 │
│ 1.00e-06 │   5.39e-06 │     1.00 │
│ 1.00e-07 │   5.39e-07 │     1.00 │
│ 1.00e-08 │   5.39e-08 │     1.00 │
│ 1.00e-09 │   5.39e-09 │     1.00 │
│ 1.00e-10 │   5.36e-10 │     1.00 │
│ 1.00e-11 │   5.09e-11 │     1.02 │
│ 1.00e-12 │   4.06e-12 │     1.10 │
└──────────┴────────────┴──────────┘

julia> generalized_kawahara_convergence()
tspan = (0.0, 715.909090909091)
  0.215225 seconds (28.96 k allocations: 2.686 MiB)
  0.492384 seconds (31.34 k allocations: 5.477 MiB)
  0.504287 seconds (31.34 k allocations: 5.477 MiB)
  0.497190 seconds (31.34 k allocations: 5.477 MiB)
  0.516425 seconds (31.35 k allocations: 6.597 MiB)
  0.920130 seconds (31.35 k allocations: 7.769 MiB)
  0.925924 seconds (31.35 k allocations: 8.915 MiB)
  0.918223 seconds (31.35 k allocations: 8.915 MiB)
  0.917341 seconds (31.35 k allocations: 8.915 MiB)
  0.921656 seconds (31.35 k allocations: 8.915 MiB)
  0.923719 seconds (31.35 k allocations: 8.915 MiB)
  0.922995 seconds (31.35 k allocations: 8.915 MiB)
  0.927638 seconds (31.35 k allocations: 8.915 MiB)
[ Info: Errors with respect to the numerical generalized Kawahara solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   2.65e+00 │      NaN │
│ 1.00e-02 │   3.40e-01 │     0.89 │
│ 1.00e-03 │   3.38e-02 │     1.00 │
│ 1.00e-04 │   3.45e-03 │     0.99 │
│ 1.00e-05 │   3.47e-04 │     1.00 │
│ 1.00e-06 │   3.47e-05 │     1.00 │
│ 1.00e-07 │   3.47e-06 │     1.00 │
│ 1.00e-08 │   3.47e-07 │     1.00 │
│ 1.00e-09 │   3.47e-08 │     1.00 │
│ 1.00e-10 │   3.42e-09 │     1.01 │
│ 1.00e-11 │   2.91e-10 │     1.07 │
│ 1.00e-12 │   2.49e-11 │     1.07 │
└──────────┴────────────┴──────────┘

julia> generalized_kawahara_error_growth()
tspan = (0.0, 7159.09090909091)
  8.363344 seconds (571.87 k allocations: 33.991 MiB, 0.13% gc time, 1.56% compilation time)
  8.317045 seconds (715.61 k allocations: 19.915 MiB, 0.07% compilation time)
 20.152425 seconds (640.71 k allocations: 50.246 MiB, 0.72% compilation time)
 20.207540 seconds (724.00 k allocations: 35.646 MiB, 0.01% compilation time)
136.334913 seconds (295.33 k allocations: 93.223 MiB)
138.492213 seconds (722.75 k allocations: 99.744 MiB, 0.00% gc time)
138.669300 seconds (295.34 k allocations: 125.319 MiB, 0.00% gc time)
138.584468 seconds (722.74 k allocations: 131.840 MiB)
139.368624 seconds (295.35 k allocations: 125.319 MiB, 0.00% gc time)
139.867927 seconds (722.75 k allocations: 131.840 MiB, 0.00% gc time)

julia> linear_biharmonic_convergence()
  0.001546 seconds (709 allocations: 315.172 KiB)
  0.003166 seconds (1.45 k allocations: 792.742 KiB)
  0.003108 seconds (1.45 k allocations: 792.742 KiB)
  0.002373 seconds (1.45 k allocations: 792.742 KiB)
  0.002430 seconds (1.45 k allocations: 792.742 KiB)
  0.002149 seconds (1.45 k allocations: 792.742 KiB)
  0.001918 seconds (1.45 k allocations: 792.742 KiB)
  0.002137 seconds (1.45 k allocations: 792.742 KiB)
  0.001826 seconds (1.45 k allocations: 792.742 KiB)
  0.001716 seconds (1.45 k allocations: 792.742 KiB)
  0.001751 seconds (1.45 k allocations: 792.742 KiB)
  0.001672 seconds (1.45 k allocations: 792.742 KiB)
  0.001519 seconds (1.45 k allocations: 792.742 KiB)
[ Info: Errors with respect to the numerical linear bi-harmomic solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   6.52e-02 │      NaN │
│ 1.00e-02 │   6.55e-03 │     1.00 │
│ 1.00e-03 │   6.52e-04 │     1.00 │
│ 1.00e-04 │   6.52e-05 │     1.00 │
│ 1.00e-05 │   6.52e-06 │     1.00 │
│ 1.00e-06 │   6.52e-07 │     1.00 │
│ 1.00e-07 │   6.52e-08 │     1.00 │
│ 1.00e-08 │   6.52e-09 │     1.00 │
│ 1.00e-09 │   6.52e-10 │     1.00 │
│ 1.00e-10 │   6.53e-11 │     1.00 │
│ 1.00e-11 │   6.64e-12 │     0.99 │
│ 1.00e-12 │   7.71e-13 │     0.93 │
└──────────┴────────────┴──────────┘

julia> kuramoto_sivashinsky_convergence()
  0.021833 seconds (1.13 k allocations: 3.742 MiB)
  0.028820 seconds (4.29 k allocations: 9.416 MiB)
  0.022843 seconds (4.29 k allocations: 9.416 MiB)
  0.020888 seconds (4.29 k allocations: 9.416 MiB)
  0.019844 seconds (4.29 k allocations: 9.808 MiB)
  0.022741 seconds (4.29 k allocations: 10.599 MiB)
  0.022888 seconds (4.29 k allocations: 10.599 MiB)
  0.022630 seconds (4.29 k allocations: 10.599 MiB)
  0.023306 seconds (4.29 k allocations: 10.599 MiB)
  0.023325 seconds (4.29 k allocations: 10.599 MiB)
  0.023048 seconds (4.29 k allocations: 10.599 MiB)
  0.029460 seconds (4.36 k allocations: 10.600 MiB, 18.90% gc time)
  0.023032 seconds (4.29 k allocations: 10.599 MiB)
[ Info: Errors with respect to the numerical Kuramoto-Sivashinsky solution
┌──────────┬────────────┬──────────┐
│  $\\tau$ │ L2 error u │ L2 EOC u │
├──────────┼────────────┼──────────┤
│ 1.00e-01 │   3.30e+00 │      NaN │
│ 1.00e-02 │   4.28e-01 │     0.89 │
│ 1.00e-03 │   4.38e-02 │     0.99 │
│ 1.00e-04 │   4.39e-03 │     1.00 │
│ 1.00e-05 │   4.39e-04 │     1.00 │
│ 1.00e-06 │   4.39e-05 │     1.00 │
│ 1.00e-07 │   4.39e-06 │     1.00 │
│ 1.00e-08 │   4.39e-07 │     1.00 │
│ 1.00e-09 │   4.39e-08 │     1.00 │
│ 1.00e-10 │   4.39e-09 │     1.00 │
│ 1.00e-11 │   4.42e-10 │     1.00 │
│ 1.00e-12 │   4.86e-11 │     0.96 │
└──────────┴────────────┴──────────┘
```

