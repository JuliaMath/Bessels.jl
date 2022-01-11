# Bessels.jl
[![Build Status](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/heltonmc/Bessels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/heltonmc/Bessels.jl)


Implementations of Bessel's functions `besselj0`, `besselj1`, `bessely0`, `bessely1`, `besseli0`, `besseli1`, `besselk0`, `besselk1` in Julia.
Most implementations are ported to Julia from the [Cephes](https://www.netlib.org/cephes/) math library  originally written by Stephen L. Moshier in the C programming language
which are (partly) used in [SciPy](https://docs.scipy.org/doc/scipy/reference/special.html#faster-versions-of-common-bessel-functions) and [GCC libquadmath](https://gcc.gnu.org/onlinedocs/libquadmath/).

# Quick start

Only implemented for real arguments so far.

```julia
using Bessels

# Bessel function of the first kind of order 0
julia> besselj0(1.0)
0.7651976865579665
julia> besselj0(1.0f0)
0.7651977f0

# Bessel function of the first kind of order 1
julia> besselj1(1.0)
0.44005058574493355
julia> besselj1(1.0f0)
0.4400506f0

# Bessel function of the second kind of order 0
julia> bessely0(1.0)
0.08825696421567697
julia> bessely0(1.0f0)
0.08825697f0

# Bessel function of the second kind of order 1
julia> bessely1(1.0)
-0.7812128213002888
julia> bessely1(1.0f0)
-0.7812128f0

# Modified Bessel function of the first kind of order 0
julia> besseli0(1.0)
1.2660658777520082
julia> besseli0(1.0f0)
1.266066f0

# Scaled modified Bessel function of the first kind of order 0
julia> besseli0x(1.0)
0.46575960759364043
julia> besseli0x(1.0f0)
0.46575963f0

# Scaled modified Bessel function of the first kind of order 1
julia> besseli1x(1.0)
0.2079104153497085
julia> besseli1x(1.0f0)
0.20791042f0

# Modified Bessel function of the second kind of order 0
julia> besselk0(1.0)
0.42102443824070823
julia> besselk0(1.0f0)
0.42102447f0

# Scaled modified Bessel function of the second kind of order 0
julia> besselk0x(1.0)
1.1444630798068947
julia> besselk0x(1.0f0)
1.1444632f0

# Modified Bessel function of the second kind of order 1
julia> besselk1(1.0)
0.6019072301972346
julia> besselk1(1.0f0)
0.6019073f0

# Scaled modified Bessel function of the second kind of order 1
julia> besselk1x(1.0)
1.636153486263258
julia> besselk1x(1.0f0)
1.6361537f0
```

# Benchmarks

```julia
# Bessels.jl
julia> @benchmark besselj0(x) setup=(x=100.0*rand())
BenchmarkTools.Trial: 10000 samples with 999 evaluations.
 Range (min … max):   7.465 ns … 28.946 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     25.860 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   24.915 ns ±  4.090 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
 Memory estimate: 0 bytes, allocs estimate: 0.
 
 # SpecialFunctions.jl
 julia> @benchmark besselj0(x) setup=(x=100.0*rand())
BenchmarkTools.Trial: 10000 samples with 999 evaluations.
 Range (min … max):   9.050 ns … 79.412 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     51.677 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   51.029 ns ±  5.653 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
 Memory estimate: 0 bytes, allocs estimate: 0.
 ```
 
 ```julia
 # Bessels.jl
 julia> @benchmark besselk1(x) setup=(x=100.0*rand())
BenchmarkTools.Trial: 10000 samples with 988 evaluations.
 Range (min … max):  47.781 ns … 97.672 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     47.950 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   48.495 ns ±  2.656 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
 Memory estimate: 0 bytes, allocs estimate: 0.
 
 # SpecialFunctions.jl
 julia> @benchmark besselk(1, x) setup=(x=10.0*rand())
BenchmarkTools.Trial: 10000 samples with 681 evaluations.
 Range (min … max):  184.962 ns …  2.122 μs  ┊ GC (min … max): 0.00% … 81.41%
 Time  (median):     377.081 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   388.243 ns ± 96.971 ns  ┊ GC (mean ± σ):  0.04% ±  0.81%
 Memory estimate: 16 bytes, allocs estimate: 1.
 ```

Comparing the relative speed (`SpecialFunctions.jl / Bessels.jl`) for a vector of values between 0 and 100.

## Float32

| function | Relative speed |
| ------------- | ------------- |
| besselj0  | 1.7x  |
| besselj1  | 1.7x |
| bessely0  | 1.6x  |
| bessely1  | 1.6x  |
| besseli0  | 24x  |
| besseli1  | 22x  |
| besselk0  | 13x  |
| besselk1  | 15x  |

* it looks like SpecialFunctions.jl doesn't always preserve the correct input type so some of the calculations may be done in Float64. This might skew the benchmarks for `Bessels.jl` as it should have all calculations done in the lower precision.

```julia
# SpecialFunctions.jl 
julia> @btime besselk(1, x) setup=(x=Float32(10.0*rand()))
  179.272 ns (1 allocation: 16 bytes)
0.021948646168061196 # notice incorrect Float64 return type

# Bessels.jl
julia> @btime besselk1(x) setup=(x=Float32(10.0*rand()))
  22.967 ns (0 allocations: 0 bytes)
0.0027777348f0 # notice correct Float32 return type
```

## Float64

| function | Relative speed |
| ------------- | ------------- |
| besselj0  | 1.7x  |
| besselj1  | 1.9x |
| bessely0  | 1.5x  |
| bessely1  | 1.5x  |
| besseli0  | 9x  |
| besseli1  | 7x  |
| besselk0  | 5x  |
| besselk1  | 5x  |

```julia
# SpecialFunctions.jl 
julia> @btime besselk(1, x) setup=(x=10.0*rand())
  184.726 ns (1 allocation: 16 bytes) # notice the small difference in Float32 and Float64 implementations
0.0007517428778913419

# Bessels.jl
julia> @btime besselk1(x) setup=(x=10.0*rand())
  47.824 ns (0 allocations: 0 bytes)
0.0057366790518002045
```
