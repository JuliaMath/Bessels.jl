# Bessels.jl
[![Build Status](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/heltonmc/Bessels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/heltonmc/Bessels.jl)

A Julia implementation of Bessel's functions and modified Bessel's functions of the first and second kind. 

The library currently supports Bessel's function of the first and second kind for orders 0 and 1 and for any integer order for modified Bessel's functions for real arguments.

# Quick start

```julia
# add the package
pkg> add https://github.com/heltonmc/Bessels.jl.git

julia> using Bessels

# Bessel function of the first kind of order 0
julia> besselj0(1.0)
0.7651976865579665

julia> besselj0(1.0f0)
0.7651977f0
```

# Supported functions

- `besselj0(x)`
- `besselj1(x)`
- `bessely0(x)`
- `bessely1(x)`
- `besseli0(x)`
- `besseli1(x)`
- `besselk0(x)`
- `besselk1(x)`
- `besseli(nu, x)`
- `besselk(nu, x)`

Exponentially scaled versions of Modified Bessel's function can also be called with `besselix(nu, x)` and `besselkx(nu, x)`.

# Benchmarks

These results compare the median value from BenchmarkTools obtained on one machine for arguments between 0 and 100.

We compare the relative [speed](https://github.com/heltonmc/Bessels.jl/blob/master/benchmark/benchmark.jl) to implementations provided by [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl).

| function | `Float32` | `Float64`
| ------------- | ------------- | ------------- |
| besselj0  | 1.7x  | 3.1x
| besselj1  | 1.7x | 3.0x 
| bessely0  | 1.9x  | 2.7x
| bessely1  | 1.7x  | 2.7x
| besseli0  | 26x  | 13.2x
| besseli1  | 22x  | 13.9x
| besseli(20, x)  |   5.4x   | 2.1x  |
| besseli(120, x)  |   6x  | 4.5x  |
| besselk0  | 16x  | 12.7x
| besselk1  | 15x  | 14.3x
| besselk(20, x)  |    5.4x  | 5.7x  |
| besselk(120, x)  |   2.9x  | 3.4x  |

* SpecialFunctions.jl doesn't always preserve the correct input type so some of the calculations may be done in Float64. This might skew the benchmarks for `Bessels.jl`.

# Current Development Plans

- More robust implementations of `besselj(nu, x)` and `bessely(nu, x)`
- Support non-integer orders
- Support complex arguments

# Contributing 

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue](https://github.com/heltonmc/Bessels.jl/issues/new) for discussion on newer implementations, share papers, new features, or if you encounter any problems.
 
