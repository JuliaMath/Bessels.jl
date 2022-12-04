# Bessels.jl
[![Build Status](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliamath.github.io/Bessels.jl/stable)
[![Coverage](https://codecov.io/gh/heltonmc/Bessels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/heltonmc/Bessels.jl)

[![version](https://juliahub.com/docs/Bessels/version.svg)](https://juliahub.com/ui/Packages/Bessels/29L49)
[![deps](https://juliahub.com/docs/Bessels/deps.svg)](https://juliahub.com/ui/Packages/Bessels/29L49?t=2)
[![Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Bessels)](https://pkgs.genieframework.com?packages=Bessels)

Numerical routines for computing Bessel, Airy, and Hankel functions for real arguments. These routines are written in the Julia programming language and are self contained without any external dependencies.

The goal of the library is to provide high quality numerical implementations of Bessel functions with high accuracy without comprimising on computational time. In general, we try to match (and often exceed) the accuracy of other open source routines such as those provided by [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl). There are instances where we don't quite match that desired accuracy (within a digit or two) but in general will provide implementations that are 5-10x faster (see [benchmarks](https://github.com/JuliaMath/Bessels.jl#benchmarks)).

The library currently supports Bessel functions, modified Bessel functions, Hankel functions, spherical Bessel functions, and Airy functions of the first and second kind for positive real arguments and integer and noninteger orders. Negative arguments are also supported only if the return value is real. We plan to support complex arguments in the future. An unexported gamma function is also provided.

# Quick start

```julia
# add the package
pkg> add https://github.com/heltonmc/Bessels.jl.git

julia> using Bessels

julia> x = 12.3; nu = 1.3

julia> besselj(nu, x)
-0.2267581644816917
```

# Supported functions

### Bessel functions of the first kind

$$ J_{\nu}(x) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m!\Gamma(m+\nu+1)}(\frac{x}{2})^{2m+\nu} $$

Bessel functions of the first kind, denoted as $J_{\nu}(x)$, can be called with `besselj(nu, x)` where `nu` is the order of the Bessel function with argument `x`. Routines are also available for orders `0` and `1` which can be called with `besselj0(x)` and `besselj1(x)`.

```julia
julia> ν, x = 1.4, 12.3

# generic call for any order ν
julia> besselj(ν, x)
-0.22796228516266345

# ν = 0
julia> besselj0(x)
0.11079795030758544

# ν = 1
julia> besselj1(x)
-0.1942588480405914
```

### Bessel functions of the second kind

$$ Y_{\nu}(x) = \frac{J_{\nu}(x) \cos(\nu \pi) - J_{-\nu}(x)}{\sin(\nu \pi)} $$

Bessel functions of the second kind, denoted as $Y_{\nu}(x)$, can be called with `bessely(nu, x)`. Routines are also available for orders `0` and `1` which can be called with `bessely0(x)` and `bessely1(x)`.

```julia
julia> ν, x = 1.4, 12.3

# generic call for any order ν
julia> bessely(ν, x)
0.00911009829832235

# ν = 0
julia> bessely0(x)
-0.19859309463502633

# ν = 1
julia> bessely1(x)
-0.11894840329926631
```

### Modified Bessel functions of the first kind

$$ I_{\nu}(x) = \sum_{m=0}^{\infty} \frac{1}{m!\Gamma(m+\nu+1)}(\frac{x}{2})^{2m+\nu} $$

Modified Bessel functions of the first kind, denoted as $I_{\nu}(x)$, can be called with `besseli(nu, x)` where `nu` is the order of the Bessel function with argument `x`. Routines are also available for orders `0` and `1` which can be called with `besseli0(x)` and `besseli1(x)`. Exponentially scaled versions of these functions $I_{\nu}(x) \cdot e^{-x}$ are also provided which can be called with `besseli0x(x)`, `besseli1x(x)`, and `besselix(nu, x)`.

```julia
julia> ν, x = 1.4, 12.3

# generic call for any order v
julia> besseli(ν, x)
23242.698263113296

# exponentially scaled version
julia> besselix(ν, x)
0.10579482312624018

# ν = 0
julia> besseli0(x)
25257.48759692308
julia> besseli0x(x)
0.11496562932068803

# ν = 1
julia> besseli1(x)
24207.933018435186
julia> besseli1x(x)
0.11018832507935208
```

### Modified Bessel functions of the second kind

$$ K_{\nu}(x) = \frac{\pi}{2} \frac{I_{-\nu}(x) - I_{\nu}(x)}{\sin(\nu \pi)} $$

Modified Bessel functions of the second kind, denoted as $K_{\nu}(x)$, can be called with `besselk(nu, x)`. Routines are available for orders `0` and `1` which can be called with `besselk0(x)` and `besselk1(x)`. Exponentially scaled versions of these functions $K_{\nu}(x) \cdot e^{x}$ are also provided which can be called with `besselk0x(x)`, `besselk1x(x)`, and `besselkx(nu, x)`.

```julia
julia> ν, x = 1.4, 12.3

julia> besselk(ν, x)
1.739055243080153e-6

julia> besselk0(x)
1.6107849768886856e-6

julia> besselk1(x)
1.6750295538365835e-6
```
## Support for sequence of orders

We also provide support for `besselj(nu::M, x::T)`, `bessely(nu::M, x::T)`, `besseli(nu::M, x::T)`, `besselk(nu::M, x::T)`, `besseli(nu::M, x::T)`, `besselh(nu::M, k, x::T)` when `M` is some `AbstractRange` and `T` is some float.

```julia
julia> besselj(0:10, 1.0)
11-element Vector{Float64}:
 0.7651976865579666
 0.44005058574493355
 0.11490348493190049
 0.019563353982668407
 0.0024766389641099553
 0.00024975773021123444
 2.0938338002389273e-5
 1.5023258174368085e-6
 9.422344172604502e-8
 5.249250179911876e-9
 2.630615123687453e-10
```

In general, this provides a fast way to generate a sequence of Bessel functions for many orders.
```julia
julia> @btime besselj(0:100, 50.0)
  398.095 ns (1 allocation: 896 bytes)
```
This function will allocate so it is recommended that you calculate the Bessel functions at the top level of your function outside any hot loop. You can also call the mutating function on your preallocated vector `Bessels.besselj!(out, nu, x)`
```julia
a = zeros(10)
out = Bessels.besselj!(a, 1:10, 1.0)
```

### Support for negative arguments

Support is provided for negative arguments and orders only if the return value is real. A domain error will be thrown if the return value is complex. See https://github.com/heltonmc/Bessels.jl/issues/30 for more details.

```julia
julia> ν, x = 13.0, -1.0
julia> besseli(ν, x)
-1.9956316782072005e-14

julia> ν, x = -14.0, -9.9
julia> besseli(ν, x)
0.2892290867115618

julia> ν, x = 12.6, -3.0
julia> besseli(ν, x)
ERROR: DomainError with -3.0:
Complex result returned for real arguments. Complex arguments are currently not supported
Stacktrace:
 [1] _besseli(nu::Float64, x::Float64)
   @ Bessels ~/.julia/packages/Bessels/OBoYU/src/besseli.jl:181
 [2] besseli(nu::Float64, x::Float64)
   @ Bessels ~/.julia/packages/Bessels/OBoYU/src/besseli.jl:167
 [3] top-level scope
   @ REPL[62]:1
```
#### Gamma
We also provide an unexported gamma function for real arguments that can be called with `Bessels.gamma(x)`.

# Accuracy

We report the relative errors (`abs(1 - Bessels.f(x)/ArbNumerics.f(ArbFloat(x)))`) compared to `ArbNumerics.jl` when computed in a higher precision. The working precision was set to `setworkingprecision(ArbFloat, 500); setextrabits(128)` for the calculations in arbitrary precision. We generate a thousand random points for $x \in (0, 100)$ and compute the mean and maximum absolute relative errors.


| function | `mean` | `maximum`
| -------------  | ------------- | ------------- |
| besselj0(x)  | 3e-16   | 6e-14  |
| besselj1(x)  | 2e-15   | 7e-13  |
| besselj(5.0, x)  | 3e-14   | 2e-11  |
| besselj(12.8, x)  | 2e-14   | 2e-12  |
| besselj(111.6, x)  | 8e-15   | 4e-14  |
| bessely0(x)  | 2e-15   | 5e-13  |
| bessely1(x)  | 1e-15   | 2e-13  |
| bessely(4.0, x)  | 3e-15   | 2e-12   |
| bessely(6.92, x)  | 3e-14   | 5e-12   |
| bessely(113.92, x)  | 8e-15   | 8e-14   |
| besselk0(x)  | 1.2e-16   | 4e-16  |
| besselk1(x)  | 1.2e-16   | 5e-16  |
| besselk(14.0, x)  | 4e-15   | 3e-14  |
| besselk(27.32, x)  | 6e-15   | 3e-14  |
| besseli0(x)  | 1.5e-16   | 6e-16  |
| besseli1(x)  | 1.5e-16   | 5e-16  |
| besseli(9.0, x)  | 2e-16   | 2e-15  |
| besseli(92.12, x)  | 9e-15   | 7e-14  |
| Bessels.gamma(x)   | 1.3e-16  | 5e-16

In general the largest relative errors are observed near the zeros of Bessel functions for `besselj` and `bessely`. Accuracy might also be slightly worse for very large arguments when using `Float64` precision.

# Benchmarks

We give brief performance comparisons to the implementations provided by [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl). In general, special functions are computed with separate algorithms in different domains leading to computational time being dependent on argument. For these comparisons we show the relative speed increase for computing random values between `0` and `100` for `x` and order `nu`. In some ranges, performance may be significantly better while others will be more similar.

| function | `Float64`
| -------------  | ------------- |
| besselj0(x)  | 2.5x
| besselj(nu, x)  | 6x
| bessely0(x)  | 2.3x
| bessely(nu, x)  | 5x
| besseli0  | 10x
| besseli(nu, x)   | 7x  |
| besselk0  | 10x
| besselk(nu, x)   | 4x  |
| Bessels.gamma(x)   | 5x  |

Benchmarks were run using Julia Version 1.7.2 on an Apple M1 using Rosetta. 

# API

- `besselj0(x)`
- `besselj1(x)`
- `besselj(nu, x)`
- `bessely0(x)`
- `bessely1(x)`
- `bessely(nu, x)`
- `besseli0(x)`
- `besseli1(x)`
- `besseli(nu, x)`
- `besselk0(x)`
- `besselk1(x)`
- `besselk(nu, x)`
- `besselh(nu, k, x)`
- `hankelh1(nu, x)`
- `hankelh2(nu, x)`
- `sphericalbesselj(nu, x)`
- `sphericalbessely(nu, x)`
- `Bessels.sphericalbesseli(nu, x)`
- `Bessels.sphericalbesselk(nu, x)`
- `airyai(x)`
- `airyaiprime(x)`
- `airybi(x)`
- `airybiprime(x)`
- `Bessels.gamma(x)`

# Current Development Plans

- Support for higher precision `Double64`, `Float128`
- Support for complex arguments (`x` and `nu`)
- Support for derivatives with respect to argument and order

# Contributing 

Contributions are very welcome, as are feature requests, suggestions or general discussions. Please open an [issue](https://github.com/heltonmc/Bessels.jl/issues/new) for discussion on newer implementations, share papers, new features, or if you encounter any problems. Our goal is to provide high quality implementations of Bessel functions that match or exceed the accuracy of the implementatations provided by SpecialFunctions.jl. Please let us know if you encounter any accuracy or performance differences.
 
