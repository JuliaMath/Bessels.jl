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

### Bessel Functions of the first kind

$$ J_{\nu} = \sum_{m=0}^{\infty} \frac{(-1)^m}{m!\Gamma(m+\nu+1)}(\frac{x}{2})^{2m+\nu}    $$

Bessel functions of the first kind, denoted as $J_{\nu}(x)$, can be called with `besselj(nu, x)` where `nu` is the order of the Bessel function with argument `x`. Routines are also available for orders `0` and `1` which can be called with `besselj0(x)` and `besselj1`.

```julia
julia> v, x = 1.4, 12.3

# generic call for any order v
julia> besselj(v, x)
-0.22796228516266664

# v = 0
julia> besselj0(x)
0.11079795030758544

# v = 1
julia> besselj1(x)
-0.1942588480405914
```

Support is also provided for negative argument and orders.

```julia
julia> besselj(-1.4, 12.1)
0.1208208567052962

julia> besselj(5.8, -12.1)
-0.17685468273930427 + 0.12849244828700035im

julia> besselj(-2.8, -12.1)
0.1785554860917274 + 0.1297281542543098im
```

### Bessel Functions of the second kind

$$ Y_{\nu} = \frac{J_{\nu} \cos(\nu \pi) - J_{-\nu}}{\sin(\nu \pi)}    $$

Bessel functions of the second kind, denoted as $Y_{\nu}(x)$, can be called with `bessely(nu, x)`. Routines are also available for orders `0` and `1` which can be called with `bessely0(x)` and `bessely1`.

```julia
julia> v, x = 1.4, 12.3

# generic call for any order v
julia> bessely(v, x)
0.00911009829832235

# v = 0
julia> bessely0(x)
-0.19859309463502633

# v = 1
julia> bessely1(x)
-0.11894840329926633
```

Support is also provided for negative argument and orders.

```julia
julia> bessely(-1.4, 12.1)
0.19576089434668542

julia> bessely(5.8, -12.1)
-0.08847808383435121 - 0.41799245618068837im

julia> bessely(-2.8, -12.1)
0.05892453605716959 + 0.31429979079493564im
```

### Modified Bessel functions of the first kind

$$ I_{\nu} = \sum_{m=0}^{\infty} \frac{1}{m!\Gamma(m+\nu+1)}(\frac{x}{2})^{2m+\nu}    $$

Modified Bessel functions of the first kind, denoted as $I_{\nu}(x)$, can be called with `besseli(nu, x)` where `nu` is the order of the Bessel function with argument `x`. Routines are also available for orders `0` and `1` which can be called with `besseli0(x)` and `besseli1`. Exponentially scaled versions of these functions $I_{\nu}(x) * e^{-x}$ are also provided which can be called with `besseli0x(nu, x)`, `besseli1x(nu, x)`, and `besselix(nu, x)`.

```julia
julia> v, x = 1.4, 12.3

# generic call for any order v
julia> besseli(v, x)
23781.28963619158
julia> besselix(v, x)
0.10824635342651369

# v = 0
julia> besseli0(x)
25257.48759692308
julia> besseli0x(x)
0.11496562932068803

# v = 1
julia> besseli1(x)
24207.933018435186

julia> besseli1x(x)
0.11018832507935208
```

Support is also provided for negative argument and orders.

```julia
julia> besseli(-1.4, 12.1)
19162.123079744106

julia> besseli(5.8, -12.1)
4072.854035672388 - 2959.1016672742567im

julia> besseli(-2.8, -12.1)
-12045.563277269033 - 8751.613994715595im
```

### Modified Bessel Functions of the second kind

$$ K_{\nu} = \frac{\pi}{2} \frac{I_{-\nu} - I_{\nu}}{\sin(\nu \pi)}    $$

Modified Bessel functions of the second kind, denoted as $K_{\nu}(x)$, can be called with `besselk(nu, x)`. Routines are available for orders `0` and `1` which can be called with `besselk0(x)` and `besselk1`. Exponentially scaled versions of these functions $I_{\nu}(x) * e^{-x}$ are also provided which can be called with `besselk0x(nu, x)`, `besselk1x(nu, x)`, and `besselkx(nu, x)`.

```julia
julia> v, x = 1.4, 12.3
(1.4, 12.3)

julia> besselk(v, x)
1.739055243080153e-6

julia> besselk0(x)
1.6107849768886856e-6

julia> besselk1(x)
1.6750295538365835e-6
```

Support is also provided for negative argument and orders.

```julia
julia> besselk(-1.4, 12.1)
2.1438456178173037e-6

julia> besselk(5.8, -12.1)
5.988301577988307e-6 - 15815.796705207398im

julia> besselk(-2.8, -12.1)
-2.188744151278995e-6 - 46775.597252033134im
```

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
 
