# Bessels.jl
[![Build Status](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/heltonmc/Bessels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/heltonmc/Bessels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/heltonmc/Bessels.jl)

Numerical routines for computing Bessel functions and modified Bessel functions of the first and second kind. These routines are written entirely in the Julia programming language and are self contained without relying on any external dependencies.

The goal of the library is to provide high quality numerical implementations of Bessel functions with high accuracy without comprimising on computational time. In general, we try to match (and often exceed) the accuracy of other open source routines such as those provided by [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl). There are instances where we don't quite match that desired accuracy (within a digit or two) but in general will provide implementations that are 5-10x faster.

The library currently only supports Bessel functions and modified Bessel functions of the first and second kind for negative or positive real arguments and integer and noninteger orders. We plan to support complex arguments in the future. An unexported gamma function is also provided.

# Quick start

```julia
# add the package
pkg> add https://github.com/heltonmc/Bessels.jl.git

julia> using Bessels

julia> x = 12.3; nu = 1.3

julia> besselj(nu, x)
-0.2267581644816903
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

#### Gamma
We also provide an unexported gamma function for real arguments that can be called with `Bessels.gamma(x)`.

# Benchmarks

We give brief performance comparisons to the implementations provided by [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl). In general, special functions are computed with separate algorithms in different domains leading to computational time being dependent on argument. For these comparisons we show the relative speed increase for computing random values between `0` and `100` for `x` and order `nu`. In some ranges, performance may be significantly better while others more similar.

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
- `Bessels.gamma(x)`

# Current Development Plans

- Support for higher precision `Double64`, `Float128`
- Support for complex arguments (`x` and `nu`)
- Airy functions
- Support for derivatives with respect to argument and order

# Contributing 

Contributions are very welcome, as are feature requests, suggestions or general discussions. Please open an [issue](https://github.com/heltonmc/Bessels.jl/issues/new) for discussion on newer implementations, share papers, new features, or if you encounter any problems. Our goal is to provide high quality implementations of Bessel functions that match or exceed the accuracy of the implementatations provided by SpecialFunctions.jl. Please let us know if you encounter any accuracy or performance differences.
 
