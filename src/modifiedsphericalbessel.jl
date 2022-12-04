#                           Modified Spherical Bessel functions
#
#                                 sphericalbesseli(nu, x), sphericalbesselk(nu, x)
#
#    A numerical routine to compute the modified spherical bessel functions of the first and second kind.
#    The modified spherical bessel function of the first kind is computed using the power series for small arguments,
#    explicit formulas for (nu=0,1,2), and using its relation to besseli for other arguments [1].
#    The modified bessel function of the second kind is computed for small to moderate integer orders using forward recurrence starting from explicit formulas for k0(x) = exp(-x) / x  and k1(x) = k0(x) * (x+1) / x [2].
#    Large orders are determined from the uniform asymptotic expansions (see src/besselk.jl for details)
#    For non-integer orders, we directly call the besselk routine using the relation k_{n}(x) = sqrt(pi/(2x))*besselk(n+1/2, x) [2].
#   
# [1] https://mathworld.wolfram.com/ModifiedBesselFunctionoftheFirstKind.html 
# [2] https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheSecondKind.html
#

"""
    Bessels.sphericalbesselk(ν, x)

Returns the modified spherical bessel function of the second kind of order `ν`, ``k_ν(x)``.

```math
k_{\\nu}(x) = \\sqrt(\\frac{\\pi}{2x}) K_{\\nu + 1/2}(x)
```

where ``K_{\\nu}`` is the modified Bessel function of the second kind. Routine supports single and double precision (e.g., `Float32`,  `Float64`) for real arguments.

# Examples

```
julia> Bessels.sphericalbesselk(1, 1.2)
0.46015782375475334

julia> Bessels.sphericalbesselk(1.5, 1.2)
0.7595606983594666
```

External links: [DLMF](https://dlmf.nist.gov/10.47.9), [Wikipedia](https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1)

See also: [`besselk`](@ref), [`Bessels.sphericalbesseli`](@ref)
"""
sphericalbesselk(nu::Real, x::Real) = _sphericalbesselk(nu, float(x))

_sphericalbesselk(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_sphericalbesselk(Float32(nu), Float32(x)))

function _sphericalbesselk(nu, x::T) where T <: Union{Float32, Float64}
    if ~isfinite(x)
        isnan(x) && return x
        isinf(x) && return zero(x)
    end
    if isinteger(nu) && sphericalbesselk_cutoff(nu)
        if x < zero(x)
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
        end
        # using ifelse here to cut out a branch on nu < 0 or not.
        # The symmetry here is that
        # k_{-n} = (...)*K_{-n     + 1/2}
        #        = (...)*K_{|n|    - 1/2}
        #        = (...)*K_{|n|-1  + 1/2}
        #        = k_{|n|-1}
        _nu = ifelse(nu<zero(nu), -one(nu)-nu, nu)
        return sphericalbesselk_int(Int(_nu), x)
    else
        return inv(SQPIO2(T)*sqrt(x))*besselk(nu+T(1)/2, x)
    end
end
sphericalbesselk_cutoff(nu) = nu < 41.5

function sphericalbesselk_int(v::Int, x)
    xinv = inv(x)
    b0 = exp(-x) * xinv
    b1 = b0 * (x + one(x)) * xinv
    iszero(v) && return b0
    _v = one(v)
    invx = inv(x)
    while _v < v
        _v += one(_v)
        b0, b1 = b1, b0 + (2*_v - one(_v))*b1*invx
    end
    b1
end

"""
    sphericalbesseli(nu, x::T) where T <: {Float32, Float64}

Computes `i_{ν}(x)`, the modified first-kind spherical Bessel function.
"""
"""
    Bessels.sphericalbesseli(ν, x)

Returns the modified spherical bessel function of the first kind of order `ν`, ``i_ν(x)``.

```math
i_{\\nu}(x) = \\sqrt(\\frac{\\pi}{2x}) I_{\\nu + 1/2}(x)
```

where ``I_{\\nu}`` is the modified Bessel function of the first kind. Routine supports single and double precision (e.g., `Float32`,  `Float64`) for real arguments.

# Examples

```
julia> Bessels.sphericalbesseli(1, 1.2)
0.4606425870674146

julia> Bessels.sphericalbesseli(1.5, 1.2)
0.23179256391926525
```

External links: [DLMF](https://dlmf.nist.gov/10.47.7), [Wikipedia](https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1)

See also: [`besseli`](@ref), [`Bessels.sphericalbesselk`](@ref)
"""
sphericalbesseli(nu::Real, x::Real) = _sphericalbesseli(nu, float(x))

_sphericalbesseli(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_sphericalbesseli(Float32(nu), Float32(x)))

function _sphericalbesseli(nu, x::T) where T <: Union{Float32, Float64}
    isinf(x) && return x
    x < zero(x) && throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
   
    sphericalbesselj_small_args_cutoff(nu, x::T) && return sphericalbesseli_small_args(nu, x)
    isinteger(nu) && return _sphericalbesseli_small_orders(Int(nu), x)
    return SQPIO2(T)*besseli(nu+T(1)/2, x) / sqrt(x)
end

function _sphericalbesseli_small_orders(nu::Integer, x::T) where T
    # prone to cancellation in the subtraction
    # best to expand and group
    nu_abs = abs(nu)
    x2 = x*x
    sinhx = sinh(x)
    coshx = cosh(x)
    nu_abs == 0 && return sinhx / x
    nu_abs == 1 && return (x*coshx - sinhx) / x2
    nu_abs == 2 && return (x2*sinhx + 3*(sinhx - x*coshx)) / (x2*x)
    return SQPIO2(T)*besseli(nu+T(1)/2, x) / sqrt(x)
end

function sphericalbesseli_small_args(nu, x::T) where T
    iszero(x) && return iszero(nu) ? one(T) : x
    x2 = x^2 / 4
    coef = evalpoly(x2, (1, inv(T(3)/2 + nu), inv(5 + nu), inv(T(21)/2 + nu), inv(18 + nu)))
    a = SQPIO2(T) / (gamma(T(3)/2 + nu) * 2^(nu + T(1)/2))
    return x^nu * a * coef
end
