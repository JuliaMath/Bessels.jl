#                           Modified Spherical Bessel functions
#
#                                 sphericalbesselk(nu, x)
#
#    A numerical routine to compute the modified spherical bessel functions of the second kind.
#    For moderate sized integer orders, forward recurrence is used starting from explicit formulas for k0(x) = exp(-x) / x  and k1(x) = k0(x) * (x+1) / x [1].
#    Large orders are determined from the uniform asymptotic expansions (see src/besselk.jl for details)
#    For non-integer orders, we directly call the besselk routine using the relation k_{n}(x) = sqrt(pi/(2x))*besselk(n+1/2, x) [1].
#    
# [1] https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheSecondKind.html
#
"""
    sphericalbesselk(nu, x::T) where T <: {Float32, Float64}

Computes `k_{ν}(x)`, the modified second-kind spherical Bessel function, and offers special branches for integer orders.
"""
sphericalbesselk(nu::Real, x::Real) = _sphericalbesselk(nu, float(x))

_sphericalbesselk(nu, x::Float16) = Float16(_sphericalbesselk(nu, Float32(x)))

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
        return inv(SQPIO2(T)*sqrt(x))*besselk(nu+1/2, x)
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
