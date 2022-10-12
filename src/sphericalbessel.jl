#                           Spherical Bessel functions
#
#                sphericalbesselj(nu, x), sphericalbessely(nu, x)
#
#    A numerical routine to compute the spherical bessel functions of the first and second kind.
#    For small arguments, the power series series is used for sphericalbesselj. If nu is not a big integer
#    then forward recurrence is used if x >= nu. If x < nu, forward recurrence for sphericalbessely is used
#    and then a continued fraction and wronskian is used to compute sphericalbesselj [1]. For all other values,
#    we directly call besselj and bessely routines.
#    
# [1] Ratis, Yu L., and P. Fernández de Córdoba. "A code to calculate (high order) Bessel functions based on the continued fractions method." 
#     Computer physics communications 76.3 (1993): 381-388.
#

#####
##### Generic routine for `sphericalbesselj`
#####

"""
    sphericalbesselj(nu, x)

Spherical bessel function of the first kind of order `nu`, ``j_ν(x)``. This is the non-singular
solution to the radial part of the Helmholz equation in spherical coordinates.
"""
sphericalbesselj(nu::Real, x::Real) = _sphericalbesselj(nu, float(x))

_sphericalbesselj(nu, x::Float32) = Float32(_sphericalbesselj(nu, Float64(x)))

_sphericalbesselj(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_sphericalbesselj(Float32(nu), Float32(x)))

function _sphericalbesselj(nu::Real, x::T) where T <: Float64
    x < zero(T) && return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
    if ~isfinite(x)
        isnan(x) && return x
        isinf(x) && return zero(x)
    end
    return nu < zero(T) ? sphericalbesselj_generic(nu, x) : sphericalbesselj_positive_args(nu, x)
end

sphericalbesselj_generic(nu, x::T) where T = SQPIO2(T) * besselj(nu + one(T)/2, x) / sqrt(x)

#####
##### Positive arguments for `sphericalbesselj`
#####

function sphericalbesselj_positive_args(nu::Real, x::T) where T
    isinteger(nu) && return sphericalbesselj_positive_args(Int(nu), x)
    return sphericalbesselj_small_args_cutoff(nu, x) ? sphericalbesselj_small_args(nu, x) : sphericalbesselj_generic(nu, x)
end

function sphericalbesselj_positive_args(nu::Integer, x::T) where T
    sphericalbesselj_small_args_cutoff(nu, x) && return sphericalbesselj_small_args(nu, x)
    return (x >= nu && nu < 250) || (x < nu && nu < 60) ? sphericalbesselj_recurrence(nu, x) : sphericalbesselj_generic(nu, x)
end

#####
##### Power series expansion for small arguments
#####

sphericalbesselj_small_args_cutoff(nu, x::T) where T = x^2 / (4*nu + 110) < eps(T)

function sphericalbesselj_small_args(nu, x::T) where T
    iszero(x) && return iszero(nu) ? one(T) : zero(T)
    x2 = x^2 / 4
    coef = evalpoly(x2, (1, -inv(T(3)/2 + nu), -inv(5 + nu), -inv(T(21)/2 + nu), -inv(18 + nu)))
    a = SQPIO2(T) / (gamma(T(3)/2 + nu) * 2^(nu + T(1)/2))
    return x^nu * a * coef
end

#####
##### Integer recurrence and/or wronskian with continued fraction
#####

# very accurate approach however need to consider some performance issues
# if recurrence is stable (x>=nu) can generate very fast up to orders around 250
# for larger orders it is more efficient to use other expansions
# if (x<nu) we can use forward recurrence from sphericalbessely_recurrence and
# then use a continued fraction approach. However, for largish orders (>60) the
# continued fraction is slower converging and more efficient to use other methods
function sphericalbesselj_recurrence(nu::Integer, x::T) where T
    if x >= nu
        # forward recurrence if stable
        xinv = inv(x)
        s, c = sincos(x)
        sJ0 = s * xinv
        sJ1 = (sJ0 - c) * xinv

        nu_start = one(T)
        while nu_start < nu + 0.5
            sJ0, sJ1 = sJ1, muladd((2*nu_start + 1)*xinv, sJ1, -sJ0)
            nu_start += 1
        end
        return sJ0
    elseif x < nu
        # compute sphericalbessely with forward recurrence and use continued fraction
        sYnm1, sYn = sphericalbessely_forward_recurrence(nu, x)
        H = besselj_ratio_jnu_jnum1(nu + T(3)/2, x)
        return inv(x^2 * (H*sYnm1 - sYn))
    end
end

#####
##### Generic routine for `sphericalbessely`
#####

"""
    sphericalbessely(nu, x)

Spherical bessel function of the second kind at order `nu`, ``y_ν(x)``. This is the singular
solution to the radial part of the Helmholz equation in spherical coordinates. Sometimes
known as a spherical Neumann function.
"""
sphericalbessely(nu::Real, x::Real) = _sphericalbessely(nu, float(x))

_sphericalbessely(nu, x::Float32) = Float32(_sphericalbessely(nu, Float64(x)))

_sphericalbessely(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_sphericalbessely(Float32(nu), Float32(x)))

function _sphericalbessely(nu::Real, x::T) where T <: Float64
    x < zero(T) && return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
    if ~isfinite(x)
        isnan(x) && return x
        isinf(x) && return zero(x)
    end
    return nu < zero(T) ? sphericalbessely_generic(nu, x) : sphericalbessely_positive_args(nu, x)
end

sphericalbessely_generic(nu, x::T) where T = SQPIO2(T) * bessely(nu + one(T)/2, x) / sqrt(x)

#####
##### Positive arguments for `sphericalbesselj`
#####

sphericalbessely_positive_args(nu::Real, x) = isinteger(nu) ? sphericalbessely_positive_args(Int(nu), x) : sphericalbessely_generic(nu, x)

function sphericalbessely_positive_args(nu::Integer, x::T) where T
    if besseljy_debye_cutoff(nu, x)
        # for very large orders use expansion nu >> x to avoid overflow in recurrence
        return SQPIO2(T) * besseljy_debye(nu + one(T)/2, x)[2] / sqrt(x)
    elseif nu < 250
        return sphericalbessely_forward_recurrence(nu, x)[1]
    else
        return sphericalbessely_generic(nu, x)
    end
end

#####
##### Integer recurrence
#####

function sphericalbessely_forward_recurrence(nu::Integer, x::T) where T
    xinv = inv(x)
    s, c = sincos(x)
    sY0 = -c * xinv
    sY1 = xinv * (sY0 - s)

    nu_start = one(T)
    while nu_start < nu + 0.5
        sY0, sY1 = sY1, muladd((2*nu_start + 1)*xinv, sY1, -sY0)
        nu_start += 1
    end
    # need to check if NaN resulted during loop
    # this could happen if x is very small and nu is large which eventually results in overflow (-> -Inf)
    # NaN inputs were checked in top level function so if sY0 is NaN we should return -infinity
    return isnan(sY0) ? (-T(Inf), -T(Inf)) : (sY0, sY1)
end
