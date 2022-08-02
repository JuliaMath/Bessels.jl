#    Bessel functions of the first kind of order zero and one
#                       besselj0, besselj1
#
#    Calculation of besselj0 is done in three branches using polynomial approximations
#
#    Branch 1: x <= pi/2
#              besselj0 is calculated using a 9 term, even minimax polynomial
#
#    Branch 2: pi/2 < x < 26.0
#              besselj0 is calculated by one of 16 different degree 13 minimax polynomials
#       Each polynomial is an expansion around either a root or extrema of the besselj0.
#       This ensures accuracy near the roots. Method taken from [2]
#
#   Branch 3: x >= 26.0
#              besselj0 = sqrt(2/(pi*x))*beta(x)*(cos(x - pi/4 - alpha(x))
#   See modified expansions given in [2]. Exact coefficients are used.
#
#   Calculation of besselj1 is done in a similar way as besselj0.
#   See [2] for details on similarities.
#
# [1] https://github.com/deepmind/torch-cephes
# [2] Harrison, John. "Fast and accurate Bessel function computation."
#     2009 19th IEEE Symposium on Computer Arithmetic. IEEE, 2009.
#

"""
    besselj0(x::T) where T <: Union{Float32, Float64}

Bessel function of the first kind of order zero, ``J_0(x)``.
"""
function besselj0(x::Float64)
    T = Float64
    x = abs(x)

    if x < 26.0
        x < pi/2 && return evalpoly(x * x, J0_POLY_PIO2(T))
        n = unsafe_trunc(Int, TWOOPI(T) * x)
        root = @inbounds J0_ROOTS(T)[n]
        r = x - root[1] - root[2]
        return evalpoly(r, @inbounds J0_POLYS(T)[n])
    else
        xinv = inv(x)
        iszero(xinv) && return zero(T)
        x2 = xinv * xinv

        if x < 120.0
            p1 = (one(T), -1/16, 53/512, -4447/8192, 3066403/524288, -896631415/8388608, 796754802993/268435456, -500528959023471/4294967296)
            q1 = (-1/8, 25/384, -1073/5120, 375733/229376, -55384775/2359296, 24713030909/46137344, -7780757249041/436207616)
            p = evalpoly(x2, p1)
            q = evalpoly(x2, q1)
        else
            p2 = (one(T), -1/16, 53/512, -4447/8192)
            q2 = (-1/8, 25/384, -1073/5120, 375733/229376)
            p = evalpoly(x2, p2)
            q = evalpoly(x2, q2)
        end

        a = SQ2OPI(T) * sqrt(xinv) * p
        xn = muladd(xinv, q, -PIO4(T))

        # the following computes b = cos(x + xn) more accurately
        # see src/misc.jl
        b = cos_sum(x, xn)
        return a * b
    end
end
function besselj0(x::Float32)
    T = Float32
    x = abs(x)

    if x <= 2.0f0
        z = x * x
        if x < 1.0f-3
            return 1.0f0 - 0.25f0 * z
        end
        DR1 = 5.78318596294678452118f0
        p = (z - DR1) * evalpoly(z, JP_j0(T))
        return p
    else
        q = inv(x)
        iszero(q) && return zero(T)
        w = sqrt(q)
        p = w * evalpoly(q, MO_j0(T))
        w = q * q
        xn = q * evalpoly(w, PH_j0(T)) - PIO4(Float32)
        p = p * cos(xn + x)
        return p
    end
end

"""
    besselj1(x::T) where T <: Union{Float32, Float64}

Bessel function of the first kind of order one, ``J_1(x)``.
"""
function besselj1(x::Float64)
    T = Float64
    s = sign(x)
    x = abs(x)

    if x <= 26.0
        x <= pi/2 && return x * evalpoly(x * x, J1_POLY_PIO2(T))
        n = unsafe_trunc(Int, TWOOPI(T) * x)
        root = @inbounds J1_ROOTS(T)[n]
        r = x - root[1] - root[2]
        return evalpoly(r, @inbounds J1_POLYS(T)[n]) * s
    else
        xinv = inv(x)
        iszero(xinv) && return zero(T)
        x2 = xinv * xinv
        if x < 120.0
            p1 = (one(T), 3/16, -99/512, 6597/8192, -4057965/524288, 1113686901/8388608, -951148335159/268435456, 581513783771781/4294967296)
            q1 = (3/8, -21/128, 1899/5120, -543483/229376, 8027901/262144, -30413055339/46137344, 9228545313147/436207616)
            p = evalpoly(x2, p1)
            q = evalpoly(x2, q1)
        else
            p2 = (one(T), 3/16, -99/512, 6597/8192)
            q2 = (3/8, -21/128, 1899/5120, -543483/229376)
            p = evalpoly(x2, p2)
            q = evalpoly(x2, q2)
        end

        a = SQ2OPI(T) * sqrt(xinv) * p
        xn = muladd(xinv, q, -3 * PIO4(T))

        # the following computes b = cos(x + xn) more accurately
        # see src/misc.jl
        b = cos_sum(x, xn)
        return a * b * s
    end
end
function besselj1(x::Float32)
    T = Float32
    s = sign(x)
    x = abs(x)

    if x <= 2.0f0
        z = x * x
        Z1 = 1.46819706421238932572f1
        p = (z - Z1) * x * evalpoly(z, JP32)
        return p * s
    else
        q = inv(x)
        iszero(q) && return zero(T)
        w = sqrt(q)
        p = w * evalpoly(q, MO132)
        w = q * q
        xn = q * evalpoly(w, PH132) - THPIO4(T)
        p = p * cos(xn + x)
        return p * s
    end
end

function besselj(nu::Real, x::T) where T
    isinteger(nu) && return besselj(Int(nu), x)
    abs_nu = abs(nu)
    abs_x = abs(x)

    Jnu = besselj_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        return x >= zero(T) ? Jnu : Jnu * cispi(abs_nu)
    else
        Ynu = bessely_positive_args(abs_nu, abs_x)
        spi, cpi = sincospi(abs_nu)
        out = Jnu * cpi - Ynu * spi
        return x >= zero(T) ? out : out * cispi(nu)
    end
end

function besselj(nu::Integer, x::T) where T
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    Jnu = besselj_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        return x >= zero(T) ? Jnu : Jnu * sg
    else
        if x >= zero(T)
            return Jnu * sg
        else
            Ynu = bessely_positive_args(abs_nu, abs_x)
            spi, cpi = sincospi(abs_nu)
            return (cpi*Jnu - spi*Ynu) * sg
        end
    end
end

"""
    besselj_positive_args(nu, x::T) where T <: Float64

Bessel function of the first kind of order nu, ``J_{nu}(x)``.
nu and x must be real and nu and x must be positive.

No checks on arguments are performed and should only be called if certain nu, x >= 0.
"""
function besselj_positive_args(nu::Real, x::T) where T
    nu == 0 && return besselj0(x)
    nu == 1 && return besselj1(x)

    # x < ~nu branch see src/U_polynomials.jl
    besseljy_debye_cutoff(nu, x) && return besseljy_debye(nu, x)[1]

    # large argument branch see src/asymptotics.jl
    besseljy_large_argument_cutoff(nu, x) && return besseljy_large_argument(nu, x)[1]

    # x > ~nu branch see src/U_polynomials.jl on computing Hankel function
    hankel_debye_cutoff(nu, x) && return real(hankel_debye(nu, x))

    # use power series for small x and for when nu > x
    besselj_series_cutoff(nu, x) && return besselj_power_series(nu, x)

    # At this point we must fill the region when x ≈ v with recurrence
    # Backward recurrence is always stable and forward recurrence is stable when x > nu
    # However, we only use backward recurrence by shifting the order up and using `besseljy_debye` to generate start values
    # Both `besseljy_debye` and `hankel_debye` get more accurate for large orders,
    # however `besseljy_debye` is slightly more efficient (no complex variables) and we need no branches if only consider one direction.
    # On the other hand, shifting the order down avoids any concern about underflow for large orders
    # Shifting the order too high while keeping x fixed could result in numerical underflow
    # Therefore we need to shift up only until the `besseljy_debye` is accurate and need to test that no underflow occurs
    # Shifting the order up decreases the value substantially for high orders and results in a stable forward recurrence
    # as the values rapidly increase

    debye_cutoff = 2.0 + 1.00035*x + Base.Math._approx_cbrt(302.681*Float64(x))
    nu_shift = ceil(Int, debye_cutoff - nu)
    v = nu + nu_shift
    jnu = besseljy_debye(v, x)[1]
    jnup1 = besseljy_debye(v+1, x)[1]
    return besselj_down_recurrence(x, jnu, jnup1, v, nu)[1]
end

#####
##### Power series for J_{nu}(x)
#####

# accurate for x < 7.0 or nu > 2+ 0.109x + 0.062x^2 for Float64
# accurate for x < 20.0 or nu > 14.4 - 0.455x + 0.027x^2 for Float32 (when using F64 precision)
# only valid in non-oscillatory regime (v>1/2, 0<t<sqrt(v^2 - 0.25))
# power series has premature underflow for large orders though use besseljy_debye for large orders
"""
    besselj_power_series(nu, x::T) where T <: Float64

Computes ``J_{nu}(x)`` using the power series.
In general, this is most accurate for small arguments and when nu > x.
"""
function besselj_power_series(v, x::T) where T
    MaxIter = 3000
    out = zero(T)
    a = (x/2)^v / gamma(v + one(T))
    t2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        abs(a) < eps(T) * abs(out) && break
        a *= -inv((v + i + one(T)) * (i + one(T))) * t2
    end
    return out
end

besselj_series_cutoff(v, x::Float64) = (x < 7.0) || v > (2 + x*(0.109 + 0.062x))
besselj_series_cutoff(v, x::Float32) = (x < 20.0) || v > (14.4 + x*(-0.455 + 0.027x))

#=
# this needs a better way to sum these as it produces large errors
# use when v is large and x is small
# though when v is large we should use the debye expansion instead
# also do not have a julia implementation of loggamma so will not use for now
function log_besselj_small_arguments_orders(v, x::T) where T
    MaxIter = 3000
    out = zero(T)
    a = one(T)
    x2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        a *= -x2 * inv((i + one(T)) * (v + i + one(T)))
        (abs(a) < eps(T) * abs(out)) && break
    end
    logout = -loggamma(v + 1) + fma(v, log(x/2), log(out))
    return exp(logout)
end
=#
