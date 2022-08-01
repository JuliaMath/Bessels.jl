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

function besselj(nu::Int, x::T) where T
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(Int(abs_nu)) ? 1 : -1

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

function besselj_positive_args(nu::Real, x::T) where T
    nu == 0 && return besselj0(x)
    nu == 1 && return besselj1(x)

    x < 4.0 && return besselj_small_arguments_orders(nu, x)

    large_arg_cutoff = 1.65*nu
    (x > large_arg_cutoff && x > 20.0) && return besseljy_large_argument(nu, x)[1]


    debye_cutoff = 2.0 + 1.00035*x + Base.Math._approx_cbrt(302.681*Float64(x))
    nu > debye_cutoff && return besseljy_debye(nu, x)[1]

    if nu >= x
        nu_shift = ceil(Int, debye_cutoff - nu)
        v = nu + nu_shift
        jnu = besseljy_debye(v, x)[1]
        jnup1 = besseljy_debye(v+1, x)[1]
        return besselj_down_recurrence(x, jnu, jnup1, v, nu)[1]
    end

    # at this point x > nu and  x < nu * 1.65
    # in this region forward recurrence is stable
    # we must decide if we should do backward recurrence if we are closer to debye accuracy
    # or if we should do forward recurrence if we are closer to large argument expansion
    debye_cutoff = 5.0 + 1.00033*x + Base.Math._approx_cbrt(1427.61*Float64(x))

    debye_diff = debye_cutoff - nu
    large_arg_diff = nu - x / 2.0

    if (debye_diff > large_arg_diff && x > 20.0)
        nu_shift = ceil(Int, large_arg_diff)
        v2 = nu - nu_shift
        jnu = besseljy_large_argument(v2, x)[1]
        jnum1 = besseljy_large_argument(v2 - 1, x)[1]
        return besselj_up_recurrence(x, jnu, jnum1, v2, nu)[1]
    else
        nu_shift = ceil(Int, debye_diff)
        v = nu + nu_shift
        jnu = besseljy_debye(v, x)[1]
        jnup1 = besseljy_debye(v+1, x)[1]
        return besselj_down_recurrence(x, jnu, jnup1, v, nu)[1]
    end
end

# generally can only use for x < 4.0
# this needs a better way to sum these as it produces large errors
# only valid in non-oscillatory regime (v>1/2, 0<t<sqrt(v^2 - 0.25))
# power series has premature underflow for large orders
function besselj_small_arguments_orders(v, x::T) where T
    v > 60 && return log_besselj_small_arguments_orders(v, x)

    MaxIter = 2000
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

# this needs a better way to sum these as it produces large errors
# use when v is large and x is small
# need for bessely 
function log_besselj_small_arguments_orders(v, x::T) where T
    MaxIter = 2000
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
