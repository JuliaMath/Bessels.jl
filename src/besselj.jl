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
"""
    besselj(nu, x::T) where T <: Union{Float32, Float64}

Bessel function of the first kind of order nu, ``J_{nu}(x)``.
Nu must be real.
"""
function _besselj(nu, x)
    nu == 0 && return besselj0(x)
    nu == 1 && return besselj1(x)

    x < 4.0 && return besselj_small_arguments_orders(nu, x)

    large_arg_cutoff = 1.65*nu
    (x > large_arg_cutoff && x > 20.0) && return besselj_large_argument(nu, x)


    debye_cutoff = 2.0 + 1.00035*x + (302.681*x)^(1/3)
    nu > debye_cutoff && return besselj_debye(nu, x)

    if nu >= x
        nu_shift = ceil(Int, 5.2 + 1.00033*x + (1427.61*x)^(1/3) - nu)
        v = nu + nu_shift
        arr = range(v, stop = nu, length = nu_shift + 1)
        jnu = besselj_debye(v, x)
        jnup1 = besselj_debye(v+1, x)
        return besselj_down_recurrence(x, jnu, jnup1, arr)[2]
    end

    # at this point x > nu and  x < nu * 1.65
    # in this region forward recurrence is stable
    # we must decide if we should do backward recurrence if we are closer to debye accuracy
    # or if we should do forward recurrence if we are closer to large argument expansion
    debye_cutoff = 5.0 + 1.00033*x + (1427.61*x)^(1/3)

    debye_diff = debye_cutoff - nu
    large_arg_diff = nu - x / 2.0

    if (debye_diff > large_arg_diff && x > 20.0)
        nu_shift = ceil(Int, large_arg_diff)
        v2 = nu - nu_shift
        jnu = besselj_large_argument(v2, x)
        jnum1 = besselj_large_argument(v2 - 1, x)
        return besselj_up_recurrence(x, jnu, jnum1, v2, nu)[2]
    else
        nu_shift = ceil(Int, debye_diff)
        v = nu + nu_shift
        arr = range(v, stop = nu, length = nu_shift + 1)
        jnu = besselj_debye(v, x)
        jnup1 = besselj_debye(v+1, x)
        return besselj_down_recurrence(x, jnu, jnup1, arr)[2]
    end
end

# for moderate size arguments of x and v this has relative errors ~9e-15
# for large arguments relative errors ~1e-13
function besselj_large_argument(v, x::T) where T
    α, αp = _α_αp_asymptotic(v, x)
    b = SQ2OPI(T) / sqrt(αp * x)

    S, C = sincos(PIO2(T)*v)
    Sα, Cα = sincos(α)
    s1 = (C - S) * Cα
    s2 = (C + S) * Sα

    return SQ2O2(T) * (s1 + s2) * b
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

# valid when x < v (uniform asymptotic expansions)
function besselj_debye(v, x)
    T = eltype(x)
    S = promote_type(T, Float64)
    x = S(x)

    vmx = (v + x) * (v - x)
    vs = sqrt(vmx)
    n  = muladd(v, -log(x / (v + vs)), -vs)

    coef = SQ1O2PI(S) * exp(-n) / sqrt(vs)
    p = v / vs
    p2  = v^2 / vmx

    return coef * Uk_poly_Jn(p, v, p2, x)
end

# For 0.0 <= x < 171.5
# Mean ULP = 0.55
# Max ULP = 2.4
# Adapted from Cephes Mathematical Library (MIT license https://en.smath.com/view/CephesMathLibrary/license) by Stephen L. Moshier
function gamma(x)
    if x > 11.5
        return large_gamma(x)
    elseif x < 0.0
        #p = floor(x)
        #isequal(p, abs(x)) && return throw(DomainError(x, "NaN result for non-NaN input."))
        # need reflection formula
        return throw(DomainError(x, "Negative numbers are currently not implemented"))
    elseif x <= 11.5
        return small_gamma(x)
    elseif isnan(x)
        return x
    end
end
function large_gamma(x)
    isinf(x) && return x
    T = Float64
    w = inv(x)
    s = (
        8.333333333333331800504e-2, 3.472222222230075327854e-3, -2.681327161876304418288e-3, -2.294719747873185405699e-4,
        7.840334842744753003862e-4, 6.989332260623193171870e-5, -5.950237554056330156018e-4, -2.363848809501759061727e-5,
        7.147391378143610789273e-4
    )
    w = w * evalpoly(w, s) + one(T)
    # lose precision on following block
    y = exp((x)) 
    # avoid overflow
    v = x^(0.5 * x - 0.25)
    y = v * (v / y)

    return SQ2PI(T) * y * w
end
function small_gamma(x)
    T = Float64
    P = (
        1.000000000000000000009e0, 8.378004301573126728826e-1, 3.629515436640239168939e-1, 1.113062816019361559013e-1,
        2.385363243461108252554e-2, 4.092666828394035500949e-3, 4.542931960608009155600e-4, 4.212760487471622013093e-5
    )
    Q = (
        9.999999999999999999908e-1, 4.150160950588455434583e-1, -2.243510905670329164562e-1, -4.633887671244534213831e-2,
        2.773706565840072979165e-2, -7.955933682494738320586e-4, -1.237799246653152231188e-3, 2.346584059160635244282e-4,
        -1.397148517476170440917e-5
    )

    z = one(T)
    while x >= 3.0
        x -= one(T)
        z *= x
    end
    while x < 2.0
        z /= x
        x += one(T)
    end

    x -= T(2)
    p = evalpoly(x, P)
    q = evalpoly(x, Q)
    return z * p / q
end
