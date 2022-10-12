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
besselj0(x::Real) = _besselj0(float(x))

_besselj0(x::Float16) = Float16(_besselj0(Float32(x)))

function _besselj0(x::Float64)
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
function _besselj0(x::Float32)
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
besselj1(x::Real) = _besselj1(float(x))

_besselj1(x::Float16) = Float16(_besselj1(Float32(x)))

function _besselj1(x::Float64)
    T = Float64
    s = sign(x)
    x = abs(x)

    if x <= 26.0
        x <= pi/2 && return x * evalpoly(x * x, J1_POLY_PIO2(T)) * s
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
function _besselj1(x::Float32)
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

#                  Bessel functions of the first kind of order nu
#                               besselj(nu, x)
#
#    A numerical routine to compute the Bessel function of the first kind J_{ν}(x) [1]
#    for real orders and arguments of positive or negative value. The routine is based on several
#    publications [2, 3, 4, 5] that calculate J_{ν}(x) for positive arguments and orders where
#    reflection identities are used to compute negative arguments and orders.
#
#    In particular, the reflectance identities for negative integer orders J_{-n}(x) = (-1)^n * J_{n}(x) (Eq. 9.1.5; [6])
#    and for negative noninteger orders J_{−ν}(x) = cos(πν) * J_{ν}(x) - sin(πν) * Y_{ν}(x) are used.
#    For negative arguments of integer order, J_{n}(-x) = (-1)^n * J_{n}(x) is used and for
#    noninteger orders, J_{ν}(−x) = exp(im*π*ν) * J_{ν}(x) is used. For negative orders and arguments the previous identities are combined.
#
#    The identities are computed by calling the `besselj_positive_args(nu, x)` function which computes J_{ν}(x)
#    for positive arguments and orders. When x < ν and ν being reasonably large, the debye asymptotic expansion (Eq. 32; [3]) is used `besseljy_debye(nu, x)`.
#    For large arguments x >> ν, the phase functions are used (Eq. 14 [4]) with `besseljy_large_argument(nu, x)`.
#    When x > ν and x being reasonably large, the Hankel function is calculated from the debye expansion (Eq. 29; [3]) with `hankel_debye(nu, x)`
#    and J_{n}(x) is calculated from the real part of the Hankel function. These expansions are not uniform so are not strictly used when the above inequalities are true, therefore, cutoffs
#    were determined depending on the desired accuracy. For large arguments x >> ν, the phase functions are used (Eq. 15 [4]) with `besseljy_large_argument(nu, x)`.
#    For small arguments, J_{ν}(x) is calculated from the power series (`bessely_power_series(nu, x`)
#
#    For values where the expansions for large arguments and orders are not valid, backward recurrence is employed after shifting the order up
#    to where `besseljy_debye` is accurate then using downward recurrence. In general, the routine will be the slowest when ν ≈ x as all methods struggle at this point.
#    
# [1] http://dlmf.nist.gov/10.2.E2
# [2] Bremer, James. "An algorithm for the rapid numerical evaluation of Bessel functions of real orders and arguments." 
#     Advances in Computational Mathematics 45.1 (2019): 173-211.
# [3] Matviyenko, Gregory. "On the evaluation of Bessel functions." 
#     Applied and Computational Harmonic Analysis 1.1 (1993): 116-135.
# [4] Heitman, Z., Bremer, J., Rokhlin, V., & Vioreanu, B. (2015). On the asymptotics of Bessel functions in the Fresnel regime. 
#     Applied and Computational Harmonic Analysis, 39(2), 347-356.
# [5] Ratis, Yu L., and P. Fernández de Córdoba. "A code to calculate (high order) Bessel functions based on the continued fractions method." 
#     Computer physics communications 76.3 (1993): 381-388.
# [6] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical functions with formulas, graphs, and mathematical tables. 
#     Vol. 55. US Government printing office, 1964.
#

#####
##### Generic routine for `besselj`
#####

"""
    besselj(nu, x::T) where T <: Union{Float32, Float64}

Bessel function of the first kind of order nu, ``J_{nu}(x)``.
nu and x must be real where nu and x can be positive or negative.
"""
besselj(nu, x::Real) = _besselj(nu, float(x))

_besselj(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_besselj(Float32(nu), Float32(x)))

function _besselj(nu::T, x::T) where T <: Union{Float32, Float64}
    isinteger(nu) && return _besselj(Int(nu), x)
    abs_nu = abs(nu)
    abs_x = abs(x)

    Jnu = besselj_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        if x >= zero(T)
            return T(Jnu)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Jnu * cispi(abs_nu)
        end
    else
        Ynu = bessely_positive_args(abs_nu, abs_x)
        spi, cpi = sincospi(abs_nu)
        out = Jnu * cpi - Ynu * spi
        if x >= zero(T)
            return T(out)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return out * cispi(nu)
        end
    end
end

function _besselj(nu::Integer, x::T) where T <: Union{Float32, Float64}
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    Jnu = besselj_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        return x >= zero(T) ? T(Jnu) : T(Jnu * sg)
    else
        if x >= zero(T)
            return T(Jnu * sg)
        else
            Ynu = bessely_positive_args(abs_nu, abs_x)
            spi, cpi = sincospi(abs_nu)
            return T((cpi*Jnu - spi*Ynu) * sg)
        end
    end
end

function _besselj(nu::AbstractRange, x::T) where T
    (nu[1] >= 0 && step(nu) == 1) || throw(ArgumentError("nu must be >= 0 with step(nu)=1"))
    len = length(nu)
    isone(len) && return [besselj(nu[1], x)]

    out = zeros(T, len)
    if nu[end] < x
        out[1], out[2] = _besselj(nu[1], x), _besselj(nu[2], x)
        return besselj_up_recurrence!(out, x, nu)
    else
        k = len
        jn = zero(T)
        while abs(jn) < floatmin(T)
            if besselj_underflow_check(nu[k], x)
                jn = zero(T)
            else
                jn = _besselj(nu[k], x)
            end
            out[k] = jn
            k -= 1
            k < 1 && break
        end
        if k > 1
            out[k] = _besselj(nu[k], x)
            tmp = @view out[1:k+1]
            besselj_down_recurrence!(tmp, x, nu[1:k+1])
            return out
        else
            return out
        end
    end
end

besselj_underflow_check(nu, x::T) where T = nu > 100 + T(1.01)*x + 85*Base.Math._approx_cbrt(x)

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

    # shift nu up and use downward recurrence
    return besselj_recurrence(nu, x)
end

#####
##### Power series for J_{nu}(x)
#####

# only valid in non-oscillatory regime (v>1/2, 0<t<sqrt(v^2 - 0.25))
# power series has premature underflow for large orders though use besseljy_debye for large orders
# computes in precision of promote_type(eltype(x), Float64) but will return eltype(x)
"""
    besselj_power_series(nu, x::T) where T <: Float64

Computes ``J_{nu}(x)`` using the power series.
In general, this is most accurate for small arguments and when nu > x.
"""
function besselj_power_series(v, x::T) where T
    MaxIter = 3000
    S = promote_type(T, Float64)
    v, x = S(v), S(x)

    out = zero(S)
    a = (x/2)^v / gamma(v + one(S))
    t2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        abs(a) < eps(T) * abs(out) && break
        a *= -inv((v + i + one(S)) * (i + one(S))) * t2
    end
    return T(out)
end

# Cutoff where power series can provide good accuracy
# Cutoff for Float32 determined from using Float64 precision down to eps(Float32)
besselj_series_cutoff(v, x::Float32) = (x < 20.0) || v > (14.4 + x*(-0.455 + 0.027*x))
besselj_series_cutoff(v, x::Float64) = (x < 7.0) || v > (2 + x*(0.109 + 0.062*x))
# cutoff for Float128 for ~1e-35 relative error 
#besselj_series_cutoff(v, x::AbstractFloat) = (x < 4.0) || v > (x*(0.08 + 0.12*x))

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

#####
##### Recurrence J_{nu}(x)
#####

# At this point we must fill the region when x ≈ v with recurrence
# Backward recurrence is always stable and forward recurrence is stable when x > nu
# However, we only use backward recurrence by shifting the order up and using `besseljy_debye` to generate start values
# Both `besseljy_debye` and `hankel_debye` get more accurate for large orders,
# however `besseljy_debye` is slightly more efficient (no complex variables) and we need no branches if only consider one direction.
# On the other hand, shifting the order down avoids any concern about underflow for large orders
# Shifting the order too high while keeping x fixed could result in numerical underflow
# Therefore we need to shift up only until the `besseljy_debye` is accurate and need to test that no underflow occurs
# Shifting the order up decreases the value substantially for high orders and results 
# in a stable forward recurrence as the values rapidly increase
function besselj_recurrence(nu, x)
    # shift order up to where expansions are valid see src/U_polynomials.jl
    debye_cutoff = ceil(besseljy_debye_fit(x))
    nu_shift = ceil(Int, debye_cutoff - floor(nu))
    v = nu + nu_shift
    # compute jnu and jnup1 then use downard recurrence starting from order v down to nu see src/recurrence.jl
    jnu = besseljy_debye(v, x)[1]
    jnup1 = besseljy_debye(v+1, x)[1]
    return besselj_down_recurrence(x, jnu, jnup1, v, nu)[1]
end
