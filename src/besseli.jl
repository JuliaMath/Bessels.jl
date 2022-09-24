#    Modified Bessel functions of the first kind of order zero and one
#                       besseli0, besseli1
#
#    Scaled modified Bessel functions of the first kind of order zero and one
#                       besseli0x, besselix
#
#    (Scaled) Modified Bessel functions of the first kind of order nu
#                       besseli, besselix
#
#    Calculation of besseli0 is done in two branches using polynomial approximations [1]
#
#    Branch 1: x < 7.75 
#              besseli0 = [x/2]^2 P16([x/2]^2)
#    Branch 2: x >= 7.75
#              sqrt(x) * exp(-x) * besseli0(x) = P22(1/x)
#    where P16 and P22 are a 16 and 22 degree polynomial respectively.
#
#    Remez.jl is then used to approximate the polynomial coefficients of
#    P22(y) = sqrt(1/y) * exp(-inv(y)) * besseli0(inv(y))
#    N,D,E,X = ratfn_minimax(g, (1/1e6, 1/7.75), 21, 0)
#
#    A third branch is used for scaled functions for large values
#
#
#    Calculation of besseli1 is done in two branches using polynomial approximations [2]
#
#    Branch 1: x < 7.75 
#              besseli1 = x / 2 * (1 + 1/2 * (x/2)^2 + (x/2)^4 * P13([x/2]^2)
#    Branch 2: x >= 7.75
#              sqrt(x) * exp(-x) * besseli1(x) = P22(1/x)
#    where P13 and P22 are a 16 and 22 degree polynomial respectively.
#
#    Remez.jl is then used to approximate the polynomial coefficients of
#    P13(y) = (besseli1(2 * sqrt(y)) / sqrt(y) - 1 - y/2) / y^2
#    N,D,E,X = ratfn_minimax(g, (1/1e6, 1/7.75), 21, 0)
#
#    A third branch is used for scaled functions for large values
#
#    Horner's scheme is then used to evaluate all polynomials.
#    ArbNumerics.jl is used as the reference bessel implementations with 75 digits.
#
#    Calculation of besseli and besselkx can be done with downward recursion starting with
#    besseli_{nu+1} and besseli_{nu}. Higher orders are determined by a uniform asymptotic
#    expansion similar to besselk (see notes there) using Equation 10.41.3 [3].
#
# 
# [1] "Rational Approximations for the Modified Bessel Function of the First Kind 
#     - I0(x) for Computations with Double Precision" by Pavel Holoborodko     
# [2] "Rational Approximations for the Modified Bessel Function of the First Kind 
#     - I1(x) for Computations with Double Precision" by Pavel Holoborodko
# [3] https://dlmf.nist.gov/10.41

"""
    besseli0(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the first kind of order zero, ``I_0(x)``.
"""
function besseli0(x::T) where T <: Union{Float32, Float64}
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, besseli0_small_coefs(T)), 1)
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), besseli0_med_coefs(T)) / sqrt(x)
        return a * s
    end
end

"""
    besseli0x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the first kind of order zero, ``I_0(x)*e^{-x}``.
"""
function besseli0x(x::T) where T <: Union{Float32, Float64}
    T == Float32 ? branch = 50 : branch = 500
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, besseli0_small_coefs(T)), 1) * exp(-x)
    elseif x < branch
        return evalpoly(inv(x), besseli0_med_coefs(T)) / sqrt(x)
    else
        return evalpoly(inv(x), besseli0_large_coefs(T)) / sqrt(x)
    end
end

"""
    besseli1(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the first kind of order one, ``I_1(x)``.
"""
function besseli1(x::T) where T <: Union{Float32, Float64}
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2
    else
        a = exp(z / 2)
        s = a * evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
        z =  a * s
    end
    if x < zero(x)
        z = -z
    end
    return z
end

"""
    besseli1x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the first kind of order one, ``I_1(x)*e^{-x}``.
"""
function besseli1x(x::T) where T <: Union{Float32, Float64}
    T == Float32 ? branch = 50 : branch = 500
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2 * exp(-z)
    elseif z < branch
        z = evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
    else
        z = evalpoly(inv(z), besseli1_large_coefs(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end

#              Modified Bessel functions of the first kind of order nu
#                           besseli(nu, x)
#
#    A numerical routine to compute the modified Bessel function of the first kind I_{ν}(x) [1]
#    for real orders and arguments of positive or negative value. The routine is based on several
#    publications [2-6] that calculate I_{ν}(x) for positive arguments and orders where
#    reflection identities are used to compute negative arguments and orders.
#
#    In particular, the reflectance identities for negative noninteger orders I_{−ν}(x) = I_{ν}(x) + 2 / πsin(πν)*Kν(x)
#    and for negative integer orders I_{−n}(x) = I_n(x) are used.
#    For negative arguments of integer order, In(−x) = (−1)^n In(x) is used and for
#    noninteger orders, Iν(−x) = exp(iπν) Iν(x) is used. For negative orders and arguments the previous identities are combined.
#
#    The identities are computed by calling the `besseli_positive_args(nu, x)` function which computes I_{ν}(x)
#    for positive arguments and orders. For large orders, Debye's uniform asymptotic expansions are used where large arguments (x>>nu)
#    a large argument expansion is used. The rest of the values are computed using the power series.

# [1] https://dlmf.nist.gov/10.40.E1
# [2] Amos, Donald E. "Computation of modified Bessel functions and their ratios." Mathematics of computation 28.125 (1974): 239-251.
# [3] Gatto, M. A., and J. B. Seery. "Numerical evaluation of the modified Bessel functions I and K." 
#     Computers & Mathematics with Applications 7.3 (1981): 203-209.
# [4] Temme, Nico M. "On the numerical evaluation of the modified Bessel function of the third kind." 
#     Journal of Computational Physics 19.3 (1975): 324-337.
# [5] Amos, DEv. "Algorithm 644: A portable package for Bessel functions of a complex argument and nonnegative order." 
#     ACM Transactions on Mathematical Software (TOMS) 12.3 (1986): 265-273.
# [6] Segura, Javier, P. Fernández de Córdoba, and Yu L. Ratis. "A code to evaluate modified bessel functions based on thecontinued fraction method." 
#     Computer physics communications 105.2-3 (1997): 263-272.
#

"""
    besseli(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order nu, ``I_{nu}(x)``.
"""
besseli(nu::Real, x) = _besseli(nu, float(x))

_besseli(nu, x::Float16) = Float16(_besseli(nu, Float32(x)))

function _besseli(nu, x::T) where T <: Union{Float32, Float64}
    isinteger(nu) && return _besseli(Int(nu), x)
    ~isfinite(x) && return x
    abs_nu = abs(nu)
    abs_x = abs(x)

    if nu >= 0
        if x >= 0
            return besseli_positive_args(abs_nu, abs_x)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return cispi(abs_nu) * besseli_positive_args(abs_nu, abs_x)
        end
    else
        if x >= 0
            return besseli_positive_args(abs_nu, abs_x) + 2 / π * sinpi(abs_nu) * besselk_positive_args(abs_nu, abs_x)
        else
            #Iv = besseli_positive_args(abs_nu, abs_x)
            #Kv = besselk_positive_args(abs_nu, abs_x)
            #return cispi(abs_nu) * Iv + 2 / π * sinpi(abs_nu) * (cispi(-abs_nu) * Kv - im * π * Iv)
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
        end
    end
end
function _besseli(nu::Integer, x::T) where T <: Union{Float32, Float64}
    ~isfinite(x) && return x
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    if x >= 0
        return besseli_positive_args(abs_nu, abs_x)
    else
        return sg * besseli_positive_args(abs_nu, abs_x)
    end
end

"""
    besseli_positive_args(nu, x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the first kind of order nu, ``I_{nu}(x)`` for positive arguments.
"""
function besseli_positive_args(nu, x::T) where T <: Union{Float32, Float64}
    iszero(nu) && return besseli0(x)
    isone(nu) && return besseli1(x)

    # use large argument expansion if x >> nu
    besseli_large_argument_cutoff(nu, x) && return besseli_large_argument(nu, x)

    # use uniform debye expansion if x or nu is large
    besselik_debye_cutoff(nu, x) && return besseli_large_orders(nu, x)

    # for rest of values use the power series
    return besseli_power_series(nu, x)
end

"""
    besselix(nu, x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the first kind of order nu, ``I_{nu}(x)*e^{-x}``.
Nu must be real.
"""
besselix(nu::Real, x::Real) = _besselix(nu, float(x))

_besselix(nu, x::Float16) = Float16(_besselix(nu, Float32(x)))

function _besselix(nu, x::T) where T <: Union{Float32, Float64}
    iszero(nu) && return besseli0x(x)
    isone(nu) && return besseli1x(x)
    isinf(x) && return T(Inf)

    # use large argument expansion if x >> nu
    besseli_large_argument_cutoff(nu, x) && return besseli_large_argument_scaled(nu, x)

    # use uniform debye expansion if x or nu is large
    besselik_debye_cutoff(nu, x) && return besseli_large_orders_scaled(nu, x)

    # for rest of values use the power series
    return besseli_power_series(nu, x) * exp(-x)
end

#####
#####  Debye's uniform asymptotic for I_{nu}(x)
#####

# Implements the uniform asymptotic expansion https://dlmf.nist.gov/10.41
# In general this is valid when either x or nu is gets large
# see the file src/U_polynomials.jl for more details
"""
    besseli_large_orders(nu, x::T)

Debey's uniform asymptotic expansion for large order valid when v-> ∞ or x -> ∞
"""
function besseli_large_orders(v, x::T) where T
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQ1O2PI(S) * sqrt(inv(v)) * exp(v*n) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_In(p, v, p2, T))
end

function besseli_large_orders_scaled(v, x::T) where T
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQ1O2PI(S) * sqrt(inv(v)) * exp(v*n - x) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_In(p, v, p2, T))
end

#####
#####  Large argument expansion (x>>nu) for I_{nu}(x)
#####

# Implements the uniform asymptotic expansion https://dlmf.nist.gov/10.40.E1
# In general this is valid when x > nu^2
"""
    besseli_large_orders(nu, x::T)

Debey's uniform asymptotic expansion for large order valid when v-> ∞ or x -> ∞
"""
function besseli_large_argument(v, x::T) where T
    a = exp(x / 2)
    coef = a / sqrt(2 * (π * x))
    return T(_besseli_large_argument(v, x) * coef * a)
end

besseli_large_argument_scaled(v, x::T) where T =  T(_besseli_large_argument(v, x) / sqrt(2 * (π * x)))

function _besseli_large_argument(v, x::T) where T
    MaxIter = 5000
    S = promote_type(T, Float64)
    v, x = S(v), S(x)

    fv2 = 4 * v^2
    term = one(S)
    res = term
    s = -term
    for i in 1:MaxIter
        offset = muladd(2, i, -1)
        term *= muladd(offset, -offset, fv2) / (8 * x * i)
        res = muladd(term, s, res)
        s = -s
        abs(term) <= eps(T) && break
    end
    return res
end

besseli_large_argument_cutoff(nu, x::Float64) = x > 23.0 && x > nu^2 / 1.8 + 23.0
besseli_large_argument_cutoff(nu, x::Float32) = x > 18.0f0 && x > nu^2 / 19.5f0 + 18.0f0

#####
#####  Power series for I_{nu}(x)
#####

# Use power series form of I_v(x) which is generally accurate across all values though slower for larger x
# https://dlmf.nist.gov/10.25.E2
"""
    besseli_power_series(nu, x::T) where T <: Float64

Computes ``I_{nu}(x)`` using the power series for any value of nu.
"""
function besseli_power_series(v, x::ComplexOrReal{T}) where T
    MaxIter = 3000
    S = eltype(x)
    out = zero(S)
    xs = (x/2)^v
    a = xs / gamma(v + one(T))
    t2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        abs(a) < eps(T) * abs(out) && break
        a *= inv((v + i + one(T)) * (i + one(T))) * t2
    end
    return out
end

#=
# the following is a deprecated version of the continued fraction approach
# using K0 and K1 as starting values then forward recurrence up till nu
# then using the wronskian to getting I_{nu}
# in general this method is slow and depends on starting values of K0 and K1
# which is not very flexible for arbitary orders

function _besseli_continued_fractions(nu, x::T) where T
    S = promote_type(T, Float64)
    xx = S(x)
    knum1, knu = besselk_up_recurrence(xx, besselk1(xx), besselk0(xx), 1, nu-1)
    # if knu or knum1 is zero then besseli will likely overflow
    (iszero(knu) || iszero(knum1)) && return throw(DomainError(x, "Overflow error"))
    return 1 / (x * (knum1 + knu / steed(nu, x)))
end
function _besseli_continued_fractions_scaled(nu, x::T) where T
    S = promote_type(T, Float64)
    xx = S(x)
    knum1, knu = besselk_up_recurrence(xx, besselk1x(xx), besselk0x(xx), 1, nu-1)
    # if knu or knum1 is zero then besseli will likely overflow
    (iszero(knu) || iszero(knum1)) && return throw(DomainError(x, "Overflow error"))
    return 1 / (x * (knum1 + knu / steed(nu, x)))
end
function steed(n, x::T) where T
    MaxIter = 1000
    xinv = inv(x)
    xinv2 = 2 * xinv
    d = x / (n + n)
    a = d
    h = a
    b = muladd(2, n, 2) * xinv
    for _ in 1:MaxIter
        d = inv(b + d)
        a *= muladd(b, d, -1)
        h = h + a
        b = b + xinv2
        abs(a / h) <= eps(T) && break
    end
    return h
end
=#
function _besseli(nu, x::Complex{T}) where T <: Union{Float32, Float64}
    isinteger(nu) && return _besseli(Int(nu), x)
    ~isfinite(x) && return x
    abs_nu = abs(nu)
    abs_x = abs(x)

    if nu >= 0
        if x >= 0
            return besseli_positive_args(abs_nu, abs_x)
        else
            return cispi(abs_nu) * besseli_positive_args(abs_nu, abs_x)
        end
    else
        return throw(DomainError(nu, "nu must be positive if x is complex"))
        #=
        if x >= 0
            return besseli_positive_args(abs_nu, abs_x) + 2 / π * sinpi(abs_nu) * besselk_positive_args(abs_nu, abs_x)
        else
            Iv = besseli_positive_args(abs_nu, abs_x)
            Kv = besselk_positive_args(abs_nu, abs_x)
            return cispi(abs_nu) * Iv + 2 / π * sinpi(abs_nu) * (cispi(-abs_nu) * Kv - im * π * Iv)
        end
        =#
    end
end
function _besseli(nu::Integer, x::Complex{T}) where T <: Union{Float32, Float64}
    ~isfinite(x) && return x
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    if x >= 0
        return besseli_positive_args(abs_nu, abs_x)
    else
        return sg * besseli_positive_args(abs_nu, abs_x)
    end
end

Base.eps(::Type{Complex{T}}) where {T<:AbstractFloat} = eps(T)