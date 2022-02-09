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

"""
    besseli(nu, x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the first kind of order nu, ``I_{nu}(x)``.
Nu must be real.
"""
function besseli(nu, x::T) where T <: Union{Float32, Float64}
    nu == 0 && return besseli0(x)
    nu == 1 && return besseli1(x)
    
    if x > maximum((T(30), nu^2 / 4))
        return T(besseli_large_argument(nu, x))
    elseif x <= 2 * sqrt(nu + 1)
        return T(besseli_small_arguments(nu, x))
    elseif nu < 100
        return T(_besseli_continued_fractions(nu, x))
    else
        return T(besseli_large_orders(nu, x))
    end
end

"""
    besselix(nu, x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the first kind of order nu, ``I_{nu}(x)*e^{-x}``.
Nu must be real.
"""
function besselix(nu, x::T) where T <: Union{Float32, Float64}
    nu == 0 && return besseli0x(x)
    nu == 1 && return besseli1x(x)

    if x > maximum((T(30), nu^2 / 4))
        return T(besseli_large_argument_scaled(nu, x))
    elseif x <= 2 * sqrt(nu + 1)
        return T(besseli_small_arguments(nu, x)) * exp(-x)
    elseif nu < 100
        return T(_besseli_continued_fractions_scaled(nu, x))
    else
        return besseli_large_orders_scaled(nu, x)
    end
end
function besseli_large_orders(v, x::T) where T <: Union{Float32, Float64}
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQ1O2PI(S) * sqrt(inv(v)) * exp(v*n) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return coef*Uk_poly_In(p, v, p2, T)
end
function besseli_large_orders_scaled(v, x::T) where T <: Union{Float32, Float64}
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
function _besseli_continued_fractions(nu, x::T) where T
    S = promote_type(T, Float64)
    xx = S(x)
    knu, knum1 = up_recurrence(xx, besselk0(xx), besselk1(xx), nu)
    # if knu or knum1 is zero then besseli will likely overflow
    (iszero(knu) || iszero(knum1)) && return throw(DomainError(x, "Overflow error"))
    return 1 / (x * (knum1 + knu / steed(nu, x)))
end
function _besseli_continued_fractions_scaled(nu, x::T) where T
    S = promote_type(T, Float64)
    xx = S(x)
    knu, knum1 = up_recurrence(xx, besselk0x(xx), besselk1x(xx), nu)
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
function besseli_large_argument(v, z::T) where T
    MaxIter = 1000
    a = exp(z / 2)
    coef = a / sqrt(2 * T(pi) * z)
    fv2 = 4 * v^2
    term = one(T)
    res = term
    s = -term
    for i in 1:MaxIter
        i = T(i)
        offset = muladd(2, i, -1)
        term *= T(0.125) * muladd(offset, -offset, fv2) / (z * i)
        res = muladd(term, s, res)
        s = -s
        abs(term) <= eps(T) && break
    end
    return res * coef * a
end
function besseli_large_argument_scaled(v, z::T) where T
    MaxIter = 1000
    coef = inv(sqrt(2 * T(pi) * z))
    fv2 = 4 * v^2
    term = one(T)
    res = term
    s = -term
    for i in 1:MaxIter
        i = T(i)
        offset = muladd(2, i, -1)
        term *= T(0.125) * muladd(offset, -offset, fv2) / (z * i)
        res = muladd(term, s, res)
        s = -s
        abs(term) <= eps(T) && break
    end
    return res * coef
end

function besseli_small_arguments(v, z::T) where T
    S = promote_type(T, Float64)
    x = S(z)
    if v < 20
        coef = (x / 2)^v / factorial(v)
    else
        vinv = inv(v)
        coef = sqrt(vinv / (2 * π)) * MathConstants.e^(v * (log(x / (2 * v)) + 1)) 
        coef *= evalpoly(vinv, (1, -1/12, 1/288,  139/51840, -571/2488320, -163879/209018880, 5246819/75246796800, 534703531/902961561600))
    end

    MaxIter = 1000
    out = one(S)
    zz = x^2 / 4
    a = one(S)
    for k in 1:MaxIter
        a *= zz / (k * (k + v))
        out += a
        a <= eps(T) && break
    end
    return coef * out
end
