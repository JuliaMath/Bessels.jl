#    Modified Bessel functions of the second kind of order zero and one
#                       besselk0, besselk1
#
#    Scaled modified Bessel functions of the second kind of order zero and one
#                       besselk0x, besselk1x
#
#    (Scaled) Modified Bessel functions of the second kind of order nu
#                       besselk, besselkx
#
#
#    Calculation of besselk0 is done in two branches using polynomial approximations [1]
#
#    Branch 1: x < 1.0 
#              besselk0(x) + log(x)*besseli0(x) = P7(x^2)
#                            besseli0(x) = [x/2]^2 * P6([x/2]^2) + 1
#    Branch 2: x >= 1.0
#              sqrt(x) * exp(x) * besselk0(x) = P22(1/x) / Q2(1/x)
#    where P7, P6, P22, and Q2 are 7, 6, 22, and 2 degree polynomials respectively.
#
#
#
#    Calculation of besselk1 is done in two branches using polynomial approximations [2]
#
#    Branch 1: x < 1.0 
#              besselk1(x) - log(x)*besseli1(x) - 1/x = x*P8(x^2)
#                            besseli1(x) = [x/2]^2 * (1 + 0.5 * (*x/2)^2 + (x/2)^4 * P5([x/2]^2))
#    Branch 2: x >= 1.0
#              sqrt(x) * exp(x) * besselk1(x) = P22(1/x) / Q2(1/x)
#    where P8, P5, P22, and Q2 are 8, 5, 22, and 2 degree polynomials respectively.
#
#
#    The polynomial coefficients are taken from boost math library [3].
#    Evaluation of the coefficients using Remez.jl is prohibitive due to the increase
#    precision required in ArbNumerics.jl. 
#
#    Horner's scheme is then used to evaluate all polynomials.
#
#    Calculation of besselk and besselkx can be done with recursion starting with
#    besselk0 and besselk1 and using upward recursion for any value of nu (order).
#
#                    K_{nu+1} = (2*nu/x)*K_{nu} + K_{nu-1}
#
#    When nu is large, a large amount of recurrence is necesary.
#    We consider uniform asymptotic expansion for large orders to more efficiently
#    compute besselk(nu, x) when nu is larger than 100 (Let's double check this cutoff)
#    The boundary should be carefully determined for accuracy and machine roundoff.
#    We use 10.41.4 from the Digital Library of Math Functions [5].
#    This is also 9.7.8 in Abramowitz and Stegun [6].
#    K_{nu}(nu*z) = sqrt(pi / 2nu) *exp(-nu*n)/(1+z^2)^1/4 * sum((-1^k)U_k(p) /nu^k)) for k=0 -> infty
#    The U polynomials are the most tricky. They are listed up to order 4 in Table 9.39
#    of [6]. For Float32, >=4 U polynomials are usually necessary. For Float64 values,
#    >= 8 orders are needed. However, this largely depends on the cutoff of order you need.
#    For moderatelly sized orders (nu=50), might need 11-12 orders to reach good enough accuracy
#    in double precision. 
#
#    However, calculation of these higher order U polynomials are tedious. These have been hand
#    calculated and somewhat crosschecked with symbolic math. There could be errors. They are listed
#    in src/U_polynomials.jl as a reference as higher orders are impossible to find while being needed for any meaningfully accurate calculation.
#    For large orders these formulas will converge much faster than using upward recurrence.
#
#    
# [1] "Rational Approximations for the Modified Bessel Function of the Second Kind 
#     - K0(x) for Computations with Double Precision" by Pavel Holoborodko     
# [2] "Rational Approximations for the Modified Bessel Function of the Second Kind 
#     - K1(x) for Computations with Double Precision" by Pavel Holoborodko
# [3] https://github.com/boostorg/math/tree/develop/include/boost/math/special_functions/detail
# [4] "Computation of Bessel Functions of Complex Argument and Large Order" by Donald E. Amos
#      Sandia National Laboratories
# [5] https://dlmf.nist.gov/10.41
# [6] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical functions with formulas, graphs, and mathematical tables. 
#     Vol. 55. US Government printing office, 1964.
#

"""
    besselk0(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order zero, ``K_0(x)``.
"""
function besselk0(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        a = x * x / 4
        s = muladd(evalpoly(a, P1_k0(T)), inv(evalpoly(a, Q1_k0(T))), T(Y_k0))
        a = muladd(s, a, 1)
        return muladd(-a, log(x), evalpoly(x * x, P2_k0(T)))
    else
        s = exp(-x / 2)
        a = muladd(evalpoly(inv(x), P3_k0(T)), inv(evalpoly(inv(x), Q3_k0(T))), one(T)) * s / sqrt(x)
        return a * s
    end
end
"""
    besselk0x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the second kind of order zero, ``K_0(x)*e^{x}``.
"""
function besselk0x(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        a = x * x / 4
        s = muladd(evalpoly(a, P1_k0(T)), inv(evalpoly(a, Q1_k0(T))), T(Y_k0))
        a = muladd(s, a, 1)
        return muladd(-a, log(x), evalpoly(x * x, P2_k0(T))) * exp(x)
    else
        return muladd(evalpoly(inv(x), P3_k0(T)), inv(evalpoly(inv(x), Q3_k0(T))), one(T)) / sqrt(x)
    end
end
"""
    besselk1(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order one, ``K_1(x)``.
"""
function besselk1(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        z = x * x
        a = z / 4
        pq = muladd(evalpoly(a, P1_k1(T)), inv(evalpoly(a, Q1_k1(T))), Y_k1(T))
        pq = muladd(pq * a, a, (a / 2 + 1))
        a = pq * x / 2
        pq = muladd(evalpoly(z, P2_k1(T)) / evalpoly(z, Q2_k1(T)), x, inv(x))
        return muladd(a, log(x), pq)
    else
        s = exp(-x / 2)
        a = muladd(evalpoly(inv(x), P3_k1(T)), inv(evalpoly(inv(x), Q3_k1(T))), Y2_k1(T)) * s / sqrt(x)
        return a * s
    end
end
"""
    besselk1x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the second kind of order one, ``K_1(x)*e^{x}``.
"""
function besselk1x(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        z = x * x
        a = z / 4
        pq = muladd(evalpoly(a, P1_k1(T)), inv(evalpoly(a, Q1_k1(T))), Y_k1(T))
        pq = muladd(pq * a, a, (a / 2 + 1))
        a = pq * x / 2
        pq = muladd(evalpoly(z, P2_k1(T)) / evalpoly(z, Q2_k1(T)), x, inv(x))
        return muladd(a, log(x), pq) * exp(x)
    else
        return muladd(evalpoly(inv(x), P3_k1(T)), inv(evalpoly(inv(x), Q3_k1(T))), Y2_k1(T)) / sqrt(x)
    end
end
#=
If besselk0(x) or besselk1(0) is equal to zero
this will underflow and always return zero even if besselk(x, nu)
is larger than the smallest representing floating point value.
In other words, for large values of x and small to moderate values of nu,
this routine will underflow to zero.
For small to medium values of x and large values of nu this will overflow and return Inf.
=#
#=
"""
    besselk(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order nu, ``K_{nu}(x)``.
"""
function besselk(nu, x::T) where T <: Union{Float32, Float64, BigFloat}
    T == Float32 ? branch = 20 : branch = 50
    if nu < branch
        return besselk_up_recurrence(x, besselk1(x), besselk0(x), 1, nu)[1]
    else
        return besselk_large_orders(nu, x)
    end
end
=#
"""
    besselk(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the second kind of order nu, ``K_{nu}(x)*e^{x}``.
"""
function besselkx(nu::Int, x::T) where T <: Union{Float32, Float64}
    T == Float32 ? branch = 20 : branch = 50
    if nu < branch
        return besselk_up_recurrence(x, besselk1x(x), besselk0x(x), 1, nu)[1]
    else
        return besselk_large_orders_scaled(nu, x)
    end
end
function besselk_large_orders(v, x::T) where T <: Union{Float32, Float64, BigFloat}
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQPIO2(S) * sqrt(inv(v)) * exp(-v*n) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_Kn(p, v, p2, T))
end
function besselk_large_orders_scaled(v, x::T) where T <: Union{Float32, Float64, BigFloat}
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQPIO2(S) * sqrt(inv(v)) * exp(-v*n + x) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_Kn(p, v, p2, T))
end

function besselk(nu, x::T) where T <: Union{Float32, Float64, BigFloat}
    (isinteger(nu) && nu < 250) && return besselk_up_recurrence(x, besselk1(x), besselk0(x), 1, nu)[1]

    if nu > 25.0 || x > 35.0
        return besselk_large_orders(nu, x)
    elseif x < 2.0 || nu > 1.6x - 1.0
        return besselk_power_series(nu, x)
    else
        return besselk_continued_fraction(nu, x)
    end
end

# could also use the continued fraction for inu/inmu1
# but it seems like adapting the besseli_power series
# to give both nu and nu-1 is faster

#inum1 = besseli_power_series(nu-1, x)
#H_inu = steed(nu, x)
#inu = besseli_power_series(nu, x)#inum1 * H_inu
function besselk_continued_fraction(nu, x)
    inu, inum1 = besseli_power_series_inu_inum1(nu, x)
    H_knu = besselk_ratio_knu_knup1(nu-1, x)
    return 1 / (x * (inum1 + inu / H_knu))
end

function besseli_power_series_inu_inum1(v, x::T) where T
    MaxIter = 3000
    out = zero(T)
    out2 = zero(T)
    x2 = x / 2
    xs = (x2)^v
    gmx = xs / gamma(v)
    a = gmx / v
    b = gmx / x2
    t2 = x2*x2
    for i in 0:MaxIter
        out += a
        out2 += b
        abs(a) < eps(T) * abs(out) && break
        a *= inv((v + i + one(T)) * (i + one(T))) * t2
        b *= inv((v + i) * (i + one(T))) * t2
    end
    return out, out2
end


# slightly modified version of https://github.com/heltonmc/Bessels.jl/issues/17#issuecomment-1195726642 from @cgeoga
function besselk_ratio_knu_knup1(v, x::T) where T
    MaxIter = 1000
    # do the modified Lentz method:
    (hn, Dn, Cn) = (1e-50, zero(v), 1e-50)

    jf   = one(T)
    vv   = v*v
    for _ in 1:MaxIter
        an  = (vv - ((2*jf - 1)^2) * T(0.25))
        bn  = 2 * (x + jf)
        Cn  = an / Cn + bn      
        Dn  = inv(muladd(an, Dn, bn))
        del = Dn * Cn
        hn *= del
        abs(del - 1) < eps(T) && break
        jf += one(T)
    end
    xinv = inv(x)
    return xinv * (v + x + 1/2) + xinv * hn
end

# originally contributed by @cgeoga https://github.com/cgeoga/BesselK.jl/blob/main/src/besk_ser.jl
# Equation 3.2 from Geoga, Christopher J., et al. "Fitting Mat\'ern Smoothness Parameters Using Automatic Differentiation." 
# arXiv preprint arXiv:2201.00090 (2022).
function besselk_power_series(v, x::T) where T
    MaxIter = 1000
    # precompute a handful of things:
    xd2  = x / 2
    xd22 = xd2 * xd2
    half = one(T) / 2
    # (x/2)^(±v). Writing the literal power doesn't seem to change anything here,
    # and I think this is faster:
    lxd2 = log(xd2)
    xd2_v = exp(v*lxd2)
    xd2_nv = exp(-v*lxd2)
    # use the gamma function a couple times to start:
    gam_v = gamma(v)
    xp1 = abs(v) + one(T)
    gam_nv = π / sinpi(xp1) / _gamma(xp1)
    gam_1mv = -gam_nv*v # == gamma(one(T)-v)
    gam_1mnv = gam_v*v   # == gamma(one(T)+v)
    (gpv, gmv) = (gam_1mnv, gam_1mv)
    # One final re-compression of a few things:
    _t1 = gam_v*xd2_nv*gam_1mv
    _t2 = gam_nv*xd2_v*gam_1mnv
    # A couple series-specific accumulators:
    (xd2_pow, fact_k, floatk, out) = (one(T), one(T), zero(T), zero(T))
    for _ in 0:MaxIter
      t1 = half*xd2_pow
      tmp = _t1/(gmv*fact_k)
      tmp += _t2/(gpv*fact_k)
      term = t1*tmp
      out += term
      abs(term/out) < eps(T) && return out
      # Use the trick that gamma(1+k+1+v) == gamma(1+k+v)*(1+k+v) to skip gamma calls:
      (gpv, gmv) = (gpv*(one(T)+v+floatk), gmv*(one(T)-v+floatk)) 
      xd2_pow *= xd22
      fact_k *= (floatk+1)
      floatk += one(T)
    end
    return out
  end
