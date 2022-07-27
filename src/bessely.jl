#    Bessel functions of the second kind of order zero and one
#                       bessely0, bessely1
#
#    Calculation of bessely0 is done in three branches using polynomial approximations
#
#    Branch 1: x <= 5.0
#              bessely0 = R(x^2) + 2*log(x)*besselj0(x) / pi
#    where r1 and r2 are zeros of J0
#    and P3 and Q8 are a 3 and 8 degree polynomial respectively
#    Polynomial coefficients are from [1] which is based on [2].
#    For tiny arugments the power series expansion is used.
#
#    Branch 2: 5.0 < x < 25.0
#              bessely0 = sqrt(2/(pi*x))*(sin(x - pi/4)*R7(x) - cos(x - pi/4)*R8(x))
#    Hankel's asymptotic expansion is used
#    where R7 and R8 are rational functions (Pn(x)/Qn(x)) of degree 7 and 8 respectively
#    See section 4 of [3] for more details and [1] for coefficients of polynomials
#
#   Branch 3: x >= 25.0
#              bessely0 = sqrt(2/(pi*x))*beta(x)*(sin(x - pi/4 - alpha(x))
#   See modified expansions given in [3]. Exact coefficients are used.
#
#   Calculation of bessely1 is done in a similar way as bessely0.
#   See [3] for details on similarities.
#
# [1] https://github.com/deepmind/torch-cephes
# [2] Cephes Math Library Release 2.8:  June, 2000 by Stephen L. Moshier
# [3] Harrison, John. "Fast and accurate Bessel function computation."
#     2009 19th IEEE Symposium on Computer Arithmetic. IEEE, 2009.
#

"""
    bessely0(x::T) where T <: Union{Float32, Float64}

Bessel function of the second kind of order zero, ``Y_0(x)``.
"""
function bessely0(x::T) where T <: Union{Float32, Float64}
    if x <= zero(x)
        if iszero(x)
            return T(-Inf)
        else
            return throw(DomainError(x, "NaN result for non-NaN input."))
        end
    elseif isinf(x)
        return zero(x)
    end
    return _bessely0_compute(x)
end
function _bessely0_compute(x::Float64)
    T = Float64
    if x <= 5.0
        z = x * x
        w = evalpoly(z, YP_y0(T)) / evalpoly(z, YQ_y0(T))
        w += TWOOPI(T) * log(x) * besselj0(x)
        return w
    elseif x < 25.0
        w = 5.0 / x
        z = w * w
        p = evalpoly(z, PP_y0(T)) / evalpoly(z, PQ_y0(T))
        q = evalpoly(z, QP_y0(T)) / evalpoly(z, QQ_y0(T))
        xn = x - PIO4(T)
        sc = sincos(xn)
        p = p * sc[1] + w * q * sc[2]
        return p * SQ2OPI(T) / sqrt(x)
    else
        xinv = inv(x)
        x2 = xinv*xinv
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
        xn = muladd(xinv, q, - PIO4(T))

        # the following computes b = sin(x + xn) more accurately
        # see src/misc.jl
        b = sin_sum(x, xn)
        return a * b
    end
end
function _bessely0_compute(x::Float32)
    T = Float32
    if x <= 2.0f0
        z = x * x
        YZ1 =  0.43221455686510834878f0
        w = (z - YZ1) * evalpoly(z, YP_y1(T))
        w += TWOOPI(T) * log(x) * besselj0(x)
        return w
    else
        q = 1.0f0 / x
        w = sqrt(q)
        p = w * evalpoly(q, MO_y1(T))
        w = q * q
        xn = q * evalpoly(w, PH_y1(T)) - PIO4(T)
        p = p * sin(xn + x)
        return p
    end
end

"""
    bessely1(x::T) where T <: Union{Float32, Float64}

Bessel function of the second kind of order one, ``Y_1(x)``.
"""
function bessely1(x::T) where T <: Union{Float32, Float64}
    if x <= zero(x)
        if iszero(x)
            return T(-Inf)
        else
            return throw(DomainError(x, "NaN result for non-NaN input."))
        end
    elseif isinf(x)
        return zero(x)
    end
    return _bessely1_compute(x)
end
function _bessely1_compute(x::Float64)
    T = Float64
    if x <= 5
        z = x * x
        w = x * (evalpoly(z, YP_y1(T)) / evalpoly(z, YQ_y1(T)))
        w += TWOOPI(T) * (besselj1(x) * log(x) - inv(x))
        return w
    elseif x < 25.0
        w = 5.0 / x
        z = w * w
        p = evalpoly(z, PP_j1(T)) / evalpoly(z, PQ_j1(T))
        q = evalpoly(z, QP_j1(T)) / evalpoly(z, QQ_j1(T))
        xn = x - THPIO4(T)
        sc = sincos(xn)
        p = p * sc[1] + w * q * sc[2]
        return p * SQ2OPI(T) / sqrt(x)
    else
      
        return besseljy_large_argument(1.0, x)[2]
    end
end
function _bessely1_compute(x::Float32)
    T = Float32
    if x <= 2.0f0
        z = x * x
        YO1 = 4.66539330185668857532f0
        w = (z - YO1) * x * evalpoly(z, YP32)
        w += TWOOPI(Float32) * (besselj1(x) * log(x) - inv(x))
        return w
    else
        q = inv(x)
        w = sqrt(q)
        p = w * evalpoly(q, MO132)
        w = q * q
        xn = q * evalpoly(w, PH132) - THPIO4(Float32)
        p = p * sin(xn + x)
        return p
    end
end

"""
    bessely(nu, x::T) where T <: Union{Float32, Float64}

Bessel function of the first kind of order nu, ``J_{nu}(x)``.
Nu must be real.
"""
function _bessely(nu, x::T) where T
    nu == 0 && return bessely0(x)
    nu == 1 && return bessely1(x)

    # large argument branch see src/asymptotics.jl
    besseljy_large_argument_cutoff(nu, x) && return besseljy_large_argument(nu, x)[2]

    # x < nu branch see src/U_polynomials.jl
    besseljy_debye_cutoff(nu, x) && return besseljy_debye(nu, x)[2]

    # use forward recurrence if nu is an integer up until it becomes inefficient 
    if (isinteger(nu) && nu < 250)
        ynum1 = bessely0(x)
        ynu = bessely1(x)
        return besselj_up_recurrence(x, ynu, ynum1, 1, nu)[2]
    end

    # use power series for small x and for when nu > x
    bessely_series_cutoff(nu, x) && return bessely_power_series(nu, x)

    # use forward recurrence by shifting nu down
    # if x > besseljy_large_argument_min (see src/asymptotics.jl) we can shift nu down and use large arg. expansion for start values
    # if x <= 20.0 we can shift nu such that nu < -1.5*x where the power series can be used 
    if x > besseljy_large_argument_min(T)
        large_arg_diff = ceil(Int, nu - x * T(0.6))
        v2 = nu - large_arg_diff
        ynu = besseljy_large_argument(v2, x)[2]
        ynum1 = besseljy_large_argument(v2 - 1, x)[2]
        return besselj_up_recurrence(x, ynu, ynum1, v2, nu)[2]
    else
        # this method is very inefficient so ideally we could swap out a different algorithm here in the future
        return bessely_intermediate_args(nu, x)
    end
end


# Use power series form of J_v(x) to calculate Y_v(x) with
# Y_v(x) = (J_v(x)cos(v*π) - J_{-v}(x)) / sin(v*π),    v ~= 0, 1, 2, ...
# combined to calculate both J_v and J_{-v} in the same loop
# J_{-v} always converges slower so just check that convergence
# Combining the loop was faster than using two separate loops
# 
# this seems to work well for small arguments x < 7.0 for rel. error ~1e-14
# this also works well for nu > 1.35x - 4.5
# for nu > 25 more cancellation occurs near integer values
bessely_series_cutoff(v, x) = (x < 7.0) || v > 1.35*x - 4.5
function bessely_power_series(v, x::T) where T
    MaxIter = 3000
    out = zero(T)
    out2 = zero(T)
    a = (x/2)^v
    b = inv(a)
    a /= gamma(v + one(T))
    b /= gamma(-v + one(T))
    t2 = (x/2)^2
    for i in 0:3000
        out += a
        out2 += b
        abs(b) < eps(Float64) * abs(out2) && break
        a *= -inv((v + i + one(T)) * (i + one(T))) * t2
        b *= -inv((-v + i + one(T)) * (i + one(T))) * t2
    end
    s, c = sincospi(v)
    return (out*c - out2) / s
end

#=
### don't quite have this right
# probably not needed because we should use debye exapnsion for large nu
function log_bessely_power_series(v, x::T) where T
    MaxIter = 2000
    out = zero(T)
    out2 = zero(T)
    a = one(T)
    b = inv(gamma(1-v))
    x2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        out2 += b
        a *= -x2 * inv((i + one(T)) * (v + i + one(T)))
        b *= -x2 * inv((i + one(T)) * (-v + i + one(T)))
        (abs(b) < eps(T) * abs(out2)) && break
    end
    logout = -loggamma(v + 1) + fma(v, log(x/2), log(out))
    sign = 1

    if out2 <= zero(T)
        sign = -1
        out2 = -out2
    end
    logout2 = -v * log(x/2) + log(out2)

    spi, cpi = sincospi(v)
   
    #tmp = logout2 + log(abs(sign*(inv(spi)) - exp(logout - logout2) * cpi / spi))
    tmp = logout + log((-cpi + exp(logout2) - logout) / spi)

    return -exp(tmp)
end
=#
function bessely_hyper(v, x)
    spi, cpi = sincospi(v)
    a = besselj_hyper(v,x)*cpi - besselj_hyper(-v,x)
    return a/sinpi(v)
end

function bessely_intermediate_args(v, x)
    v < 2 && return bessely_hyper(v, x)

    v_floor = v - floor(v) 
    Jnv = besselj_hyper(-v_floor,x)
    Jnvm1 = Jnv / steed_j(-v_floor, x)

    Jv = besselj_hyper(v_floor + 1,x)
    Jvm1 = Jv / steed_j(v_floor + 1, x)

    spi, cpi = sincospi(v_floor + 1)
    Yv = (Jv*cpi - Jnvm1) / spi

    spi, cpi = sincospi(v_floor)
    Yvm1 = (Jvm1*cpi - Jnv) / spi

    return  besselj_up_recurrence(x, Yv, Yvm1, v_floor+1, v)[2]
end

function steed_j(n, x::T) where T
    MaxIter = 1000
    xinv = inv(x)
    xinv2 = 2 * xinv
    d = x / (n + n)
    a = d
    h = a
    b = muladd(2, n, 2) * xinv
    for _ in 1:MaxIter
        d = inv(b - d)
        a *= muladd(b, d, -1)
        h = h + a
        b = b + xinv2
        abs(a / h) <= eps(T) && break
    end
    return h
end

function besselj_hyper(v, x)
       a = drummond0F1(v + 1, -x^2/4)
       return (x/2)^v / gamma(v+1) * a
end

function drummond0F1(β::T1, z::T2; kmax::Int = 10_000) where {T1, T2}
    T = promote_type(T1, T2)
    if norm(z) < eps(real(T))
        return one(T)
    end
    ζ = inv(z)
    Nlo = β*ζ
    Dlo = β*ζ
    Tlo = Nlo/Dlo
    Nmid = ((β+1)*(2)*ζ - 1)*Nlo + (β+1)*(2)*ζ
    Dmid = ((β+1)*(2)*ζ - 1)*Dlo
    Tmid = Nmid/Dmid
    Nhi = ((β+2)*(3)*ζ - 1)*Nmid + (β+4)*ζ*Nlo + (β+4)*ζ
    Dhi = ((β+2)*(3)*ζ - 1)*Dmid + (β+4)*ζ*Dlo
    Thi = Nhi/Dhi
    k = 2
    Nhi, Nmid, Nlo = ((β+k+1)*(k+2)*ζ-1)*Nhi + k*(β+2k+2)*ζ*Nmid + k*(k-1)*ζ*Nlo + 2ζ, Nhi, Nmid
    Dhi, Dmid, Dlo = ((β+k+1)*(k+2)*ζ-1)*Dhi + k*(β+2k+2)*ζ*Dmid + k*(k-1)*ζ*Dlo, Dhi, Dmid
    Thi, Tmid, Tlo = Nhi/Dhi, Thi, Tmid
    k += 1
    while k < kmax && errcheck(Tmid, Thi, eps(real(T)))
        Nhi, Nmid, Nlo = ((β+k+1)*(k+2)*ζ-1)*Nhi + k*(β+2k+2)*ζ*Nmid + k*(k-1)*ζ*Nlo, Nhi, Nmid
        Dhi, Dmid, Dlo = ((β+k+1)*(k+2)*ζ-1)*Dhi + k*(β+2k+2)*ζ*Dmid + k*(k-1)*ζ*Dlo, Dhi, Dmid
        Thi, Tmid, Tlo = Nhi/Dhi, Thi, Tmid
        k += 1
    end
    return isfinite(Thi) ? Thi : isfinite(Tmid) ? Tmid : Tlo
end
@inline errcheck(x, y, tol) = isfinite(x) && isfinite(y) && (norm2(x-y) > max(norm2(x), norm2(y))*tol)
@inline norm2(x) = norm(x)

