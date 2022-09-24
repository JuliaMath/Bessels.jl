#                           Airy functions
#
#                       airyai(x), airybi(nu, x)
#                       airyaiprime(x), airybiprime(nu, x)
#
#    A numerical routine to compute the airy functions and their derivatives.
#    These routines use their relations to other special functions using https://dlmf.nist.gov/9.6.
#    Specifically see (NIST 9.6.E1 - 9.6.E9) for computation from the defined bessel functions.
#    For negative arguments these definitions are prone to some cancellation leading to higher errors.
#    In the future, these could be replaced with more custom routines as they depend on a single variable.
#

"""
    airyai(x)
Airy function of the first kind ``\\operatorname{Ai}(x)``.
"""
airyai(z::Number) = _airyai(float(z))

_airyai(x::Float16) = Float16(_airyai(Float32(x)))
_airyai(z::ComplexF16) = ComplexF16(_airyai(ComplexF32(z)))

function _airyai(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        isnan(z) && return z
        isinf(z) && return throw(DomainError(z, "airyai(z) not defined at infinity"))
    end
    x, y = real(z), imag(z)
    zabs = abs(z)

    airy_large_argument_cutoff(z) && return airyai_large_argument(z)
    airyai_power_series_cutoff(x, y) && return airyai_power_series(z)

    if x > zero(T)
        # use relation to besselk (http://dlmf.nist.gov/9.6.E1)
        zz = 2 * z^(T(3)/2) / 3
        return sqrt(z / 3) * besselk_continued_fraction_shift(one(T)/3, zz) / π
    else
        # z is close to the negative real axis
        # for imag(z) == 0 use reflection to compute in terms of bessel functions of first kind (http://dlmf.nist.gov/9.6.E5)
        # use computation for real numbers then convert to input type for stability
        # for imag(z) != 0 use rotation identity (http://dlmf.nist.gov/9.2.E14)
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs^(T(3)/2) / 3
            Jv, Yv = besseljy_positive_args(one(T)/3, xx)
            Jmv = (Jv - sqrt(T(3)) * Yv) / 2
            return convert(eltype(z), sqrt(xabs) * (Jmv + Jv) / 3)
        else
            return exp(pi*im/3) * _airyai(-z*exp(pi*im/3))  + exp(-pi*im/3) * _airyai(-z*exp(-pi*im/3))
        end
    end
end

"""
    airyaiprime(x)
Derivative of the Airy function of the first kind ``\\operatorname{Ai}'(x)``.
"""
airyaiprime(z::Number) = _airyaiprime(float(z))

_airyaiprime(x::Float16) = Float16(_airyaiprime(Float32(x)))
_airyaiprime(z::ComplexF16) = ComplexF16(_airyaiprime(ComplexF32(z)))

function _airyaiprime(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        isnan(z) && return z
        isinf(z) && return throw(DomainError(z, "airyai(z) not defined at infinity"))
    end
    x, y = real(z), imag(z)
    zabs = abs(z)

    airy_large_argument_cutoff(z) && return airyaiprime_large_argument(z)
    airyai_power_series_cutoff(x, y) && return airyaiprime_power_series(z)

    if x > zero(T)
        # use relation to besselk (http://dlmf.nist.gov/9.6.E2)
        zz = 2 * z^(T(3)/2) / 3
        return -z * besselk_continued_fraction_shift(T(2)/3, zz) / (π * sqrt(T(3)))
    else
        # z is close to the negative real axis
        # for imag(z) == 0 use reflection to compute in terms of bessel functions of first kind (http://dlmf.nist.gov/9.6.E5)
        # use computation for real numbers then convert to input type for stability
        # for imag(z) != 0 use rotation identity (http://dlmf.nist.gov/9.2.E14)
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs^(T(3)/2) / 3
            Jv, Yv = besseljy_positive_args(T(2)/3, xx)
            Jmv = -(Jv + sqrt(T(3))*Yv) / 2
            return convert(eltype(z), xabs * (Jv - Jmv) / 3)
        else
            return -(exp(2pi*im/3)*_airyaiprime(-z*exp(pi*im/3)) + exp(-2pi*im/3)*_airyaiprime(-z*exp(-pi*im/3)))
        end
    end
end

"""
    airybi(x)
Airy function of the second kind ``\\operatorname{Bi}(x)``.
"""
airybi(z::Number) = _airybi(float(z))

_airybi(x::Float16) = Float16(_airybi(Float32(x)))
_airybi(z::ComplexF16) = ComplexF16(_airybi(ComplexF32(z)))

function _airybi(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        isnan(z) && return z
        isinf(z) && return throw(DomainError(z, "airyai(z) not defined at infinity"))
    end
    x, y = real(z), imag(z)
    zabs = abs(z)

    airy_large_argument_cutoff(z) && return airybi_large_argument(z)

    airybi_power_series_cutoff(x, y) && return airybi_power_series(z)

    if x > zero(T)
        zz = 2 * z^(T(3)/2) / 3
        shift = 20
        order = one(T)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return sqrt(z/3) * (inu + inu2)
    else
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs^(T(3)/2) / 3
            Jv, Yv = besseljy_positive_args(one(T)/3, xx)
            Jmv = (Jv - sqrt(T(3)) * Yv) / 2
            return convert(eltype(z), sqrt(xabs/3) * (Jmv - Jv))
        else
            return exp(pi*im/3) * _airybi(-z*exp(pi*im/3))  + exp(-pi*im/3) * _airybi(-z*exp(-pi*im/3))
        end
    end
end

"""
    airybiprime(x)
Derivative of the Airy function of the second kind ``\\operatorname{Bi}'(x)``.
"""

airybiprime(z::Number) = _airybiprime(float(z))

_airybiprime(x::Float16) = Float16(_airybiprime(Float32(x)))
_airybiprime(z::ComplexF16) = ComplexF16(_airybiprime(ComplexF32(z)))

function _airybiprime(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    x, y = real(z), imag(z)
    zabs = abs(z)

    airy_large_argument_cutoff(z) && return airybiprime_large_argument(z)

    airybi_power_series_cutoff(x, y) && return airybiprime_power_series(z)

    if x > zero(T)
        zz = 2 * z^(T(3)/2) / 3
        shift = 20
        order = T(2)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return z / sqrt(3) * (inu + inu2)
    else
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs^(T(3)/2) / 3
            Jv, Yv = besseljy_positive_args(T(2)/3, xx)
            Jmv = -(Jv + sqrt(T(3))*Yv) / 2
            return convert(eltype(z), xabs * (Jv + Jmv) / sqrt(T(3)))
        else
            return -(exp(2pi*im/3)*_airybiprime(-z*exp(pi*im/3)) + exp(-2pi*im/3)*_airybiprime(-z*exp(-pi*im/3)))
        end
    end
end

#####
##### Power series for airyai(x)
#####

# cutoffs for power series valid for both airyai and airyaiprime
airyai_power_series_cutoff(x::T, y::T) where T <: Float64 = x < 2 && abs(y) > -1.4*(x + 5.5)
airyai_power_series_cutoff(x::T, y::T) where T <: Float32 = x < 4.5f0 && abs(y) > -1.4f0*(x + 9.5f0)

function airyai_power_series(x::ComplexOrReal{T}) where T
    S = eltype(x)
    iszero(x) && return S(0.3550280538878172)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_TWO_THIRDS(T)
    t2 = 3*x / GAMMA_ONE_THIRD(T)
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < eps(T) * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(2)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(4)/3))
    end
    return (ai1*3^(-T(2)/3) - ai2*3^(-T(4)/3))
end

#####
##### Power series for airyaiprime(x)
#####

function airyaiprime_power_series(x::ComplexOrReal{T}) where T
    S = eltype(x)
    iszero(x) && return S(-0.2588194037928068)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_ONE_THIRD(T)
    t2 = 3*x2 / (2*GAMMA_TWO_THIRDS(T))
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < eps(T) * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(1)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(5)/3))
    end
    return -ai1*3^(-T(1)/3) + ai2*3^(-T(5)/3)
end

#####
##### Power series for airybi(x)
#####

# cutoffs for power series valid for both airybi and airybiprime
# has a more complicated validity as it works well close to positive real line and for small negative arguments also works for angle(z) ~ 2pi/3
# the statements are somewhat complicated but we absolutely want to hit this branch when we can as the other algorithms are 10x slower
# the Float32 cutoff can be simplified because it overlaps with the large argument expansion so there isn't a need for more complicated statements
airybi_power_series_cutoff(x::T, y::T) where T <: Float64 = (abs(y) > -1.4*(x + 5.5) && abs(y) < -2.2*(x - 4)) || (x > zero(T) && abs(y) < 3)
airybi_power_series_cutoff(x::T, y::T) where T <: Float32 = abs(complex(x, y)) < 9

function airybi_power_series(x::ComplexOrReal{T}) where T
    S = eltype(x)
    iszero(x) && return S(0.6149266274460007)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_TWO_THIRDS(T)
    t2 = 3*x / GAMMA_ONE_THIRD(T)
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < eps(T) * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(2)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(4)/3))
    end
    return (ai1*3^(-T(1)/6) + ai2*3^(-T(5)/6))
end
function airybiprime_power_series(x::ComplexOrReal{T}) where T
    S = eltype(x)
    iszero(x) && return S(0.4482883573538264)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_ONE_THIRD(T)
    t2 = 3*x2 / (2*GAMMA_TWO_THIRDS(T))
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < eps(T) * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(1)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(5)/3))
    end
    return (ai1*3^(T(1)/6) + ai2*3^(-T(7)/6))
end

#####
#####  Large argument expansion for airy functions
#####
airy_large_argument_cutoff(z::ComplexOrReal{Float64}) = abs(z) > 8
airy_large_argument_cutoff(z::ComplexOrReal{Float32}) = abs(z) > 4

function airyai_large_argument(x::Real)
    x < zero(x) && return real(airyai_large_argument(complex(x)))
    return airy_large_arg_a(abs(x))
end
function airyai_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    a = airy_large_arg_a(z)
    if x < zero(T) && abs(y) < 5.5
        b = airy_large_arg_b(z)
        y >= zero(T) ? (return a + im*b) : (return a - im*b)
    end
    return a
end

function airyaiprime_large_argument(x::Real)
    x < zero(x) && return real(airyaiprime_large_argument(complex(x)))
    return airy_large_arg_c(abs(x))
end
function airyaiprime_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    c = airy_large_arg_c(z)
    if x < zero(T) && abs(y) < 5.5
        d = airy_large_arg_d(z)
        y >= zero(T) ? (return c + im*d) : (return c - im*d)
    end
    return c
end

function airybi_large_argument(x::Real)
    if x < zero(x)
        return 2*real(airy_large_arg_b(complex(x)))
    else
        return 2*(airy_large_arg_b(x))
    end
end

function airybi_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    b = airy_large_arg_b(z)
    abs(y) <= 1.7*(x - 6) && return 2*b

    check_conj = false
    if y < zero(T)
        z = conj(z)
        b = conj(b)
        y = abs(y)
        check_conj = true
    end

    a = airy_large_arg_a(z)
    if x < zero(T) && y < 5
        out = b + im*a
        check_conj && (out = conj(out))
        return out
    else
        out = 2*b + im*a
        check_conj && (out = conj(out))
        return out
    end
end
function airybiprime_large_argument(x::Real)
    if x < zero(x)
        return 2*real(airy_large_arg_d(complex(x)))
    else
        return 2*(airy_large_arg_d(x))
    end
end
function airybiprime_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    d = airy_large_arg_d(z)
    abs(y) <= 1.7*(x - 6) && return 2*d

    check_conj = false
    if y < zero(T)
        z = conj(z)
        d = conj(d)
        y = abs(y)
        check_conj = true
    end

    c = airy_large_arg_c(z)
    if x < zero(T) && y < 5
        out = d + im*c
        check_conj && (out = conj(out))
        return out
    else
        out = 2*d + im*c
        check_conj && (out = conj(out))
        return out
    end
end

function airy_large_arg_a(x::ComplexOrReal{T}) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = gamma(one(T) / 6) * gamma(T(5) / 6) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < eps(T)*50 * abs(out) && break
        t *= -3*(i + one(T)/6) * (i + T(5)/6) / (a*(i + one(T)))
    end
    return out * exp(-a / 6) / (pi^(3/2) * sqrt(xsqr))
end

function airy_large_arg_b(x::ComplexOrReal{T}) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = gamma(one(T) / 6) * gamma(T(5) / 6) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < eps(T)*50 * abs(out) && break
        t *= 3*(i + one(T)/6) * (i + T(5)/6) / (a*(i + one(T)))
    end
    return out * exp(a / 6) / (pi^(3/2) * sqrt(xsqr))
end

function airy_large_arg_c(x::ComplexOrReal{T}) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = gamma(-one(T) / 6) * gamma(T(7) / 6) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < eps(T)*50* abs(out) && break
        t *= -3*(i - one(T)/6) * (i + T(7)/6) / (a*(i + one(T)))
    end
    return out * exp(-a / 6) * sqrt(xsqr) / pi^(3/2)
end

function airy_large_arg_d(x::ComplexOrReal{T}) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = gamma(-one(T) / 6) * gamma(T(7) / 6) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < eps(T)*50 * abs(out) && break
        t *= 3*(i - one(T)/6) * (i + T(7)/6) / (a*(i + one(T)))
    end
    return -out * exp(a / 6) * sqrt(xsqr) / pi^(3/2)
end

# calculates besselk from the power series of besseli using the continued fraction and wronskian
# this shift the order higher first to avoid cancellation in the power series of besseli along the imaginary axis
# for real arguments this is not needed because besseli can be computed stably over the entire real axis
function besselk_continued_fraction_shift(nu, x)
    shift = 20
    inu, inum1 = besseli_power_series_inu_inum1(nu + shift, x)
    inu, inum1 = besselk_down_recurrence(x, inum1, inu, nu + shift - 1, nu)
    H_knu = besselk_ratio_knu_knup1(nu-1, x)
    return 1 / (x * (inum1 + inu / H_knu))
end
