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
airyai(x::Real) = _airyai(float(x))

_airyai(x::Float16) = Float16(_airyai(Float32(x)))

function _airyai(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? zero(T) : sqrt(x / 3) * besselk(one(T)/3, z) / π
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(one(T)/3, z)
        Jmv = (Jv - sqrt(T(3)) * Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airyai(x) is not defined.")) : sqrt(x_abs) * (Jmv + Jv) / 3
    elseif iszero(x)
        return T(0.3550280538878172)
    end
end

"""
    airyaiprime(x)
Derivative of the Airy function of the first kind ``\\operatorname{Ai}'(x)``.
"""
airyaiprime(x::Real) = _airyaiprime(float(x))

_airyaiprime(x::Float16) = Float16(_airyaiprime(Float32(x)))

function _airyaiprime(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? zero(T) : -x * besselk(T(2)/3, z) / (sqrt(T(3)) * π)
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(T(2)/3, z)
        Jmv = -(Jv + sqrt(T(3))*Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airyaiprime(x) is not defined.")) : x_abs * (Jv - Jmv) / 3
    elseif iszero(x)
        return T(-0.2588194037928068)
    end
end

"""
    airybi(x)
Airy function of the second kind ``\\operatorname{Bi}(x)``.
"""
airybi(x::Real) = _airybi(float(x))

_airybi(x::Float16) = Float16(_airybi(Float32(x)))

function _airybi(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? z : sqrt(x / 3) * (besseli(-one(T)/3, z) + besseli(one(T)/3, z))
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(one(T)/3, z)
        Jmv = (Jv - sqrt(T(3)) * Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airybi(x) is not defined.")) : sqrt(x_abs/3) * (Jmv - Jv)
    elseif iszero(x)
        return T(0.6149266274460007)
    end
end

"""
    airybiprime(x)
Derivative of the Airy function of the second kind ``\\operatorname{Bi}'(x)``.
"""
airybiprime(x::Real) = _airybiprime(float(x))

_airybiprime(x::Float16) = Float16(_airybiprime(Float32(x)))

function _airybiprime(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? z : x * (besseli(T(2)/3, z) + besseli(-T(2)/3, z)) / sqrt(T(3))
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(T(2)/3, z)
        Jmv = -(Jv + sqrt(T(3))*Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airybiprime(x) is not defined.")) : x_abs * (Jv + Jmv) / sqrt(T(3))
    elseif iszero(x)
        return T(0.4482883573538264)
    end
end

GAMMA_TWO_THIRDS(::Type{Float64}) = 1.3541179394264005
GAMMA_ONE_THIRD(::Type{Float64}) = 2.6789385347077475
GAMMA_ONE_THIRD(::Type{T}) where T <: AbstractFloat = T(big"2.678938534707747633655692940974677644128689377957301100950428327590417610167733")
GAMMA_TWO_THIRDS(::Type{T}) where T <: AbstractFloat = T(big"1.354117939426400416945288028154513785519327266056793698394022467963782965401746")



const ComplexOrReal{T} = Union{T,Complex{T}}
# returns both (airyai, airyaiprime) using the power series definition for small arguments

function airyai2(z::ComplexOrReal{T}) where T
    x, y = real(z), imag(z)
    zabs = abs(z)

    if zabs > 8
        a = airy_large_arg_a(z)
        if x < zero(T) && abs(y) < 5.5
            b = airy_large_arg_b(z)
            y >= zero(T) ? (return a + im*b) : (return a - im*b)
        end
        return a
    elseif x < 2 && abs(y) > -1.4*(x + 5.5)
        return airyai_power_series(z)
    elseif x > zero(T)
        zz = 2 * z^(T(3)/2) / 3
        
        return sqrt(z / 3) * besselk_continued_fraction_shift(one(T)/3, zz) / π
    else
        return exp(pi*im/3) * airyai2(-z*exp(pi*im/3))  + exp(-pi*im/3) * airyai2(-z*exp(-pi*im/3))
    end
end

function airyaiprime2(z::ComplexOrReal{T}) where T
    x, y = real(z), imag(z)
    zabs = abs(z)

    if zabs > 8
        c = airy_large_arg_c(z)
        if x < zero(T) && abs(y) < 5.5
            d = airy_large_arg_d(z)
            y >= zero(T) ? (return c + im*d) : (return c - im*d)
        end
        return c
    elseif x < 2 && abs(y) > -1.4*(x + 5.5)
        return airyaiprime_power_series(z)
    elseif x > zero(T)
        zz = 2 * z^(T(3)/2) / 3
        return -z * besselk_continued_fraction_shift(T(2)/3, zz) / (π * sqrt(T(3)))
    else
        return -(exp(2pi*im/3)*airyaiprime(-z*exp(pi*im/3)) + exp(-2pi*im/3)*airyaiprime(-z*exp(-pi*im/3)))
    end
end


function airybi2(z::ComplexOrReal{T}) where T
    x, y = real(z), imag(z)
    zabs = abs(z)
    if zabs > 9
        b = airy_large_arg_b(z)
        if abs(y) <= 1.7*(x - 6)
            return 2*b
        else
            a = airy_large_arg_a(z)
            if x < zero(T)
                if abs(y) < 5
                    if y < zero(T)
                        return b - im*a
                    else
                        return b + im*a
                    end
                end
            end
            if y < zero(T)
                return 2*b - im*a
            else
                return 2*b + im*a
            end
        end
    elseif x < 2 && abs(y) > -1.4*(x + 5.5)
        return airybi_power_series(z)
    elseif x > 2 && abs(y) > 3
        zz = 2 * z^(T(3)/2) / 3
        shift = 20
        order = one(T)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return sqrt(z/3) * (inu + inu2)
    else
        return exp(pi*im/3) * airybi2(-z*exp(pi*im/3))  + exp(-pi*im/3) * airybi2(-z*exp(-pi*im/3))
    end

end

function airybiprime2(z::ComplexOrReal{T}) where T
    x, y = real(z), imag(z)
    zabs = abs(z)
    if zabs > 9
        d = airy_large_arg_d(z)
        if abs(y) <= 1.7*(x - 6)
            return 2*d
        else
            c = airy_large_arg_c(z)
            if x < zero(T)
                if abs(y) < 5
                    if y < zero(T)
                        return d - im*c
                    else
                        return d + im*c
                    end
                end
            end
            if y < zero(T)
                return 2*d - im*c
            else
                return 2*d + im*c
            end
        end
    elseif x < 0 && abs(y) < -1.4*(x + 5.5)
        return -(exp(2pi*im/3)*airybiprime(-z*exp(pi*im/3)) + exp(-2pi*im/3)*airybiprime(-z*exp(-pi*im/3)))
    elseif x > 2 && abs(y) > 3
        zz = 2 * z^(T(3)/2) / 3
        shift = 20
        order = T(2)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return z / sqrt(3) * (inu + inu2)
    else 
        return airybiprime_power_series(z)
    end
end

function airyai_power_series(x::ComplexOrReal{T}) where T
    S = eltype(x)
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
function airyaiprime_power_series(x::ComplexOrReal{T}) where T
    S = eltype(x)
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
function airybi_power_series(x::ComplexOrReal{T}) where T
    S = eltype(x)
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
