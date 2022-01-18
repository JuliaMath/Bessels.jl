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
#    At this point as nu -> Inf it is more efficient to use a uniform expansion.
#    The boundary of the expansion depends on the accuracy required.
#    For more information see [4]. This approach is not yet implemented, so recurrence
#    is used for all values of nu.
#
# 
# [1] "Rational Approximations for the Modified Bessel Function of the Second Kind 
#     - K0(x) for Computations with Double Precision" by Pavel Holoborodko     
# [2] "Rational Approximations for the Modified Bessel Function of the Second Kind 
#     - K1(x) for Computations with Double Precision" by Pavel Holoborodko
# [3] https://github.com/boostorg/math/tree/develop/include/boost/math/special_functions/detail
# [4] "Computation of Bessel Functions of Complex Argument and Large ORder" by Donald E. Amos
#      Sandia National Laboratories
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
Recurrence is used for all values of nu. If besselk0(x) or besselk1(0) is equal to zero
this will underflow and always return zero even if besselk(x, nu)
is larger than the smallest representing floating point value.
In other words, for large values of x and small to moderate values of nu,
this routine will underflow to zero.
For small to medium values of x and large values of nu this will overflow and return Inf.

In the future, a more efficient algorithm for large nu should be incorporated.
=#
"""
    besselk(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order nu, ``K_{nu}(x)``.
"""
function besselk(nu::Int, x::T) where T <: Union{Float32, Float64}
    if nu < 100
        return three_term_recurrence(x, besselk0(x), besselk1(x), nu)
    else
        return k1(BigFloat(nu), BigFloat(x))
    end
end
"""
    besselk(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the second kind of order nu, ``K_{nu}(x)*e^{x}``.
"""
function besselkx(nu::Int, x::T) where T <: Union{Float32, Float64}
    return three_term_recurrence(x, besselk0x(x), besselk1x(x), nu)
end

@inline function three_term_recurrence(x, k0, k1, nu)
    nu == 0 && return k0
    nu == 1 && return k1

    # this prevents us from looping through large values of nu when the loop will always return zero
    (iszero(k0) || iszero(k1)) && return zero(x) 

    k2 = k0
    x2 = 2 / x
    for n in 1:nu-1
        a = x2 * n
        k2 = muladd(a, k1, k0)
        k0 = k1
        k1 = k2
    end
    return k2
end


function k1(v, x::T) where T <: AbstractFloat
    z = x / v
    zs = sqrt(1 + z^2)
 
    n = zs + log(z / (1 + zs))
    coef = exp(-v * n) * sqrt(pi / (2*v)) / sqrt(zs)
 
    p = inv(zs)
 
    u0 = one(x)
    u1 = p / 24 * (3 - 5*p^2) * -1 / v
    u2 = p^2 / 1152 * (81 - 462*p^2 + 385*p^4) / v^2
    u3 = p^3 / 414720 * (30375 - 369603 * p^2 + 765765*p^4 - 425425*p^6) * -1 / v^3
    u4 = p^4 / 39813120 * (4465125 - 94121676*p^2 + 349922430*p^4 - 446185740*p^6 + 185910725*p^8) / v^4
    u5 = p^5 / 6688604160 * (-188699385875*p^10 + 566098157625*p^8 - 614135872350*p^6 + 284499769554*p^4 - 49286948607*p^2 + 1519035525) * -1 / v^5
    u6 = p^6 / 4815794995200 * (1023694168371875*p^12 - 3685299006138750*p^10 + 5104696716244125*p^8 - 3369032068261860*p^6 + 1050760774457901*p^4 - 127577298354750*p^2 + 2757049477875) * 1 / v^6
    u7 = p^7 / 115579079884800 * (-221849150488590625*p^14 + 931766432052080625*p^12 - 1570320948552481125*p^10 + 1347119637570231525*p^8 - 613221795981706275*p^6 + 138799253740521843*p^4 - 12493049053044375*p^2 + 199689155040375) * -1 / v^7
    u8 = p^8 / 22191183337881600 * (448357133137441653125*p^16 - 2152114239059719935000*p^14 + 4272845805510421639500*p^12 - 4513690624987320777000*p^10 + 2711772922412520971550*p^8 - 914113758588905038248*p^6 + 157768535329832893644*p^4 - 10960565081605263000*p^2 + 134790179652253125) * 1 / v^8
    u9 = p^9 / 263631258054033408000 * (-64041091111686478524109375*p^18 + 345821892003106984030190625*p^16 - 790370708270219620781737500*p^14 + 992115946599792610768672500*p^12 - 741743213039573443221773250*p^10 + 334380732677827878090447630*p^8 - 87432034049652400520788332*p^6 + 11921080954211358275362500*p^4 - 659033454841709672064375*p^2 + 6427469716717690265625) * -1 / v^9
    u10 = p^10 / 88580102706155225088000 * (290938676920391671935028890625*p^20 - 1745632061522350031610173343750*p^18 + 4513386761946134740461797128125*p^16 - 6564241639632418015173104205000*p^14 + 5876803711285273203043452095250*p^12 - 3327704366990695147540934069220*p^10 + 1177120360439828012193658602930*p^8 - 246750339886026017414509498824*p^6 + 27299183373230345667273718125*p^4 - 1230031256571145165088463750*p^2 + 9745329584487361980740625) * 1 / v^10
    out1 = ((u10 + u9) + u8) + u7
    out2 = ((out1 + u6 + u5) + u4) + u3
    out = ((out2 + u2) + u1) + u0
 
    return coef*out
 end
 