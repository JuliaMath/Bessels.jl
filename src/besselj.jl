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

function besselj_large_argument(v, x::T) where T
    μ = 4 * v^2
    s0 = 1
    s1 = (1 - μ) / 8
    s2 = (-μ^2 + 26μ - 25) / 128
    s3 = (-μ^3 + 115μ^2 - 1187μ + 1073) / 1024
    s4 = (-5μ^4 + 1540μ^3 - 56238μ^2 + 430436μ - 375733) / 32768
    s5 = (-7μ^5 + 4515μ^4 - 397190μ^3 + 9716998μ^2 - 64709091μ + 55384775) / 262144
    s6 = (-21μ^6 + 24486μ^5 - 4238531μ^4 + 235370036μ^3 - 4733751627μ^2 + 29215626566μ - 24713030909) / 4194304
    s7 = (-33μ^7 + 63063μ^6 - 18939349μ^5 + 1989535379μ^4 - 87480924739μ^3 + 1573356635461μ^2 - 9268603618823μ + 7780757249041) / 33554432
    s8 = (-429μ^8 + 1252680μ^7 - 598859404μ^6 + 106122595896μ^5 - 8593140373614μ^4 + 329343318168440μ^3 - 5517359285625804μ^2 + 31505470994964360μ - 26308967412122125) / 2147483648
    s9 = (-715μ^9 + 3026595μ^8 - 2163210764μ^7 + 597489288988μ^6 - 79979851361130μ^5 + 5536596631240042μ^4 - 194026764558396188μ^3 + 3095397360215483916μ^2 - 17285630127691126691μ + 14378802319925055947) / 17179869184
    s10 = (-2431μ^10 + 14318590μ^9 - 14587179035μ^8 + 5925778483368μ^7 - 1216961874423502μ^6 + 137164402798287604μ^5 - 8556293060689145118μ^4 + 281576004385356401192μ^3 - 4334421752432088249971μ^2 + 23801928703130666089534μ - 19740662615375374580231) / 274877906944
    s11 = (-4199μ^11 + 33302269μ^10 - 46575001473μ^9 + 26616800349043μ^8 - 7938052610367302μ^7 + 1355921534379626242μ^6 - 136097258378143928354μ^5 + 7882145023568373727078μ^4 - 247560495216085669748963μ^3 + 3708133715826433118900337μ^2 - 20097626064642945009122253μ + 16629305448257355302267575) / 2199023255552
    s12 = (-29393μ^12 + 305569628μ^11 - 569135407698μ^10 + 441734362333708μ^9 - 183408097342762543μ^8 + 45038004674954790968μ^7 - 6784356890740220944060μ^6 + 626292216963766324300664μ^5 - 34303895772550410316410943μ^4 + 1039206680182498322210164588μ^3 - 15233583205322702897380357650μ^2 + 81699279492428006742420030332μ - 67471218624230362526181277601) / 70368744177664


    αp = s0 + s1/x^2 + s2/x^4 + s3/x^6 + s4/x^8 + s5/x^10 + s6/x^12 + s7/x^14 + s8/x^16 + s9/x^18 + s10/x^20 + s11/x^22 + s12/x^24
    α = s0*x - s1/x - s2/3x^3 - s3/5x^5 - s4/7x^7 - s5/9x^9 - s6/11x^11 - s7/13x^13 - s8/15x^15 - s9/17x^17 - s10/19x^19 - s11/21x^21 - s12/23x^23

    α = α - T(pi)/4 - T(pi)/2 * v

    b = sqrt(2 / T(pi)) / sqrt(αp * x)
    return cos(α)*b
end

function besselj_small_arguments_orders(v, x::T) where T
    MaxIter = 100
    out = zero(T)
    a = (x/2)^v / factorial(v)
    t2 = (x/2)^2
    for i in 1:MaxIter
        out += a
        abs(a) < eps(T) * abs(out) && break
        a = -a / (v + i) * inv(i) * t2
    end
    return out
end
