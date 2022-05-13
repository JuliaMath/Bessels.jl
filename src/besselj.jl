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

function _α_αp_asymptotic(v, x::T) where T
    μ = 4 * v^2
    s0 = 1
    s1 = (1 - μ) / 8
    s2 = evalpoly(μ, (-25, 26, -1)) / 128
    s3 = evalpoly(μ, (1073, -1187, 115, -1)) / 1024
    s4 = evalpoly(μ, (-375733, 430436, -56238, 1540, -5)) / 32768
    s5 = evalpoly(μ, (55384775, -64709091, 9716998, -397190, 4515, -7)) / 262144
    s6 = evalpoly(μ, (-24713030909, 29215626566, -4733751627, 235370036, -4238531, 24486, -21)) / 4194304

    if x < 15*v
        s7 = evalpoly(μ, (7780757249041, -9268603618823, 1573356635461, -87480924739, 1989535379, -18939349, 63063, -33)) / 33554432
        s8 = evalpoly(μ, (-26308967412122125, 31505470994964360, -5517359285625804, 329343318168440, -8593140373614, 106122595896, -598859404, 1252680, -429)) / 2147483648
        s9 = evalpoly(μ, (14378802319925055947, -17285630127691126691, 3095397360215483916, -194026764558396188, 5536596631240042, -79979851361130, 597489288988, -2163210764, 3026595, -715)) / 17179869184
        s10 = evalpoly(μ, (-19740662615375374580231, 23801928703130666089534, -4334421752432088249971, 281576004385356401192, -8556293060689145118, 137164402798287604, -1216961874423502, 5925778483368, -14587179035, 14318590, -2431)) / 274877906944
        s11 = evalpoly(μ, (16629305448257355302267575, -20097626064642945009122253, 3708133715826433118900337, -247560495216085669748963, 7882145023568373727078, -136097258378143928354, 1355921534379626242, -7938052610367302, 26616800349043, -46575001473, 33302269, -4199)) / 2199023255552
        s12 = evalpoly(μ, (-67471218624230362526181277601, 81699279492428006742420030332, -15233583205322702897380357650, 1039206680182498322210164588, -34303895772550410316410943, 626292216963766324300664, -6784356890740220944060, 45038004674954790968, -183408097342762543, 441734362333708, -569135407698, 305569628, -29393)) / 70368744177664
        s13 = evalpoly(μ, (81121053093143918050102987600099,  -98383136980007985475571939027975, 18503851640550924693860888790250, -1284587641197372537980918233562, 43642612323123398148213493825, -832261743620394930038568309, 9606500048207046619311900, -69893489053616393418780, 325411350127146969525, -959952773579760385, 1717262634951770, -1676314996330, 692939975, -52003)) / 562949953421312

        αp = evalpoly((1/x)^2, (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13))
        α = x * (evalpoly((1/x)^2, (s0, -s1, -s2/3, -s3/5, -s4/7, -s5/9, -s6/11, -s7/13, -s8/15, -s9/17, -s10/19, -s11/21, -s12/23, -s13/25)))
    else
        αp = evalpoly((1/x)^2, (s0, s1, s2, s3, s4, s5, s6))
        α = x * (evalpoly((1/x)^2, (s0, -s1, -s2/3, -s3/5, -s4/7, -s5/9, -s6/11)))
    end
    α = α - T(pi)/4 - T(pi)/2 * v
    b = sqrt(2 / T(pi)) / sqrt(αp * x)

    return α, b
end
function besselj_large_argument(v, x::T) where T
    α, b = _α_αp_asymptotic(v, x)
    return cos(α)*b
end
function bessely_large_argument(v, x::T) where T
    α, b = _α_αp_asymptotic(v, x)
    return sin(α)*b
end


# only valid in non-oscillatory regime (v>1/2, 0<t<sqrt(v^2 - 0.25))
function besselj_small_arguments_orders(v, x::T) where T
    MaxIter = 1_000
    out = zero(T)
    a = (x/2)^v / gamma(v + one(T))
    t2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        abs(a) < eps(T) * abs(out) && break
        a = -a * inv((v + i + one(T)) * (i + one(T))) * t2
    end
    return out
end

# perhaps use when v is small i believe v also has to be positive for this to work
# need for bessely 
function log_besselj_small_arguments_orders(v, x::T) where T
    MaxIter = 200
    out = zero(T)
    a   = one(T)
    x2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        a = -a * x2 * inv((i + one(T)) * (v + i + one(T)))
        #(abs(a) < eps(T) * abs(out)) && break
    end
    logout = -loggamma(v + 1) + v * log(x/2) + log(out)
    return logout
end

# valid when x < v (uniform asymptotic expansions)
function besselj_debye(v, x)
    T = eltype(x)
    S = promote_type(T, Float64)
    x = S(x)
    b = sqrt(v^2 - x^2)
    n = v * log(v/x + sqrt((v/x)^2 - 1)) - b
    coef = inv(sqrt(2*T(pi))) * exp(-n) / sqrt(b)
    p = v / sqrt(v^2 - x^2)
    p2  = v^2/fma(v,v, -x^2)

    return coef * Uk_poly_Jn(p, v, p2, T)
end
