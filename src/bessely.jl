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
        xinv = inv(x)
        x2 = xinv*xinv
        if x < 130.0
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
        xn = muladd(xinv, q, - 3 * PIO4(T))

        # the following computes b = sin(x + xn) more accurately
        # see src/misc.jl
        b = sin_sum(x, xn)
        return a * b
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

#              Bessel functions of the second kind of order nu
#                           bessely(nu, x)
#
#    A numerical routine to compute the Bessel function of the second kind Y_{ν}(x) [1]
#    for real orders and arguments of positive or negative value. The routine is based on several
#    publications [2, 3, 4, 5] that calculate Y_{ν}(x) for positive arguments and orders where
#    reflection identities are used to compute negative arguments and orders.
#
#    In particular, the reflectance identities for negative integer orders Y_{-n}(x) = (-1)^n * Y_{n}(x) (Eq. 9.1.5; [6])
#    and for negative noninteger orders Y_{−ν}(x) = cos(πν) * Y_{ν}(x) + sin(πν) * J_{ν}(x) are used.
#    For negative arguments of integer order, Y_{n}(-x) = (-1)^n * Y_{n}(x) + (-1)^n * 2im * J_{n}(x) is used and for
#    noninteger orders, Y_{ν}(−x) = exp(−im*π*ν) * Y_{ν}(x) + 2im * cos(πν) * J_{ν}(x) is used.
#    For negative orders and arguments the previous identities are combined.
#
#    The identities are computed by calling the `bessely_positive_args(nu, x)` function which computes Y_{ν}(x)
#    for positive arguments and orders. For integer orders up to 250, forward recurrence is used starting from
#    `bessely0` and `bessely1` routines for calculation of Y_{n}(x) of the zero and first order.
#    For small arguments, Y_{ν}(x) is calculated from the power series (`bessely_power_series(nu, x`) form of J_{ν}(x) using the connection formula [1].
#    
#    When x < ν and ν being reasonably large, the debye asymptotic expansion (Eq. 33; [3]) is used `besseljy_debye(nu, x)`.
#    When x > ν and x being reasonably large, the Hankel function is calculated from the debye expansion (Eq. 29; [3]) with `hankel_debye(nu, x)`
#    and Y_{n}(x) is calculated from the imaginary part of the Hankel function.
#    These expansions are not uniform so are not strictly used when the above inequalities are true, therefore, cutoffs
#    were determined depending on the desired accuracy. For large arguments x >> ν, the phase functions are used (Eq. 15 [4]) with `besseljy_large_argument(nu, x)`.
#
#    For values where the expansions for large arguments and orders are not valid, forward recurrence is employed after shifting the order down
#    to where these expansions are valid then using recurrence. In general, the routine will be the slowest when ν ≈ x as all methods struggle at this point.
#    Additionally, the Hankel expansion is only accurate (to Float64 precision) when x > 19 and the power series can only be computed for x < 7 
#    without loss of precision. Therefore, when x > 19 the Hankel expansion is used to generate starting values after shifting the order down.
#    When x ∈ (7, 19) and ν ∈ (0, 2) Chebyshev approximation is used with higher orders filled by recurrence.
#    
# [1] https://dlmf.nist.gov/10.2#E3
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
##### Generic routine for `bessely`
#####

"""
    bessely(nu, x::T) where T <: Float64

Bessel function of the second kind of order nu, ``Y_{nu}(x)``.
nu and x must be real where nu and x can be positive or negative.
"""
bessely(nu::Real, x::Real) = _bessely(nu, float(x))

_bessely(nu, x::Float16) = Float16(_bessely(nu, Float32(x)))

function _bessely(nu, x::T) where T <: Union{Float32, Float64}
    isnan(nu) || isnan(x) && return NaN
    isinteger(nu) && return _bessely(Int(nu), x)
    abs_nu = abs(nu)
    abs_x = abs(x)

    Ynu = bessely_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        if x >= zero(T)
            return T(Ynu)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Ynu * cispi(-nu) + 2im * besselj_positive_args(abs_nu, abs_x) * cospi(abs_nu)
        end
    else
        Jnu = besselj_positive_args(abs_nu, abs_x)
        spi, cpi = sincospi(abs_nu)
        if x >= zero(T)
            return T(Ynu * cpi + Jnu * spi)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return cpi * (Ynu * cispi(nu) + 2im * Jnu * cpi) + Jnu * spi * cispi(abs_nu)
        end
    end
end

function _bessely(nu::Integer, x::T) where T <: Union{Float32, Float64}
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    Ynu = bessely_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        if x >= zero(T)
            return T(Ynu)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Ynu * sg + 2im * sg * besselj_positive_args(abs_nu, abs_x)
        end
    else
        if x >= zero(T)
            return T(Ynu * sg)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Ynu + 2im * besselj_positive_args(abs_nu, abs_x)
        end
    end
end

#####
#####  `bessely` for positive arguments and orders
#####

"""
    bessely_positive_args(nu, x::T) where T <: Float64

Bessel function of the second kind of order nu, ``Y_{nu}(x)``.
nu and x must be real and nu and x must be positive.

No checks on arguments are performed and should only be called if certain nu, x >= 0.
"""
function bessely_positive_args(nu, x::T) where T
    iszero(x) && return -T(Inf)
    
    # use forward recurrence if nu is an integer up until it becomes inefficient
    (isinteger(nu) && nu < 250) && return besselj_up_recurrence(x, bessely1(x), bessely0(x), 1, nu)[1]

    # x < ~nu branch see src/U_polynomials.jl
    besseljy_debye_cutoff(nu, x) && return besseljy_debye(nu, x)[2]

    # large argument branch see src/asymptotics.jl
    besseljy_large_argument_cutoff(nu, x) && return besseljy_large_argument(nu, x)[2]

    # x > ~nu branch see src/U_polynomials.jl on computing Hankel function
    hankel_debye_cutoff(nu, x) && return imag(hankel_debye(nu, x))

    # use power series for small x and for when nu > x
    bessely_series_cutoff(nu, x) && return bessely_power_series(nu, x)[1]

    return bessely_fallback(nu, x)
end

#####
#####  Power series for Y_{nu}(x)
#####

# Use power series form of J_v(x) to calculate Y_v(x) with
# Y_v(x) = (J_v(x)cos(v*π) - J_{-v}(x)) / sin(v*π),    v ~= 0, 1, 2, ...
# combined to calculate both J_v and J_{-v} in the same loop
# J_{-v} always converges slower so just check that convergence
# Combining the loop was faster than using two separate loops
# 
# this works well for small arguments x < 7.0 for rel. error ~1e-14
# this also works well for nu > 1.35x - 4.5
# for nu > 25 more cancellation occurs near integer values
# There could be premature underflow when (x/2)^v == 0.
# It might be better to use logarithms (when we get loggamma julia implementation)
"""
    bessely_power_series(nu, x::T) where T <: Float64

Computes ``Y_{nu}(x)`` using the power series when nu is not an integer.
In general, this is most accurate for small arguments and when nu > x.
Outpus both (Y_{nu}(x), J_{nu}(x)).
"""
function bessely_power_series(v, x::T) where T
    MaxIter = 3000
    S = promote_type(T, Float64)
    v, x = S(v), S(x)

    out = zero(S)
    out2 = zero(S)
    a = (x/2)^v
    # check for underflow and return limit for small arguments
    iszero(a) && return (-T(Inf), a)

    b = inv(a)
    a /= gamma(v + one(S))
    b /= gamma(-v + one(S))
    t2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        out2 += b
        abs(b) < eps(T) * abs(out2) && break
        a *= -inv((v + i + one(S)) * (i + one(S))) * t2
        b *= -inv((-v + i + one(S)) * (i + one(S))) * t2
    end
    s, c = sincospi(v)
    return (out*c - out2) / s, out
end
bessely_series_cutoff(v, x::Float64) = (x < 7.0) || v > 1.35*x - 4.5
bessely_series_cutoff(v, x::Float32) = (x < 21.0) || v > 1.38*x - 12.5

#####
#####  Fallback for Y_{nu}(x)
#####

function bessely_fallback(nu, x)
    # for x ∈ (6, 19) we use Chebyshev approximation and forward recurrence
    if besseljy_chebyshev_cutoff(nu, x)
        return bessely_chebyshev(nu, x)[1]
    else
        # at this point x > 19.0 (for Float64) and fairly close to nu
        # shift nu down and use the debye expansion for Hankel function (valid x > nu) then use forward recurrence
        nu_shift = ceil(nu) - floor(Int, hankel_debye_fit(x)) + 4
        v2 = maximum((nu - nu_shift, modf(nu)[1] + 1))
        return besselj_up_recurrence(x, imag(hankel_debye(v2, x)), imag(hankel_debye(v2 - 1, x)), v2, nu)[1]
    end
end

#####
#####  Chebyshev approximation for Y_{nu}(x)
#####

"""
    bessely_chebyshev(nu, x::T) where T <: Float64

Computes ``Y_{nu}(x)`` for medium arguments x ∈ (6, 19) for any positive order using a Chebyshev approximation.
Forward recurrence is used to fill orders starting at low orders ν ∈ (0, 2).
"""
function bessely_chebyshev(v, x)
    v_floor, _ = modf(v)
    Y0, Y1 = bessely_chebyshev_low_orders(v_floor, x)
    return besselj_up_recurrence(x, Y1, Y0, v_floor + 1, v)
end

# only implemented for Float64 so far
besseljy_chebyshev_cutoff(::Number, x) = (x <= 19.0 && x >= 6.0)

# compute bessely for x ∈ (6, 19) and ν ∈ (0, 2) using chebyshev approximation with a (16, 28) grid (see notes to generate below)
# optimized to return both (ν, ν + 1) in around the same time, therefore ν must be in (0, 1)
# no checks are performed on arguments
function bessely_chebyshev_low_orders(v, x)
    # need to rescale inputs according to
    #x0 = (x - lb) * 2 / (ub - lb) - 1
    x1 = 2*(x - 6)/13 - 1
    v1 = v - 1
    v2 = v
    a = clenshaw_chebyshev.(x1, bessely_cheb_weights)
    return clenshaw_chebyshev(v1, a), clenshaw_chebyshev(v2, a)
end

# uses the Clenshaw algorithm to recursively evaluate a linear combination of Chebyshev polynomials
function clenshaw_chebyshev(x, c)
    x2 = 2x
    c0 = c[end-1]
    c1 = c[end]
    for i in length(c)-2:-1:1
        c0, c1 = c[i] - c1, c0 + c1 * x2
    end
    return c0 + c1 * x
end

# to generate the Chebyshev weights
#=
using ArbNumerics, FastChebInterp
g(x) = bessely(ArbFloat(x[1]), ArbFloat(x[2]));
lb, ub = [0,6], [2, 19]; # lower and upper bounds of the domain, respectively
x = chebpoints((16,28), lb, ub);
c = chebinterp(g.(x), lb, ub);
map(i->tuple(c.coefs[i,:]...), 1:size(c.coefs)[1])
=#
const bessely_cheb_weights = (
    (-0.030017698430846347, 0.04841599615543705, -0.06170112748210926, 0.010992867154582408, -0.03732304718455494, -0.06704282983696781, 0.05527054188949959, 0.023931299324571078, -0.01660982874247215, -0.0038257184915278875, 0.0024486563159734553, 0.00036760669126412974, -0.00022289024621774598, -2.400043085835714e-5, 1.3967165370379739e-5, 1.1481395619070928e-6, -6.449732666954525e-7, -4.1980803435719734e-8, 2.2894937763100794e-8, 1.228873366161081e-9, -6.497632451926297e-10, -2.8492149116234243e-11, 1.4818408405099196e-11, 6.100556567848311e-13, -2.9867774919978274e-13, -6.185528280054444e-15, 3.7311423791866865e-15, 4.123685520036296e-16, -1.3481279584734043e-16),
    (0.04160742192322324, 0.034668235600932615, 0.11174468340078995, -0.04099618095057443, 0.1254503646682523, -0.13898784797365799, -0.09733131946545694, 0.07149834579877713, 0.023375835755720973, -0.014572641168112577, -0.002981779009597293, 0.0016841968136507765, 0.0002422436246301404, -0.000127737004832643, -1.3814377122225202e-5, 6.9111332786206e-6, 5.866764647818481e-7, -2.8135569182003675e-7, -1.9403344167501224e-8, 8.972400084810467e-9, 5.113342130666102e-10, -2.2951010536509386e-10, -1.1304711135449809e-11, 4.905817815473565e-12, 1.894873147278986e-13, -8.217632923341561e-14, -4.218847493575595e-15, 1.6772297836301473e-15, -7.137148015447435e-17),
    (0.022713476918759194, -0.03501714644493984, 0.04900399934159489, -0.004572920361822816, 0.03414506477673644, 0.0583364038403724, -0.04624123224424829, -0.02357706101501925, 0.01424650161095809, 0.004131847507694573, -0.0021768978247202514, -0.0004222406947531938, 0.00020419469487529897, 2.882041410641021e-5, -1.3118879606205788e-5, -1.419547866831786e-6, 6.171197932015276e-7, 5.3215526542899864e-8, -2.2283397359466684e-8, -1.5689346295358737e-9, 6.370162334131838e-10, 3.785641839686755e-11, -1.4927226121841386e-11, -7.280852508197353e-13, 2.822365357297534e-13, 1.4096858601066386e-14, -5.194257722353411e-15, -1.784287003861859e-17, 1.784287003861859e-17),
    (-0.005482743163667869, -0.0026156864489958096, -0.013446106107341984, 0.003959772572751581, -0.013050739904566335, 0.012785648905104951, 0.012302789524159771, -0.007125881422379595, -0.0032178006311644504, 0.0015436443743600617, 0.0004291283948110706, -0.00018601906106350898, -3.5730026469563336e-5, 1.4521000123576475e-5, 2.0641877773886624e-6, -8.0134383976041e-7, -8.833134689166324e-8, 3.3105750509021536e-8, 2.9248427531622776e-9, -1.0652720751644758e-9, -7.761774457286969e-11, 2.758318276166857e-11, 1.6716350526024826e-12, -5.818832518996889e-13, -3.163144349623995e-14, 1.0744381574921492e-14, 3.598312124454748e-16, -1.2688263138573217e-16, -1.4373423086664974e-17),
    (-0.000821102524680051, 0.001369166415512963, -0.0017176445978307614, -0.00038167202153855176, -0.0010693704116036165, -0.0034162564467415902, 0.0019105668931819009, 0.0015608651654028757, -0.000662690872417737, -0.00028896692147194577, 0.00010910809251752316, 3.0258725355010946e-5, -1.0717669999955374e-5, -2.0862028639579696e-6, 7.091627814767861e-7, 1.0306000805131983e-7, -3.4017752731151496e-8, -3.856023535105163e-9, 1.2437339960318393e-9, 1.1358064900927889e-10, -3.594555586962962e-11, -2.7016821319796643e-12, 8.405911755350802e-13, 5.4115690103654197e-14, -1.660105584745862e-14, -8.187894806610529e-16, 2.462687791788503e-16, 2.0816681711721685e-17, -6.443258625056712e-18),
    (0.0002240588217758784, -2.139628782819116e-5, 0.0004963922147532327, -0.00011956525731319659, 0.00038470659301256783, -0.00022845214105494214, -0.0004692565509103452, 0.0001676797752522885, 0.0001335452722506947, -4.206397012022229e-5, -1.847619067631446e-5, 5.51981330361833e-6, 1.5583457267047952e-6, -4.525989246540121e-7, -9.035200300660385e-8, 2.5768522858261958e-8, 3.852458639733455e-9, -1.084135171755376e-9, -1.2724325563785672e-10, 3.5416192958546876e-11, 3.3285729083723912e-12, -9.174161771285542e-13, -7.329222174604474e-14, 2.0034423068009816e-14, 1.2002737593513098e-15, -3.2511254198458856e-16, -2.7507757976203656e-17, 6.753030674338285e-18, -4.336808689942018e-19),
    (4.6589086507212064e-6, -1.419986317236695e-5, 7.726102539451062e-6, 2.5019808838475857e-5, -5.353457546572625e-7, 7.876549687169565e-5, -1.996847202131808e-5, -4.169957016198162e-5, 9.924101147153856e-6, 8.2415996393497e-6, -1.9586527669761305e-6, -8.849155270353938e-7, 2.1145772481562803e-7, 6.146714709097454e-8, -1.4757833806872662e-8, -3.0347995540046756e-9, 7.310991241996968e-10, 1.1291045426976899e-10, -2.724231275323183e-11, -3.306873481836321e-12, 7.984081946667783e-13, 7.777996686699921e-14, -1.874952442498305e-14, -1.5555232495803793e-15, 3.750913580232083e-16, 2.4244503044553087e-17, -6.63977026881971e-18, -8.441288342922856e-19, 3.7947076036992655e-19),
    (-3.837876843131191e-6, 1.5306249581045373e-6, -7.808225833446716e-6, 8.447928863780004e-7, -4.570697567070479e-6, -4.1982552639485557e-7, 8.226238919809268e-6, -9.86994172463081e-7, -2.6319604894434756e-6, 4.2742958932228587e-7, 3.841104494715522e-7, -7.011130365317191e-8, -3.295984466784781e-8, 6.37554898520718e-9, 1.9212842748116095e-9, -3.851870296260733e-10, -8.157917257527438e-11, 1.6727904537992146e-11, 2.682049173124509e-12, -5.589600448593574e-13, -6.949400331178381e-14, 1.4635958089930247e-14, 1.5059865847402264e-15, -3.216721361622551e-16, -2.5000540452330927e-17, 5.001971562950145e-18, -1.961002277600456e-18, 9.647251501576167e-19, -6.969871108835386e-19),
    (1.576952254652365e-7, -9.540862698044819e-8, 3.3917615684812697e-7, -4.813870672453743e-7, 2.6408257938646016e-7, -8.568425702351827e-7, -1.2996848292204935e-7, 5.684699761362054e-7, -2.7544597243559516e-8, -1.247732596836196e-7, 1.4090830850669181e-8, 1.398591135642683e-8, -1.991545843477106e-9, -9.83479157367889e-10, 1.5579619154787597e-10, 4.8812239436949277e-11, -8.259978073280919e-12, -1.7977167324163993e-12, 3.156743262349288e-13, 5.3090323232950275e-14, -9.680314873025469e-15, -1.183751262421577e-15, 2.1703251434347915e-16, 2.5924059718442872e-17, -5.649951767599685e-18, 8.287120783728401e-19, -1.0219906276273448e-18, 2.5227968798668708e-18, -1.4714172340874703e-18),
    (2.8189276354316185e-8, -8.524800253702236e-9, 5.163074759458915e-8, 1.7587774571193398e-8, 1.5532689836449176e-8, 4.569379005731088e-8, -7.364565287663236e-8, -1.3794427678087878e-8, 2.8705869772299055e-8, -2.180238319299374e-10, -4.6124233142017096e-9, 3.6839126250828555e-10, 4.0853238049027397e-10, -4.58540375117209e-11, -2.4105895138400274e-11, 3.1690572156845862e-12, 1.025021315725097e-12, -1.4748687505667043e-13, -3.336472532024983e-14, 5.062095043025821e-15, 8.815484952439681e-16, -1.390034081631452e-16, -1.9199758208465145e-17, 2.8599694051079214e-18, 4.412739143428028e-19, -1.4432387356775567e-18, -2.3578466668818103e-18, -8.403587813975867e-19, -2.0135183203302226e-19),
    (-2.7963515253359716e-9, 2.6599405451754447e-9, -5.363225685883765e-9, 3.919112883175106e-9, -2.6445728931356353e-9, 3.986389181091895e-9, 4.369788563535614e-9, -4.202468942903132e-9, -8.620553821731379e-10, 1.1190715605385904e-9, 1.6458619897562414e-11, -1.364556955782257e-10, 7.770660894600997e-12, 9.797565423719047e-12, -8.709272762295716e-13, -4.951317701122077e-13, 5.5017193500686674e-14, 1.7973668773098524e-14, -2.186053134804191e-15, -5.417976610912264e-16, 7.265668497060572e-17, 1.0657145676262853e-17, -1.4951583811856726e-18, 1.001165055851074e-19, -4.215593642978497e-19, -8.996650316071422e-19, 1.1892060156355519e-18, -5.459522028374813e-19, -9.29316147844718e-19),
    (-2.869285397772589e-11, -1.425452662669203e-10, -1.4765859863811186e-11, -3.584266980792905e-10, 1.2630165211284162e-10, -4.813898813698495e-10, 2.9823341363420924e-10, 2.736833504070133e-10, -1.789940159530945e-10, -3.709803166716433e-11, 3.4844491459805023e-11, 8.72681373200755e-13, -3.2915591662608672e-12, 1.3002232883748757e-13, 1.9753949839973547e-13, -1.4075604115824882e-14, -8.601511614454014e-15, 8.194975088384275e-16, 2.6866096515073e-16, -2.6539953077529138e-17, -6.38116169641218e-18, -1.8536358293679684e-18, 4.749772369976817e-19, -6.718043393812506e-19, 1.608438964002761e-19, 5.165497356149306e-20, 2.605394963005647e-19, 1.0581028516295723e-19, 3.601100072898283e-19),
    (1.7263987678277334e-11, -1.3813366927208365e-11, 2.934247457063122e-11, -7.979165795689896e-12, 4.195660738851445e-12, 3.4756966192072763e-12, -3.8868044414248185e-11, 1.4360156231687025e-11, 1.2343845937660092e-11, -6.048594850472885e-12, -1.1764419064241353e-12, 8.840620586141411e-13, 2.5310500957954275e-14, -6.57823088425572e-14, 1.5812120648028614e-15, 3.437403053731452e-15, -2.0773399722137888e-16, -1.2502298901942747e-16, 1.1188335458126739e-17, 1.318770475529763e-18, 1.826317128659254e-18, -1.371135591032674e-18, -7.556266590585322e-19, -1.3319774595248458e-19, -6.907940207156658e-20, 1.774511101624311e-18, -4.537139020388408e-20, 4.949220345202049e-19, -1.1868141638100254e-18),
    (-8.498193197325338e-13, 1.953201750327066e-12, -1.6836406424518253e-12, 2.5267159459094905e-12, -1.1786713409725903e-12, 2.057320315641163e-12, 2.5591575645466365e-13, -2.0678475108903006e-12, 5.290174964157149e-13, 4.2270269001999225e-13, -1.680245826111033e-13, -2.838265511092047e-14, 1.8427577852297096e-14, 5.755842428415847e-16, -1.1124766162836294e-15, 1.08483878138819e-17, 5.233365672134661e-17, -4.1398608828316095e-18, -1.5252680555707005e-18, 1.5518391236648752e-21, -1.914160606977295e-18, -1.1560851262788287e-18, 8.614032803477796e-19, -2.4613082189127555e-18, 2.3715585034500812e-18, -7.375149035246488e-19, 1.5108762141152207e-18, -1.5506658311469656e-18, 8.17023779980148e-19),
    (-2.5827236864549892e-14, -3.4718790561504414e-14, -2.2011098530056326e-14, -9.110104031477491e-14, 7.290507373316453e-14, -1.234623611877002e-13, 1.6275850841628636e-13, 1.798444625683851e-14, -8.154482889879963e-14, 1.634453336575052e-14, 1.129456516347965e-14, -3.897439775077071e-15, -5.305238116730616e-16, 3.1481637373066765e-16, 1.2762608430395386e-17, -1.6640493419786607e-17, 4.907490184278107e-19, 3.285996490596493e-19, -1.6693969668826682e-18, -6.276996493483055e-19, 2.2721860225308528e-18, -1.0636166658032876e-18, 5.4043840077706323e-20, -6.733003923003965e-19, 8.654687466171472e-19, -1.6353891516146702e-18, -1.1401886941945345e-18, 1.806451000706121e-18, 1.0454806663253078e-18),
    (4.9388506677207545e-15, -8.11264980108463e-15, 8.15564695267678e-15, -6.6179014247901285e-15, 1.2560443562777774e-15, -1.1626485837362674e-15, -8.428647433288211e-15, 8.040878625052107e-15, 7.604076442794149e-16, -2.509940602859596e-15, 4.433411320740124e-16, 2.4051574424535735e-16, -7.97263854654364e-17, -8.468775456115063e-18, 3.0977204928150507e-18, -2.763609694292828e-18, -4.528057493615123e-18, 8.090667992714159e-18, -5.7670273029389466e-18, 6.935237082901993e-18, 2.3956539089710175e-18, 3.5297683311975285e-18, -2.6680862870119677e-19, -1.124331067925833e-18, -3.969891564639305e-18, 3.4988194829519685e-18, 8.636411459915624e-19, -1.3895984430563314e-18, 1.7657006809049643e-18),
    (-2.058125495426769e-16, 6.08114692374987e-16, -4.430737266654462e-16, 7.070791826861885e-16, -4.496672312473068e-16, 4.992680927678948e-16, -2.0005398723221536e-16, -4.085152288266179e-16, 2.963046065367326e-16, 1.9081397648228198e-17, -6.065778883208153e-17, 1.2267254817829564e-17, 2.5241246780253446e-18, 5.364001077188067e-19, -1.1151793774136653e-18, -2.647497194530543e-19, -9.99558548618923e-19, 1.9545043306813755e-18, 1.0295729550672243e-18, 5.850363951633064e-19, 2.039938658283451e-19, -1.622964422184252e-18, -2.1240860691038883e-18, 2.0835979355758296e-18, -1.5423943263121515e-19, -5.576729517866199e-19, -3.7259501367076117e-20, -1.8905808347578018e-19, -1.4869058365515489e-18)
)

#=
### don't quite have this right (issue with signs)
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
