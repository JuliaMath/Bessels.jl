# Adapted from Cephes Mathematical Library (MIT license https://en.smath.com/view/CephesMathLibrary/license) by Stephen L. Moshier
gamma(z::Number) = _gamma(float(z))
_gamma(x::Float32) = Float32(_gamma(Float64(x)))

function _gamma(x::Float64)
    if x < 0
        isinteger(x) && throw(DomainError(x, "NaN result for non-NaN input."))
        xp1 = abs(x) + 1.0
        return π / sinpi(xp1) / _gammax(xp1)
    else
        return _gammax(x)
    end
end
# only have a Float64 implementations
function _gammax(x)
    if x > 11.5
        return large_gamma(x)
    elseif x <= 11.5
        return small_gamma(x)
    elseif isnan(x)
        return x
    end
end
function large_gamma(x)
    isinf(x) && return x
    T = Float64
    w = inv(x)
    s = (
        8.333333333333331800504e-2, 3.472222222230075327854e-3, -2.681327161876304418288e-3, -2.294719747873185405699e-4,
        7.840334842744753003862e-4, 6.989332260623193171870e-5, -5.950237554056330156018e-4, -2.363848809501759061727e-5,
        7.147391378143610789273e-4
    )
    w = w * evalpoly(w, s) + one(T)
    # lose precision on following block
    y = exp((x))
    # avoid overflow
    v = x^(0.5 * x - 0.25)
    y = v * (v / y)

    return SQ2PI(T) * y * w
end
function small_gamma(x)
    T = Float64
    P = (
        1.000000000000000000009e0, 8.378004301573126728826e-1, 3.629515436640239168939e-1, 1.113062816019361559013e-1,
        2.385363243461108252554e-2, 4.092666828394035500949e-3, 4.542931960608009155600e-4, 4.212760487471622013093e-5
    )
    Q = (
        9.999999999999999999908e-1, 4.150160950588455434583e-1, -2.243510905670329164562e-1, -4.633887671244534213831e-2,
        2.773706565840072979165e-2, -7.955933682494738320586e-4, -1.237799246653152231188e-3, 2.346584059160635244282e-4,
        -1.397148517476170440917e-5
    )

    z = one(T)
    while x >= 3.0
        x -= one(T)
        z *= x
    end
    while x < 2.0
        z /= x
        x += one(T)
    end

    x -= T(2)
    p = evalpoly(x, P)
    q = evalpoly(x, Q)
    return z * p / q
end

_gamma(z::Complex) = exp(loggamma(z))

loggamma(z::Complex) = _loggamma(float(z))

_loggamma(x::Complex{Float32}) = ComplexF32(gamma(ComplexF64(x)))

# copied from https://github.com/JuliaMath/SpecialFunctions.jl/blob/master/src/gamma.jl

# Compute the logΓ(z) function using a combination of the asymptotic series,
# the Taylor series around z=1 and z=2, the reflection formula, and the shift formula.
# Many details of these techniques are discussed in D. E. G. Hare,
# "Computing the principal branch of log-Gamma," J. Algorithms 25, pp. 221-236 (1997),
# and similar techniques are used (in a somewhat different way) by the
# SciPy loggamma function.  The key identities are also described
# at http://functions.wolfram.com/GammaBetaErf/LogGamma/
function _loggamma(z::Complex{Float64})
    x, y = reim(z)
    yabs = abs(y)
    if !isfinite(x) || !isfinite(y) # Inf or NaN
        if isinf(x) && isfinite(y)
            return Complex(x, x > 0 ? (y == 0 ? y : copysign(Inf, y)) : copysign(Inf, -y))
        elseif isfinite(x) && isinf(y)
            return Complex(-Inf, y)
        else
            return Complex(NaN, NaN)
        end
    elseif x > 7 || yabs > 7 # use the Stirling asymptotic series for sufficiently large x or |y|
        return loggamma_asymptotic(z)
    elseif x < 0.1 # use reflection formula to transform to x > 0
        if x == 0 && y == 0 # return Inf with the correct imaginary part for z == 0
            return Complex(Inf, signbit(x) ? copysign(Float64(π), -y) : -y)
        end
        # the 2pi * floor(...) stuff is to choose the correct branch cut for log(sinpi(z))
        return Complex(Float64(logπ), copysign(Float64(twoπ), y) * floor(0.5*x+0.25)) -
            log(sinpi(z)) - loggamma(1-z)
    elseif abs(x - 1) + yabs < 0.1
        # taylor series at z=1
        # ... coefficients are [-eulergamma; [(-1)^k * zeta(k)/k for k in 2:15]]
        w = Complex(x - 1, y)
        return w * @evalpoly(w, -5.7721566490153286060651188e-01,8.2246703342411321823620794e-01,
                                -4.0068563438653142846657956e-01,2.705808084277845478790009e-01,
                                -2.0738555102867398526627303e-01,1.6955717699740818995241986e-01,
                                -1.4404989676884611811997107e-01,1.2550966952474304242233559e-01,
                                -1.1133426586956469049087244e-01,1.000994575127818085337147e-01,
                                -9.0954017145829042232609344e-02,8.3353840546109004024886499e-02,
                                -7.6932516411352191472827157e-02,7.1432946295361336059232779e-02,
                                -6.6668705882420468032903454e-02)
    elseif abs(x - 2) + yabs < 0.1
        # taylor series at z=2
        # ... coefficients are [1-eulergamma; [(-1)^k * (zeta(k)-1)/k for k in 2:12]]
        w = Complex(x - 2, y)
        return w * @evalpoly(w, 4.2278433509846713939348812e-01,3.2246703342411321823620794e-01,
                               -6.7352301053198095133246196e-02,2.0580808427784547879000897e-02,
                               -7.3855510286739852662729527e-03,2.8905103307415232857531201e-03,
                               -1.1927539117032609771139825e-03,5.0966952474304242233558822e-04,
                               -2.2315475845357937976132853e-04,9.9457512781808533714662972e-05,
                               -4.4926236738133141700224489e-05,2.0507212775670691553131246e-05)
    end
    # use recurrence relation loggamma(z) = loggamma(z+1) - log(z) to shift to x > 7 for asymptotic series
    shiftprod = Complex(x,yabs)
    x += 1
    sb = false # == signbit(imag(shiftprod)) == signbit(yabs)
    # To use log(product of shifts) rather than sum(logs of shifts),
    # we need to keep track of the number of + to - sign flips in
    # imag(shiftprod), as described in Hare (1997), proposition 2.2.
    signflips = 0
    while x <= 7
        shiftprod *= Complex(x,yabs)
        sb′ = signbit(imag(shiftprod))
        signflips += sb′ & (sb′ != sb)
        sb = sb′
        x += 1
    end
    shift = log(shiftprod)
    if signbit(y) # if y is negative, conjugate the shift
        shift = Complex(real(shift), signflips*-6.2831853071795864769252842 - imag(shift))
    else
        shift = Complex(real(shift), imag(shift) + signflips*6.2831853071795864769252842)
    end
    return loggamma_asymptotic(Complex(x,y)) - shift
end

# asymptotic series for log(gamma(z)), valid for sufficiently large real(z) or |imag(z)|
function loggamma_asymptotic(z::Complex{Float64})
    zinv = inv(z)
    t = zinv*zinv
    # coefficients are bernoulli[2:n+1] .// (2*(1:n).*(2*(1:n) - 1))
    return (z-0.5)*log(z) - z + 9.1893853320467274178032927e-01 + # <-- log(2pi)/2
       zinv*@evalpoly(t, 8.3333333333333333333333368e-02,-2.7777777777777777777777776e-03,
                         7.9365079365079365079365075e-04,-5.9523809523809523809523806e-04,
                         8.4175084175084175084175104e-04,-1.9175269175269175269175262e-03,
                         6.4102564102564102564102561e-03,-2.9550653594771241830065352e-02)
end