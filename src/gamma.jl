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

