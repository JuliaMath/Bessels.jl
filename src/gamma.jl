# Float64 version adapted from Cephes Mathematical Library (MIT license https://en.smath.com/view/CephesMathLibrary/license) by Stephen L. Moshier
function gamma(_x::Float64)
    T = Float64
    x = _x
    if x < 0
        s = sinpi(_x)
        s == 0 && throw(DomainError(x, "NaN result for non-NaN input."))
        x = -x # Use this rather than the traditional x = 1-x to avoid roundoff.
	s *= x
    end
    if x > 11.5
        w = inv(x)
        coefs = (1.0, 
            8.333333333333331800504e-2, 3.472222222230075327854e-3, -2.681327161876304418288e-3, -2.294719747873185405699e-4,
            7.840334842744753003862e-4, 6.989332260623193171870e-5, -5.950237554056330156018e-4, -2.363848809501759061727e-5,
            7.147391378143610789273e-4
        )
        w = evalpoly(w, coefs)
        # avoid overflow
        v = x ^ muladd(0.5, x, -0.25)
        res = SQ2PI(T) * v * (v / exp(x)) * w

        if _x < 0
            return π / (res * s)
        else
            return res
        end
    end
    P = (
        1.000000000000000000009e0, 8.378004301573126728826e-1, 3.629515436640239168939e-1, 1.113062816019361559013e-1,
        2.385363243461108252554e-2, 4.092666828394035500949e-3, 4.542931960608009155600e-4, 4.212760487471622013093e-5
    )
    Q = (
        9.999999999999999999908e-1, 4.150160950588455434583e-1, -2.243510905670329164562e-1, -4.633887671244534213831e-2,
        2.773706565840072979165e-2, -7.955933682494738320586e-4, -1.237799246653152231188e-3, 2.346584059160635244282e-4,
        -1.397148517476170440917e-5
    )

    z = 1.0
    while x >= 3.0
        x -= 1.0
        z *= x
    end
    while x < 2.0
        z /= x
        x += 1.0
    end

    x -= 2.0
    p = evalpoly(x, P)
    q = evalpoly(x, Q)
    return _x < 0 ? π * q / (s * z * p) : z * p / q
end


function gamma(_x::Float32)
    x = Float64(_x)
    if _x < 0
        s = sinpi(x)
        s == 0 && throw(DomainError(_x, "NaN result for non-NaN input."))
        x = 1 - x
    end
    if x < 5
        z = 1.0
        while x > 1
            x -= 1
            z *= x
        end
        num = evalpoly(x, (1.0, 0.41702538904450015, 0.24081703455575904, 0.04071509011391178, 0.015839573267537377))
        den = x*evalpoly(x, (1.0, 0.9942411061082665, -0.17434932941689474, -0.13577921102050783, 0.03028452206514555))
        res = z * num / den
    else
        x -= 1
        w = evalpoly(inv(x), (2.506628299028453, 0.20888413086840676, 0.008736513049552962, -0.007022997182153692, 0.0006787969600290756))
        res = @fastmath sqrt(x) * exp(log(x*1/ℯ) * x) * w
    end
    return Float32(_x < 0 ? π / (s * res) : res)
end

function gamma(_x::Float16)
    x = Float32(_x)
    if _x < 0
        s = sinpi(x)
        s == 0 && throw(DomainError(_x, "NaN result for non-NaN input."))
        x = 1 - x
    end
    x > 14 && return Float16(ifelse(_x > 0, Inf32, 0f0))
    z = 1f0
    while x > 1
        x -= 1
        z *= x
    end
    num = evalpoly(x, (1.0f0, 0.4170254f0, 0.24081704f0, 0.04071509f0, 0.015839573f0))
    den = x*evalpoly(x, (1.0f0, 0.9942411f0, -0.17434932f0, -0.13577922f0, 0.030284522f0))
    return Float16(_x < 0 ? Float32(π)*den / (s*z*num) : z * num / den)
end

function gamma(n::Integer)
    n < 0 && throw(DomainError(n, "`n` must not be negative."))
    n == 0 && return Inf*one(n)
    n > 20 && return gamma(float(n))
    @inbounds return Float64(factorial(n-1))
end
