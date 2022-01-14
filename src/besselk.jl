#=
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 2000 by Stephen L. Moshier
https://github.com/jeremybarnes/cephes/blob/master/bessel/k0.c
https://github.com/jeremybarnes/cephes/blob/master/bessel/k1.c
=#

const P1_k0(::Type{Float32}) = (
    -1.372508979104259711f-1, 2.622545986273687617f-1, 5.047103728247919836f-3
)
const Q1_k0(::Type{Float32}) = (
    1.000000000000000000f0, -8.928694018000029415f-2, 2.985980684180969241f-3
)
const P2_k0(::Type{Float32}) = (
    1.159315158f-1, 2.789828686f-1, 2.524902861f-2,
    8.457241514f-4, 1.530051997f-5
)
const P3_k0(::Type{Float32}) = (
    2.533141220f-1, 5.221502603f-1,
    6.380180669f-2, -5.934976547f-2
)
const Q3_k0(::Type{Float32}) = (
    1.000000000f0, 2.679722431f0,
    1.561635813f0, 1.573660661f-1
)
const P1_k0(::Type{Float64}) = (
    -1.372509002685546267e-1, 2.574916117833312855e-1,
    1.395474602146869316e-2, 5.445476986653926759e-4,
    7.125159422136622118e-6
)
const Q1_k0(::Type{Float64}) = (
    1.000000000000000000e+00, -5.458333438017788530e-02,
    1.291052816975251298e-03, -1.367653946978586591e-05
)
const P2_k0(::Type{Float64}) = (
    1.159315156584124484e-01, 2.789828789146031732e-01,
    2.524892993216121934e-02, 8.460350907213637784e-04,
    1.491471924309617534e-05, 1.627106892422088488e-07,
    1.208266102392756055e-09, 6.611686391749704310e-12
)
const P3_k0(::Type{Float64}) = (
    2.533141373155002416e-1, 3.628342133984595192e0,
    1.868441889406606057e1, 4.306243981063412784e1,
    4.424116209627428189e1, 1.562095339356220468e1,
    -1.810138978229410898e0, -1.414237994269995877e0,
    -9.369168119754924625e-2
)
const Q3_k0(::Type{Float64}) = (
    1.000000000000000000e0, 1.494194694879908328e1,
    8.265296455388554217e1, 2.162779506621866970e2,
    2.845145155184222157e2, 1.851714491916334995e2,
    5.486540717439723515e1, 6.118075837628957015e0,
    1.586261269326235053e-1
)
const Y_k0 = 1.137250900268554688

const LOGMAXVAL(::Type{Float32}) = 88.0f0
const LOGMAXVAL(::Type{Float64}) = 709.0 


function besselk0(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        a = x * x / 4
        s = muladd(evalpoly(a, P1_k0(T)), inv(evalpoly(a, Q1_k0(T))), T(Y_k0))
        a = muladd(s, a, 1)
        return muladd(-a, log(x), evalpoly(x * x, P2_k0(T)))
    else
        if x < LOGMAXVAL(T)
            return muladd(evalpoly(inv(x), P3_k0(T)), inv(evalpoly(inv(x), Q3_k0(T))), one(T)) * exp(-x) / sqrt(x)
        else
            s = exp(-x / 2)
            a = muladd(evalpoly(inv(x), P3_k0(T)), inv(evalpoly(inv(x), Q3_k0(T))), one(T)) * s / sqrt(x)
            return a * s
        end
    end
end
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

function besselk1(x::T) where T <: Union{Float32, Float64}
    z = T(.5) * x
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2
        y = muladd(x, x, T(-2))
        y = log(z) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y
    else
        return exp(-x) * chbevl(T(8) / x - T(2), B_k1(T)) / sqrt(x)
    end
end
function besselk1x(x::T) where T <: Union{Float32, Float64}
    z = T(.5) * x
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2
        y = muladd(x, x, T(-2))
        y = log(z) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y * exp(x)
    else
        return chbevl(T(8) / x - T(2), B_k1(T)) / sqrt(x)
    end
end
