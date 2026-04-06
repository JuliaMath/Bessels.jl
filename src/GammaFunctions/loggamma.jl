# Pure Julia loggamma and logabsgamma implementations
# Partly adapted from SpecialFunctions.jl (MIT license)
# using Stirling asymptotic series, Taylor series at z=1 and z=2,
# reflection formula, and shift recurrence.
# See: D. E. G. Hare, "Computing the principal branch of log-Gamma,"
# J. Algorithms 25, pp. 221-236 (1997)

"""
    loggamma(x::Real)

Returns the log of the absolute value of ``\\Gamma(x)`` for real `x`.
Throws a `DomainError` if ``\\Gamma(x)`` is negative.

For complex arguments, `exp(loggamma(x))` matches `gamma(x)` up to floating-point error
but may differ from `log(gamma(x))` by an integer multiple of ``2\\pi i``.

External links: [DLMF](https://dlmf.nist.gov/5.4), [Wikipedia](https://en.wikipedia.org/wiki/Gamma_function#The_log-gamma_function)
"""
loggamma(x::Float64) = _loggamma(x)
loggamma(x::Union{Float16, Float32}) = typeof(x)(_loggamma(Float64(x)))
loggamma(z::Complex{Float64}) = _loggamma(z)
loggamma(z::Complex{Float32}) = Complex{Float32}(_loggamma(Complex{Float64}(z)))
loggamma(z::Complex{Float16}) = Complex{Float16}(_loggamma(Complex{Float64}(z)))
loggamma(z::Complex{<:Integer}) = _loggamma(Complex{Float64}(z))
loggamma(x::BigFloat) = real(_loggamma_complex_bigfloat(Complex{BigFloat}(x, zero(BigFloat))))
loggamma(z::Complex{BigFloat}) = _loggamma(z)

"""
    logfactorial(x)

Compute the logarithmic factorial of a nonnegative integer `x` via loggamma.
"""
logfactorial(x::Integer) = x < 0 ? throw(DomainError(x, "`x` must be non-negative.")) : loggamma(float(x + oneunit(x)))

"""
    logabsgamma(x::Real)

Returns a tuple `(log(abs(Γ(x))), sign(Γ(x)))` for real `x`.
"""
logabsgamma(x::Real) = _logabsgamma(float(x))

const HALF_LOG2PI_F64 = 9.1893853320467274178032927e-01
const LOGPI_F64 = 1.1447298858494002
const TWO_PI_F64 = 6.2831853071795864769252842

# Stirling asymptotic series for log(Γ(x)), valid for x > 0 sufficiently large
# coefficients are bernoulli[2k] / (2k*(2k-1)) for k = 1,...,8
function _loggamma_stirling(x::Float64)
    t = inv(x)
    w = t * t
    return muladd(x - 0.5, log(x), -x + HALF_LOG2PI_F64 +  # log(2π)/2
        t * @evalpoly(w,
            8.333333333333333333333368e-02, -2.777777777777777777777778e-03,
            7.936507936507936507936508e-04, -5.952380952380952380952381e-04,
            8.417508417508417508417510e-04, -1.917526917526917526917527e-03,
            6.410256410256410256410257e-03, -2.955065359477124183006535e-02
        )
    )
end

# Asymptotic series for log(Γ(z)) for complex z with sufficiently large real(z) or |imag(z)|
function _loggamma_asymptotic(z::Complex{Float64})
    zinv = inv(z)
    t = zinv * zinv
    return (z - 0.5) * log(z) - z + HALF_LOG2PI_F64 +  # log(2π)/2
        zinv * @evalpoly(t,
            8.333333333333333333333368e-02, -2.777777777777777777777778e-03,
            7.936507936507936507936508e-04, -5.952380952380952380952381e-04,
            8.417508417508417508417510e-04, -1.917526917526917526917527e-03,
            6.410256410256410256410257e-03, -2.955065359477124183006535e-02
        )
end

# logabsgamma for Float64
function _logabsgamma(x::Float64)
    if isnan(x)
        return x, 1
    elseif x > 0
        return _loggamma(x), 1
    elseif x == 0
        return Inf, Int(sign(1/x))  # ±0 → correct sign
    else
        # reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        s = sinpi(x)
        s == 0 && return Inf, 1
        sgn = signbit(s) ? -1 : 1
        return LOGPI_F64 - log(abs(s)) - _loggamma(1.0 - x), sgn
    end
end

function _logabsgamma(x::Float32)
    y, s = _logabsgamma(Float64(x))
    return Float32(y), s
end

function _logabsgamma(x::Float16)
    y, s = _logabsgamma(Float64(x))
    return Float16(y), s
end

# Lanczos-type rational approximation for loggamma on (2, 3)
# Used as the core for reduction-based approach
const _LOGGAMMA_P = (
    -2.44167345903529816830968e-01, 6.73523010531981020863696e-02,
    -2.05808084277845478790009e-02, 7.38555102867398526627303e-03,
    -2.89051033074153369901384e-03, 1.19275391170326097711398e-03,
    -5.09669524743042422335582e-04, 2.23154759903498081132513e-04,
    -9.94575127818085337147321e-05, 4.49262367382046739858373e-05,
    -2.05077312586603517590604e-05
)

# loggamma for real Float64
function _loggamma(x::Float64)
    if isnan(x)
        return x
    elseif isinf(x)
        return x > 0 ? Inf : NaN
    elseif x ≤ 0
        x == 0 && return Inf
        s = sinpi(x)
        s == 0 && return Inf  # negative integer pole
        # reflection: log|Γ(x)| = log(π) - log|sin(πx)| - log(Γ(1-x))
        # but loggamma for real requires Γ(x)>0
        y, sgn = _logabsgamma(x)
        sgn < 0 && throw(DomainError(x, "`gamma(x)` must be non-negative"))
        return y
    elseif x < 7
        # shift x into asymptotic region [7,∞)
        n = 7 - floor(Int, x)
        z = x
        prod = one(x)
        for i in 0:n-1
            prod *= z + i
        end
        return _loggamma_stirling(z + n) - log(prod)
    else
        return _loggamma_stirling(x)
    end
end

function _loggamma(x::Float32)
    return Float32(_loggamma(Float64(x)))
end

function _loggamma(x::Float16)
    return Float16(_loggamma(Float64(x)))
end

# Complex loggamma for Float64
# Combines the asymptotic series, Taylor series at z=1 and z=2,
# the reflection formula, and the shift recurrence.
function _loggamma(z::Complex{Float64})
    x, y = reim(z)
    yabs = abs(y)

    if !isfinite(x) || !isfinite(y)
        if isinf(x) && isfinite(y)
            return Complex(x, x > 0 ? (y == 0 ? y : copysign(Inf, y)) : copysign(Inf, -y))
        elseif isfinite(x) && isinf(y)
            return Complex(-Inf, y)
        else
            return Complex(NaN, NaN)
        end
    elseif x > 7 || yabs > 7
        return _loggamma_asymptotic(z)
    elseif x < 0.1
        if x == 0 && y == 0
            return Complex(Inf, signbit(x) ? copysign(Float64(π), -y) : -y)
        end
        # reflection formula with correct branch cut.
        return Complex(LOGPI_F64, copysign(TWO_PI_F64, y) * floor(0.5 * x + 0.25)) -
            log(sinpi(z)) - _loggamma(1 - z)
    elseif abs(x - 1) + yabs < 0.1
        # Taylor series at z=1
        # coefficients: [-γ; [(-1)^k * ζ(k)/k for k in 2:15]]
        w = Complex(x - 1, y)
        return w * @evalpoly(w,
            -5.7721566490153286060651188e-01, 8.2246703342411321823620794e-01,
            -4.0068563438653142846657956e-01, 2.705808084277845478790009e-01,
            -2.0738555102867398526627303e-01, 1.6955717699740818995241986e-01,
            -1.4404989676884611811997107e-01, 1.2550966952474304242233559e-01,
            -1.1133426586956469049087244e-01, 1.000994575127818085337147e-01,
            -9.0954017145829042232609344e-02, 8.3353840546109004024886499e-02,
            -7.6932516411352191472827157e-02, 7.1432946295361336059232779e-02,
            -6.6668705882420468032903454e-02
        )
    elseif abs(x - 2) + yabs < 0.1
        # Taylor series at z=2
        # coefficients: [1-γ; [(-1)^k * (ζ(k)-1)/k for k in 2:12]]
        w = Complex(x - 2, y)
        return w * @evalpoly(w,
            4.2278433509846713939348812e-01, 3.2246703342411321823620794e-01,
            -6.7352301053198095133246196e-02, 2.0580808427784547879000897e-02,
            -7.3855510286739852662729527e-03, 2.8905103307415232857531201e-03,
            -1.1927539117032609771139825e-03, 5.0966952474304242233558822e-04,
            -2.2315475845357937976132853e-04, 9.9457512781808533714662972e-05,
            -4.4926236738133141700224489e-05, 2.0507212775670691553131246e-05
        )
    else
        # shift using recurrence: loggamma(z) = loggamma(z+n) - log(∏(z+k))
        shiftprod = Complex(x, yabs)
        x += 1
        sb = false
        signflips = 0
        while x ≤ 7
            shiftprod *= Complex(x, yabs)
            sb′ = signbit(imag(shiftprod))
            signflips += sb′ & (sb′ != sb)
            sb = sb′
            x += 1
        end
        shift = log(shiftprod)
        if signbit(y)
            shift = Complex(real(shift), signflips * -TWO_PI_F64 - imag(shift))
        else
            shift = Complex(real(shift), imag(shift) + signflips * TWO_PI_F64)
        end
        return _loggamma_asymptotic(Complex(x, y)) - shift
    end
end


# Complex BigFloat loggamma
# Adapted from SpecialFunctions.jl (MIT license)
# Uses Stirling series with Bernoulli numbers computed via Akiyama-Tanigawa,
# reflection formula, upward recurrence, and branch correction via Float64 oracle.

# Scaled Stirling coefficients B_{2k}/(2k*(2k-1)) * zr^(1-2k) for k=0,...,n
# Bernoulli numbers computed inline via the Akiyama-Tanigawa algorithm
function _scaled_stirling_coeffs(n::Integer, zr::Complex{BigFloat})
    mmax = 2n
    A = Vector{Rational{BigInt}}(undef, mmax + 1)
    E = Vector{Complex{BigFloat}}(undef, n + 1)
    @inbounds for m = 0:mmax
        A[m+1] = 1 // (m + 1)
        for j = m:-1:1
            A[j] = j * (A[j] - A[j+1])
        end
        if iseven(m)
            k = m ÷ 2
            E[k+1] = A[1] / (2k * (2k - 1)) * zr^(1 - 2k)
        end
    end
    return E
end

function _loggamma_complex_bigfloat(z::Complex{BigFloat})
    bigpi = big(π)

    # reflection formula
    if real(z) < 0.5
        val = log(bigpi) - log(sinpi(z)) - _loggamma_complex_bigfloat(1 - z)
        return _loggamma_branchcorrect(val, z)
    end

    # upward recurrence: shift z into the Stirling region
    p = precision(BigFloat)
    r = max(0, Int(ceil(p - abs(z))))
    zr = z + r

    # Stirling series
    N = max(10, p ÷ 15)
    B = _scaled_stirling_coeffs(N, zr)
    lg = sum(B[2:end]) + (zr - big"0.5") * log(zr) - zr + log(sqrt(2 * bigpi))

    # undo upward shift via log of product
    if r > 0
        prodarg = prod(z + (i - 1) for i in 1:r)
        lg -= log(prodarg)
    end

    return _loggamma_branchcorrect(lg, z)
end

# Branch correction: offset by multiples of 2πi to match the Float64 branch
function _loggamma_branchcorrect(val::Complex{BigFloat}, z::Complex{BigFloat})
    zf = _loggamma_oracle64_point(z)
    val_f = _loggamma(zf)
    imv = imag(val)
    k = round(Int, (Float64(imv) - imag(val_f)) / (2π))
    return Complex{BigFloat}(real(val), imv - 2 * big(π) * k)
end

# Map a BigFloat complex point to Float64 for branch-cut determination
function _loggamma_oracle64_point(z::Complex{BigFloat})
    xr = Float64(real(z))
    xi = Float64(imag(z))
    n = round(Int, xr)
    if n ≤ 0 && isapprox(xr, Float64(n); atol=eps(Float64)) && abs(xi) ≤ 2eps(Float64)
        xr = real(z) > n ? nextfloat(xr) : prevfloat(xr)
    end
    return Complex{Float64}(xr, xi)
end

# Complex{BigFloat} entry point with guard precision
function _loggamma(z::Complex{BigFloat})
    imz = imag(z)
    rez = real(z)
    if iszero(imz)
        return Complex(loggamma(rez))
    end
    p0 = precision(BigFloat)
    guard = 16
    setprecision(p0 + guard) do
        zhi = Complex{BigFloat}(rez, imz)
        rhi = _loggamma_complex_bigfloat(zhi)
        setprecision(p0) do
            return Complex{BigFloat}(real(rhi), imag(rhi))
        end
    end
end

