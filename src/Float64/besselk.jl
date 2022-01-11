function besselk0(x::Float64)
    T = Float64
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end

    if x <= 2.0
        y = x * x - 2.0
        y = chbevl(y, A_k0(T)) - log(0.5 * x) * besseli0(x)
        return y
    else
        z = 8.0 / x - 2.0
        y = exp(-x) * chbevl(z, B_k0(T)) / sqrt(x)
        return y
    end
end

function besselk0x(x::Float64)
    T = Float64
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2.0
        y = x * x - 2.0
        y = chbevl(y, A_k0(T)) - log(0.5 * x) * besseli0(x)
        return y * exp(x)
    else
        z = 8.0 / x - 2.0
        y = chbevl(z, B_k0(T)) / sqrt(x)
        return y
    end
end
function besselk1(x::Float64)
    T = Float64
    z = 0.5 * x
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2.0
        y = x * x - 2.0
        y = log(z) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y
    else
        return exp(-x) * chbevl(8.0 / x - 2.0, B_k1(T)) / sqrt(x)
    end
end
function besselk1x(x::Float64)
    T = Float64
    z = 0.5 * x
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2.0
        y = x * x - 2.0
        y = log(z) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y * exp(x)
    else
        return chbevl(8.0 / x - 2.0, B_k1(T)) / sqrt(x)
    end
end

