function besselk0(x::Float32)
    T = Float32
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end

    if x <= 2.0f0
        y = x * x - 2.0f0
        y = chbevl(y, A_k0(T)) - log(0.5f0 * x) * besseli0(x)
        return y
    else
        z = 8.0f0 / x - 2.0f0
        y = exp(-x) * chbevl(z, B_k0(T)) / sqrt(x)
        return y
    end
end
function besselk0x(x::Float32)
    T = Float32
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end

    if x <= 2.0f0
        y = x * x - 2.0f0
        y = chbevl(y, A_k0(T)) - log(0.5f0 * x) * besseli0(x)
        return y * exp(x)
    else
        z = 8.0f0 / x - 2.0f0
        y = chbevl(z, B_k0(T)) / sqrt(x)
        return y
    end
end

function besselk1(x::Float32)
    T = Float32
    if x <= zero(x)
        if iszero(x)
            return Inf32
        else
            return throw(DomainError(x, "NaN result for non-NaN input."))
        end
    end
    if x <= 2.0f0
        y = x * x - 2.0f0
        y = log(0.5f0 * x) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y
    else
        return exp(-x) * chbevl(8.0f0 / x - 2.0f0, B_k1(T)) / sqrt(x)
    end
end

function besselk1x(x::Float32)
    T = Float32
    if x <= zero(x)
        if iszero(x)
            return Inf32
        else
            return throw(DomainError(x, "NaN result for non-NaN input."))
        end
    end
    if x <= 2.0f0
        y = x * x - 2.0f0
        y = log(0.5f0 * x) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y * exp(x)
    else
        return chbevl(8.0f0 / x - 2.0f0, B_k1(T)) / sqrt(x)
    end
end
