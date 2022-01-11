function besseli0(x::Float32)
    T = Float32
    if x < zero(x)
        x = -x
    end
    if x <= 8.0f0
        y = 0.5f0 * x - 2.0f0
        return exp(x) * chbevl(y, A_i0(T))
    else
        return exp(x) * chbevl(32.0f0 / x - 2.0f0, B_i0(T)) / sqrt(x)
    end
end
function besseli0x(x::Float32)
    T = Float32
    if x < zero(x)
        x = -x
    end
    if x <= 8.0f0
        y = 0.5f0 * x - 2.0f0
        return chbevl(y, A_i0(T))
    else
        return chbevl(32.0f0 / x - 2.0f0, B_i0(T)) / sqrt(x)
    end
end
function besseli1(x::Float32)
    T = Float32
    z = abs(x)
    if z <= 8.0f0
        y = 0.5f0 * z - 2.0f0
        z = chbevl(y, A_i1(T)) * z * exp(z)
    else
        z = exp(z) * chbevl(32.0f0 / z - 2.0f0, B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
function besseli1x(x::Float32)
    T = Float32
    z = abs(x)
    if z <= 8.0f0
        y = 0.5f0 * z - 2.0f0
        z = chbevl(y, A_i1(T)) * z
    else
        z = chbevl(32.0f0 / z - 2.0f0, B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
