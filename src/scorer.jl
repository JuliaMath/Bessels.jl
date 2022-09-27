
# see     http://dlmf.nist.gov/9.12.vi
# the trick is to split the sum into three parts so it's easy to use gamma recurrence relation
const ComplexOrReal{T} = Union{T,Complex{T}}

# only valid for abs(imag(z)) < 1.2*sqrt(real(x) + 3.5)
# of course real(x) > -3.5 as well
function scorerhi_power_series(z::ComplexOrReal{T}; tol=eps(T)) where T
    MaxIter = 2000
    out = zero(z)
    out2 = zero(z)
    out3 = zero(z)
    z2 = z*z
    z3 = z2*z

    a = gamma(T(1)/3)
    a2 = gamma(T(2)/3)*3^(T(1)/3)*z
    a3 = gamma(T(1))*3^(T(2)/3)*z2 / 2
    for i in 0:3:MaxIter
        out += a
        out2 += a2
        out3 += a3
        abs(a) < tol * abs(out) && break
        a *= z3*(i + one(T)) / ((i + one(T))*(i + T(2))*(i + T(3)))
        a2 *= z3*(i + T(2)) / ((i + T(2))*(i + T(3))*(i + T(4)))
        a3 *= z3*(i + T(3)) / ((i + T(3))*(i + T(4))*(i + T(5)))
    end
    return 3^(-T(2)/3) * (out + out2 + out3) / pi
end

# valid when abs(imag(z)) < -2.3(real(z) - 3.2) && abs(imag(z)) > -1.45*(real(x) + 4)
# so only really works for negative arguments and not close to real line

# in float32 precision abs(imag(z)) <  -2.3(real(z)-8.2) && abs(imag(z)) > -1.4*(real(x) + 9.5)
# some larger validity but need to see where asymptotic expansion converges
function scorergi_power_series(z::ComplexOrReal{T}; tol=eps(T)) where T
    MaxIter = 2000
    out = zero(z)
    out2 = zero(z)
    out3 = zero(z)
    z2 = z*z
    z3 = z2*z

    a = gamma(1/3)
    a2 = gamma(2/3)*3^(1/3)*z
    a3 = gamma(1)*3^(2/3)*z2 / 2
    for i in 0:3:MaxIter
        out += a / 2
        out2 += a2 / 2
        out3 += -a3
        abs(a) < tol * abs(out) && break
        a *= z3*(i + one(T)) / ((i + one(T))*(i + T(2))*(i + T(3)))
        a2 *= z3*(i + T(2)) / ((i + T(2))*(i + T(3))*(i + T(4)))
        a3 *= z3*(i + T(3)) / ((i + T(3))*(i + T(4))*(i + T(5)))
    end
    return 3^(-T(2)/3) * (out + out2 + out3) / pi
end
