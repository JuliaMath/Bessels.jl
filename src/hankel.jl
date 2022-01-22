#= Aren't tested yet
function besselh(nu::Float64, k::Integer, x::AbstractFloat)
    if k == 1
        return complex(besselj(nu, x), bessely(nu, x))
    elseif k == 2
        return complex(besselj(nu, x), -bessely(nu, x))
    else
        throw(ArgumentError("k must be 1 or 2"))
    end
end

hankelh1(nu, z) = besselh(nu, 1, z)
hankelh2(nu, z) = besselh(nu, 2, z)
=#