
"""
    sphericalbesselk(nu, x::T) where T <: {Float32, Float64}

Computes `k_{ν}(x)`, the modified second-kind spherical Bessel function, and offers special branches for integer orders.
"""
sphericalbesselk(nu, x) = _sphericalbesselk(nu, float(x))

function _sphericalbesselk(nu, x::T) where T
    isnan(x) && return NaN
    if isinteger(nu) && nu < 41.5
        if x < zero(x)
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
        end
        # using ifelse here to hopefully cut out a branch on nu < 0 or not. The
        # symmetry here is that 
        # k_{-n} = (...)*K_{-n     + 1/2}
        #        = (...)*K_{|n|    - 1/2}
        #        = (...)*K_{|n|-1  + 1/2}
        #        = k_{|n|-1}  
        _nu = ifelse(nu<zero(nu), -one(nu)-nu, nu)
        return sphericalbesselk_int(Int(_nu), x)
    else
        return inv(SQRT_PID2(T)*sqrt(x))*besselk(nu+1/2, x)
    end
end

function sphericalbesselk_int(v::Int, x)
    b0 = inv(x)
    b1 = (x+one(x))/(x*x)
    iszero(v) && return b0*exp(-x)
    _v = one(v)
    invx = inv(x)
    while _v < v
        _v += one(_v)
        b0, b1 = b1, b0 + (2*_v - one(_v))*b1*invx
    end
    exp(-x)*b1
end

