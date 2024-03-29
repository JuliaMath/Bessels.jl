function _besselj0(x::BigFloat)
    x = abs(x)
    T = eltype(x)
    if iszero(x)
        return one(x)
    elseif isinf(x)
        return zero(x)
    end

    if x <= 2.0
        z = x * x
        p = z * z * evalpoly(z, J0_2N) / evalpoly(z, J0_2D)
        p -= 0.25 * z
        p += one(T)
        return p
    else
        xinv = inv(x)
        z  = xinv * xinv
        if xinv <= 0.25
            if xinv <= 0.125
                if xinv <= 0.0625
                    p = evalpoly(z, P16_IN) / evalpoly(z, P16_ID)
                    q = evalpoly(z, Q16_IN) / evalpoly(z, Q16_ID)
                else
                    p = evalpoly(z, P8_16N) / evalpoly(z, P8_16D)
                    q = evalpoly(z, Q8_16N) /  evalpoly(z, Q8_16D)
                end
            elseif xinv <= 0.1875
                p = evalpoly(z, P5_8N) / evalpoly(z, P5_8D)
	            q = evalpoly(z, Q5_8N) / evalpoly(z, Q5_8D)
            else
                p = evalpoly(z, P4_5N) / evalpoly(z, P4_5D)
	            q = evalpoly(z, Q4_5N) / evalpoly(z, Q4_5D)
            end
        elseif xinv <= 0.5
            if xinv <= 0.375
                if xinv <= 0.3125
                    p = evalpoly(z, P3r2_4N) / evalpoly(z, P3r2_4D)
	                q = evalpoly(z, Q3r2_4N) / evalpoly(z, Q3r2_4D)
                else
                    p = evalpoly(z, P2r7_3r2N) / evalpoly(z, P2r7_3r2D)
                    q = evalpoly(z, Q2r7_3r2N) / evalpoly(z, Q2r7_3r2D)
                end
            elseif xinv <= 0.4375
                p = evalpoly(z, P2r3_2r7N) / evalpoly(z, P2r3_2r7D)
	            q = evalpoly(z, Q2r3_2r7N) / evalpoly(z, Q2r3_2r7D)
            else
                p = evalpoly(z, P2_2r3N) / evalpoly(z, P2_2r3D)
	            q = evalpoly(z, Q2_2r3N) / evalpoly(z, Q2_2r3D)
            end
        end
        
        p = 1.0 + z * p
        q = z * xinv * q
        q = q - 0.125 * xinv
        c = cos(x)
        s = sin(x)
        ss = s - c
        cc = s + c
        z = -cos(x + x)
        if (s * c) < 0
            cc = z / ss;
        else
            ss = z / cc;
        end
        z = ONEOSQPI(T) * (p * cc - q * ss) / sqrt(x)
        return z
    end
end

