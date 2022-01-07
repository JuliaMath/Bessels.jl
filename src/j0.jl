#     Bessel function of the first kind, order zero
#
# Returns Bessel function of the first kind, of order
# zero, of the argument.
#
# The domain is divided into the intervals [0, 5] and
# (5, infinity). In the first interval the following rational
# approximation is used:
#
# (w - r₁²)(w - r₂²) P₃(w) / Q₈(w)
# where w = x² and the two r's are zeros of the function
#
# In the second interval, the Hankel asymptotic expansion
# is employed with two rational functions of degree 6/6
# and 7/7.
#
# Cephes Math Library Release 2.8:  June, 2000
# Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
# http://www.netlib.org/cephes/
#
#
function besselj0(x::Float64)
    if x <= 5.0
        z = x * x
        if x < 1.0e-5
            return 1.0 - z / 4.0
        end
        
        DR1 = 5.78318596294678452118E0
        DR2 = 3.04712623436620863991E1
        RP = (
            9.70862251047306323952E15, -2.49248344360967716204E14, 
            1.95617491946556577543E12, -4.79443220978201773821E9
            )
        RQ = (
            1.71086294081043136091E18, 3.18121955943204943306E16, 3.10518229857422583814E14, 
            2.11277520115489217587E12, 1.11855537045356834862E10, 4.84409658339962045305E7, 
            1.73785401676374683123E5, 4.99563147152651017219E2, 1.00000000000000000000E0
            )

        p = (z - DR1) * (z - DR2)
        p = p * evalpoly(z, RP) / evalpoly(z, RQ)
        return p
    else
        w = 5.0 / x
        q = 25.0 / (x * x)

        PP = (
            9.99999999999999997821E-1, 5.30324038235394892183E0, 8.74716500199817011941E0, 
            5.44725003058768775090E0, 1.23953371646414299388E0, 8.28352392107440799803E-2, 
            7.96936729297347051624E-4
            )
        PQ = (
            1.00000000000000000218E0, 5.30605288235394617618E0, 8.76190883237069594232E0, 
            5.47097740330417105182E0, 1.25352743901058953537E0, 8.56288474354474431428E-2, 
            9.24408810558863637013E-4
            )

        p = evalpoly(q, PP) / evalpoly(q, PQ)

        QP = (
            -6.05014350600728481186E0, -5.14105326766599330220E1, -1.47077505154951170175E2, 
            -1.77681167980488050595E2, -9.32060152123768231369E1, -1.95539544257735972385E1, 
            -1.28252718670509318512E0, -1.13663838898469149931E-2
            )
        QQ = (
            2.42005740240291393179E2, 2.06209331660327847417E3, 5.93072701187316984827E3, 
            7.24046774195652478189E3, 3.88240183605401609683E3, 8.56430025976980587198E2, 
            6.43178256118178023184E1, 1.00000000000000000000E0
            )

        q = evalpoly(q, QP) / evalpoly(q, QQ)
        xn = x - PIO4(Float64)
        p = p * cos(xn) - w * q * sin(xn)
        return p * SQ2OPI(Float64) / sqrt(x)
    end
end


function besselj0(x::Float32)
    if x < 0.0f0
        x *= -1
    end

    if x <= 2.0f0
        z = x * x
        if x < 1.0f-3
            return 1.0f0 - 0.25f0 * z
        end

        JP = (
            -1.729150680240724f-1, 1.332913422519003f-2, -3.969646342510940f-4,
            6.388945720783375f-6, -6.068350350393235f-8
        )

        DR1 = 5.78318596294678452118f0
        p = (z - DR1) * evalpoly(z, JP)
        return p
    else
        q = 1.0f0 / x
        w = sqrt(q)

        MO = (
            7.978845717621440f-1, -3.355424622293709f-6, -4.969382655296620f-2,
            -3.560281861530129f-3, 1.197549369473540f-1, -2.145007480346739f-1,
            1.864949361379502f-1, -6.838999669318810f-2
        )

        PH = (
            -1.249992184872738f-1,  6.490598792654666f-2, -1.939906941791308f-1,
            1.001973420681837f0, -4.974978466280903f0, 1.756221482109099f1,
            -3.630592630518434f1, 3.242077816988247f1,
        )

        p = w * evalpoly(q, MO)
        w = q * q
        xn = q * evalpoly(w, PH) - PIO4(Float32)
        p = p * cos(xn + x)
        return p
    end
end

# DESCRIPTION:
#
# Returns Bessel function of first kind, order zero of the argument.
#
# The domain is divided into two major intervals [0, 2] and
# (2, infinity). In the first interval the rational approximation
# is J0(x) = 1 - x^2 / 4 + x^4 R(x^2)
# The second interval is further partitioned into eight equal segments
# of 1/x.
#
# J0(x) = sqrt(2/(pi x)) (P0(x) cos(X) - Q0(x) sin(X)),
# X = x - pi/4,
#
# and the auxiliary functions are given by
#
# J0(x)cos(X) + Y0(x)sin(X) = sqrt( 2/(pi x)) P0(x),
# P0(x) = 1 + 1/x^2 R(1/x^2)
#
# Y0(x)cos(X) - J0(x)sin(X) = sqrt( 2/(pi x)) Q0(x),
# Q0(x) = 1/x (-.125 + 1/x^2 R(1/x^2))
#
function besselj0(x::BigFloat)
    xx = abs(x)
    if iszero(xx)
        return one(BigFloat)
    elseif xx <= 2.0
        z = xx * xx
        p = z * z * evalpoly(z, J0_2N) / evalpoly(z, J0_2D)
        p -= 0.25 * z
        p += 1.0
        return p
    else
        xinv = 1.0 / xx
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
        c = cos(xx)
        s = sin(xx)
        ss = s - c
        cc = s + c
        z = -cos(xx + xx)
        if (s * c) < 0
            cc = z / ss;
        else
            ss = z / cc;
        end
        z = ONEOSQPI(BigFloat) * (p * cc - q * ss) / sqrt(xx)
        return z
    end
end
