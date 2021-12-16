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
# Ported to Julia in December 2021 by Michael Helton
#
function besselj0(x)
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
        xn = x - 0.78539816339744830962
        p = p * cos(xn) - w * q * sin(xn)
        return p * .79788456080286535588 / sqrt(x)
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
        xn = q * evalpoly(w, PH) - 0.78539816339744830962f0
        p = p * cos(xn + x)
        return p
    end
end
