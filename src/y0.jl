#     Bessel function of the second kind, order zero
#
# Returns Bessel function of the second kind, of order
# zero, of the argument.
#
# The domain is divided into the intervals [0, 5] and
# (5, infinity). In the first interval a rational approximation
# R(x) is employed to compute
#   y0(x)  = R(x)  +   2 * log(x) * j0(x) / PI.
# Thus a call to j0() is required.
#
# In the second interval, the Hankel asymptotic expansion
# is employed with two rational functions of degree 6/6
# and 7/7.
#
#
# Cephes Math Library Release 2.8:  June, 2000
# Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
# http://www.netlib.org/cephes/
#
# Ported to Julia in December 2021 by Michael Helton
#
function bessely0(x)
    if x <= 5.0
        if x <= 0.0
            return NaN64
        end
        z = x * x

        YP = (
            -1.84950800436986690637E16, 4.42733268572569800351E16, -3.46628303384729719441E15,
            8.75906394395366999549E13, -9.82136065717911466409E11, 5.43526477051876500413E9,
            -1.46639295903971606143E7, 1.55924367855235737965E4
        )
        YQ = (
            2.50596256172653059228E17, 3.17157752842975028269E15, 2.02979612750105546709E13,
            8.64002487103935000337E10, 2.68919633393814121987E8, 6.26107330137134956842E5,
            1.04128353664259848412E3, 1.00000000000000000000E0
        )
    
        w = evalpoly(z, YP) / evalpoly(z, YQ)
        w += 0.6366197723675814 * log(x) * besselj0(x)
        return w
    else
        w = 5.0 / x
        z = 25.0 / (x * x)

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

        p = evalpoly(z, PP) / evalpoly(z, PQ)

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

        q = evalpoly(z, QP) / evalpoly(z, QQ)
        xn = x - 0.78539816339744830962
        p = p * sin(xn) + w * q * cos(xn);
        return p * .79788456080286535588 / sqrt(x)
    end
end


function bessely0(x::Float32)
    if x <= 2.0f0
        if x <= 0.0f0
            if x == 0.0f0
                return -Inf32
            end
            return NaN32
        end

        z = x * x
        YZ1 =  0.43221455686510834878f0
        YP = (
            1.707584643733568f-1, -1.584289289821316f-2, 5.344486707214273f-4,
            -9.413212653797057f-6, 9.454583683980369f-8
        )
 
        w = (z - YZ1) * evalpoly(z, YP)
        w += 0.636619772367581343075535f0 * log(x) * besselj0(x)
        return w
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
        xn = q * evalpoly(w, PH) - .78539816339744830962f0
        p = p * sin(xn + x)
        return p
    end
end
