#    Modified Bessel functions of the second kind of order zero and one
#                       besselk0, besselk1
#
#    Scaled modified Bessel functions of the second kind of order zero and one
#                       besselk0x, besselk1x
#
#    (Scaled) Modified Bessel functions of the second kind of order nu
#                       besselk, besselkx
#
#
#    Calculation of besselk0 is done in two branches using polynomial approximations [1]
#
#    Branch 1: x < 1.0 
#              besselk0(x) + log(x)*besseli0(x) = P7(x^2)
#                            besseli0(x) = [x/2]^2 * P6([x/2]^2) + 1
#    Branch 2: x >= 1.0
#              sqrt(x) * exp(x) * besselk0(x) = P22(1/x) / Q2(1/x)
#    where P7, P6, P22, and Q2 are 7, 6, 22, and 2 degree polynomials respectively.
#
#
#
#    Calculation of besselk1 is done in two branches using polynomial approximations [2]
#
#    Branch 1: x < 1.0 
#              besselk1(x) - log(x)*besseli1(x) - 1/x = x*P8(x^2)
#                            besseli1(x) = [x/2]^2 * (1 + 0.5 * (*x/2)^2 + (x/2)^4 * P5([x/2]^2))
#    Branch 2: x >= 1.0
#              sqrt(x) * exp(x) * besselk1(x) = P22(1/x) / Q2(1/x)
#    where P8, P5, P22, and Q2 are 8, 5, 22, and 2 degree polynomials respectively.
#
#
#    The polynomial coefficients are taken from boost math library [3].
#    Evaluation of the coefficients using Remez.jl is prohibitive due to the increase
#    precision required in ArbNumerics.jl. 
#
#    Horner's scheme is then used to evaluate all polynomials.
#
#    Calculation of besselk and besselkx can be done with recursion starting with
#    besselk0 and besselk1 and using upward recursion for any value of nu (order).
#
#                    K_{nu+1} = (2*nu/x)*K_{nu} + K_{nu-1}
#
#    When nu is large, a large amount of recurrence is necesary.
#    We consider uniform asymptotic expansion for large orders to more efficiently
#    compute besselk(nu, x) when nu is larger than 100 (Let's double check this cutoff)
#    The boundary should be carefully determined for accuracy and machine roundoff.
#    We use 10.41.4 from the Digital Library of Math Functions [5].
#    This is also 9.7.8 in Abramowitz and Stegun [6].
#    K_{nu}(nu*z) = sqrt(pi / 2nu) *exp(-nu*n)/(1+z^2)^1/4 * sum((-1^k)U_k(p) /nu^k)) for k=0 -> infty
#    The U polynomials are the most tricky. They are listed up to order 4 in Table 9.39
#    of [6]. For Float32, >=4 U polynomials are usually necessary. For Float64 values,
#    >= 8 orders are needed. However, this largely depends on the cutoff of order you need.
#    For moderatelly sized orders (nu=50), might need 11-12 orders to reach good enough accuracy
#    in double precision. 
#
#    However, calculation of these higher order U polynomials are tedious. These have been hand
#    calculated and somewhat crosschecked with symbolic math. There could be errors. They are listed
#    in src/U_polynomials.jl as a reference as higher orders are impossible to find while being needed for any meaningfully accurate calculation.
#    For large orders these formulas will converge much faster than using upward recurrence.
#
#    
# [1] "Rational Approximations for the Modified Bessel Function of the Second Kind 
#     - K0(x) for Computations with Double Precision" by Pavel Holoborodko     
# [2] "Rational Approximations for the Modified Bessel Function of the Second Kind 
#     - K1(x) for Computations with Double Precision" by Pavel Holoborodko
# [3] https://github.com/boostorg/math/tree/develop/include/boost/math/special_functions/detail
# [4] "Computation of Bessel Functions of Complex Argument and Large Order" by Donald E. Amos
#      Sandia National Laboratories
# [5] https://dlmf.nist.gov/10.41
# [6] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical functions with formulas, graphs, and mathematical tables. 
#     Vol. 55. US Government printing office, 1964.
#

"""
    besselk0(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order zero, ``K_0(x)``.

See also: [`besselk0x(x)`](@ref Bessels.besselk0x), [`besselk1(x)`](@ref Bessels.besselk1), [`besselk(nu,x)`](@ref Bessels.besselk))
"""
function besselk0(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        a = x * x / 4
        s = muladd(evalpoly(a, P1_k0(T)), inv(evalpoly(a, Q1_k0(T))), T(Y_k0))
        a = muladd(s, a, 1)
        return muladd(-a, log(x), evalpoly(x * x, P2_k0(T)))
    else
        s = exp(-x / 2)
        a = muladd(evalpoly(inv(x), P3_k0(T)), inv(evalpoly(inv(x), Q3_k0(T))), one(T)) * s / sqrt(x)
        return a * s
    end
end
"""
    besselk0x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the second kind of order zero, ``K_0(x)*e^{x}``.

See also: [`besselk0(x)`](@ref Bessels.besselk0), [`besselk1x(x)`](@ref Bessels.besselk1x), [`besselk(nu,x)`](@ref Bessels.besselk))
"""
function besselk0x(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        a = x * x / 4
        s = muladd(evalpoly(a, P1_k0(T)), inv(evalpoly(a, Q1_k0(T))), T(Y_k0))
        a = muladd(s, a, 1)
        return muladd(-a, log(x), evalpoly(x * x, P2_k0(T))) * exp(x)
    else
        return muladd(evalpoly(inv(x), P3_k0(T)), inv(evalpoly(inv(x), Q3_k0(T))), one(T)) / sqrt(x)
    end
end
"""
    besselk1(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order one, ``K_1(x)``.

See also: [`besselk0(x)`](@ref Bessels.besselk0), [`besselk1x(x)`](@ref Bessels.besselk1x), [`besselk(nu,x)`](@ref Bessels.besselk))
"""
function besselk1(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        z = x * x
        a = z / 4
        pq = muladd(evalpoly(a, P1_k1(T)), inv(evalpoly(a, Q1_k1(T))), Y_k1(T))
        pq = muladd(pq * a, a, (a / 2 + 1))
        a = pq * x / 2
        pq = muladd(evalpoly(z, P2_k1(T)) / evalpoly(z, Q2_k1(T)), x, inv(x))
        return muladd(a, log(x), pq)
    else
        s = exp(-x / 2)
        a = muladd(evalpoly(inv(x), P3_k1(T)), inv(evalpoly(inv(x), Q3_k1(T))), Y2_k1(T)) * s / sqrt(x)
        return a * s
    end
end
"""
    besselk1x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the second kind of order one, ``K_1(x)*e^{x}``.

See also: [`besselk1(x)`](@ref Bessels.besselk1), [`besselk(nu,x)`](@ref Bessels.besselk))
"""
function besselk1x(x::T) where T <: Union{Float32, Float64}
    x <= zero(T) && return throw(DomainError(x, "`x` must be nonnegative."))
    if x <= one(T)
        z = x * x
        a = z / 4
        pq = muladd(evalpoly(a, P1_k1(T)), inv(evalpoly(a, Q1_k1(T))), Y_k1(T))
        pq = muladd(pq * a, a, (a / 2 + 1))
        a = pq * x / 2
        pq = muladd(evalpoly(z, P2_k1(T)) / evalpoly(z, Q2_k1(T)), x, inv(x))
        return muladd(a, log(x), pq) * exp(x)
    else
        return muladd(evalpoly(inv(x), P3_k1(T)), inv(evalpoly(inv(x), Q3_k1(T))), Y2_k1(T)) / sqrt(x)
    end
end

#####
#####  Implementation for complex arguments
#####
using Base.MathConstants: γ

function besselk0(z::ComplexF64)
    isconj = false
    if imag(z) < 0.0
        z = conj(z)
        isconj = true
    end
    rez, imz = reim(z)
    if abs2(z) > 306.25
        # use asymptotic expansion for large arguments |z| > 17.5
        zinv = 1 / z
        e = exp(-z)
        sinv = sqrt(zinv) * SQPIO2(Float64)
        p = evalpoly(zinv*zinv, (1.0, 0.0703125, 0.112152099609375, 0.5725014209747314, 6.074042001273483, 110.01714026924674, 3038.090510922384, 118838.42625678325, 6.252951493434797e6, 4.259392165047669e8, 3.646840080706556e10, 3.8335346613939443e12, 4.8540146868529006e14, 7.286857349377656e16))
        p2 = evalpoly(zinv*zinv, (0.125, 0.0732421875, 0.22710800170898438, 1.7277275025844574, 24.380529699556064, 551.3358961220206, 18257.755474293175, 832859.3040162893, 5.0069589531988926e7, 3.8362551802304335e9, 3.6490108188498334e11, 4.218971570284097e13, 5.827244631566907e15, 9.47628809926011e17))
        r = muladd(p2, -zinv, p) * sinv * e
    elseif imz < (0.6(rez-1.1)+0.1) #abs(angle(z)) < pi/5
        _z = complex(abs(rez), imz)
        s = exp(-_z / 2)
        a = muladd(evalpoly(inv(_z), P3_k0(Float64)), inv(evalpoly(inv(_z), Q3_k0(Float64))), one(Float64)) * s / sqrt(_z)
        r =  a * s
    elseif imz < (2-1.1*(rez-0.8)) && rez > -3.0 || abs(z) < 2.0
        logx = log(z)
            log2 = log(2)
            P = (
            (-γ + log2 - logx),
            (1 - γ + log2 - logx) * 1//4,
            (3 - 2*γ + 2*log2 - 2*logx) * 1//128,
            (11 - 6*γ + 6*log2 - 6*logx)* 1//13824,
            (25 - 12*γ + 12*log2 - 12*logx) * 1//1769472,
            (137 - 60*γ + 60*log2 - 60*logx) * 1//884736000, 
            (49 - 20*γ + 20*log2 - 20*logx) * 1//42467328000,
            (363 - 140*γ + 140*log2 - 140*logx) * 1//58265174016000,
            (761 - 280*γ + 280*log2 - 280*logx) * 1//29831769096192000,
            (7129 - 2520*γ + 2520*log2 - 2520*logx) * 1//86989438684495872000,
            (7381 - 2520*γ + 2520*log2 - 2520*logx) * 1//34795775473798348800000,
            (83711 - 27720*γ + 27720*log2 - 27720*logx) * 1//185252708622502409011200000,
            (86021 - 27720*γ + 27720*log2 - 27720*logx) * 1//106705560166561387590451200000,
            (1145993 - 360360*γ + 360360*log2 - 360360*logx) * 1//937728462743741474144885145600000,
            (1171733 - 360360*γ + 360360*log2 - 360360*logx) * 1//735179114791093315729589954150400000,
            (1195757 - 360360*γ + 360360*log2 - 360360*logx) * 1.511347491729062e-39, # 1//661661203311983984156630958735360000000
            (2436559 - 720720*γ + 720720*log2 - 720720*logx) * 7.379626424458311e-43, # 1//1355082144382943199552780203490017280000000
            (42142223 - 12252240*γ + 12252240*log2 - 12252240*logx) * 3.755152872205532e-47, #  1//26630074301413599757611236558985819586560000000
            (14274301 - 4084080*γ + 4084080*log2 - 4084080*logx) * 8.692483500475768e-50, # 1//11504192098210675095288054193481874061393920000000,
            (275295799 - 77597520*γ + 77597520*log2 - 77597520*logx) * 3.168276534653655e-54, # 1//315629014406508081914323054852368696748403589120000000
            (55835135 - 15519504*γ + 15519504*log2 - 15519504*logx) * 9.900864170792672e-57, # 1//101001284610082586212583377552757982959489148518400000000
            (18858053 - 5173168*γ + 5173168*log2 - 5173168*logx) * 1.6838204372096382e-59, #1//59388755350728560692999026001021693980179619328819200000000
            (19093197 - 5173168*γ + 5173168*log2 - 5173168*logx) * 8.697419613686147e-63, #1//114976630359010493501646114337977999545627743020593971200000000
            (444316699 - 118982864*γ + 118982864*log2 - 118982864*logx) * 1.7870920550846854e-67, # 1//5595682646312322697738113092600713281886610997326267390361600000000
            (1347822955 - 356948592*γ + 356948592*log2 - 356948592*logx) * 2.585491977842427e-71, #1//38677358451310774486765837696056130204400255213519160202179379200000000
            (34052522467 - 8923714800*γ + 8923714800*log2 - 8923714800*logx) * 4.136787164547883e-76, #1//2417334903206923405422864856003508137775015950844947512636211200000000000000
            (34395742267 - 8923714800*γ + 8923714800*log2 - 8923714800*logx) * 1.529876909965933e-79, #1//6536473578271520888263426570633486004543643131084738074168315084800000000000000
            (312536252003 - 80313433200*γ + 80313433200*log2 - 80313433200*logx) * 5.829434956431691e-84, #1//171543212588157794191585366919705206703243370332187866018473261085491200000000000000
            (315404588903 - 80313433200*γ + 80313433200*log2 - 80313433200*logx) * 1.8588759427396975e-87, #1//537959514676462842584811710660195528221371209361741147833932146764100403200000000000000
            (9227046511387 - 2329089562800*γ + 2329089562800*log2 - 2329089562800*logx) * 1.9054450190041593e-92, #1//52481178413777009071203891245166034951164089700494019418087084509718578934579200000000000000 
            (9304682830147 - 2329089562800*γ + 2329089562800*log2 - 2329089562800*logx) * 5.2929028305671086e-96 #1//188932242289597232656334008482597725824190722921778469905113504234986884164485120000000000000000
            )
            r = evalpoly(z*z, P)
    
    elseif abs(angle(z)) > 3pi/6 && rez < -3.0
            _z = -z
            s = exp(-_z / 2)
            a = muladd(evalpoly(inv(_z), P3_k0(Float64)), inv(evalpoly(inv(_z), Q3_k0(Float64))), one(Float64)) * s / sqrt(_z)
            r =  a * s
            r -= im*pi*besseli0(z)
            

    elseif rez < 0.5 && rez > -4.0
        if imz < 3.5
            logx = log(z)
            log2 = log(2)
            P = (
            (-γ + log2 - logx),
            (1 - γ + log2 - logx) * 1//4,
            (3 - 2*γ + 2*log2 - 2*logx) * 1//128,
            (11 - 6*γ + 6*log2 - 6*logx)* 1//13824,
            (25 - 12*γ + 12*log2 - 12*logx) * 1//1769472,
            (137 - 60*γ + 60*log2 - 60*logx) * 1//884736000, 
            (49 - 20*γ + 20*log2 - 20*logx) * 1//42467328000,
            (363 - 140*γ + 140*log2 - 140*logx) * 1//58265174016000,
            (761 - 280*γ + 280*log2 - 280*logx) * 1//29831769096192000,
            (7129 - 2520*γ + 2520*log2 - 2520*logx) * 1//86989438684495872000,
            (7381 - 2520*γ + 2520*log2 - 2520*logx) * 1//34795775473798348800000,
            (83711 - 27720*γ + 27720*log2 - 27720*logx) * 1//185252708622502409011200000,
            (86021 - 27720*γ + 27720*log2 - 27720*logx) * 1//106705560166561387590451200000,
            (1145993 - 360360*γ + 360360*log2 - 360360*logx) * 1//937728462743741474144885145600000,
            (1171733 - 360360*γ + 360360*log2 - 360360*logx) * 1//735179114791093315729589954150400000,
            (1195757 - 360360*γ + 360360*log2 - 360360*logx) * 1.511347491729062e-39, # 1//661661203311983984156630958735360000000
            (2436559 - 720720*γ + 720720*log2 - 720720*logx) * 7.379626424458311e-43, # 1//1355082144382943199552780203490017280000000
            (42142223 - 12252240*γ + 12252240*log2 - 12252240*logx) * 3.755152872205532e-47, #  1//26630074301413599757611236558985819586560000000
            (14274301 - 4084080*γ + 4084080*log2 - 4084080*logx) * 8.692483500475768e-50, # 1//11504192098210675095288054193481874061393920000000,
            (275295799 - 77597520*γ + 77597520*log2 - 77597520*logx) * 3.168276534653655e-54, # 1//315629014406508081914323054852368696748403589120000000
            (55835135 - 15519504*γ + 15519504*log2 - 15519504*logx) * 9.900864170792672e-57, # 1//101001284610082586212583377552757982959489148518400000000
            (18858053 - 5173168*γ + 5173168*log2 - 5173168*logx) * 1.6838204372096382e-59, #1//59388755350728560692999026001021693980179619328819200000000
            (19093197 - 5173168*γ + 5173168*log2 - 5173168*logx) * 8.697419613686147e-63, #1//114976630359010493501646114337977999545627743020593971200000000
            (444316699 - 118982864*γ + 118982864*log2 - 118982864*logx) * 1.7870920550846854e-67, # 1//5595682646312322697738113092600713281886610997326267390361600000000
            (1347822955 - 356948592*γ + 356948592*log2 - 356948592*logx) * 2.585491977842427e-71, #1//38677358451310774486765837696056130204400255213519160202179379200000000
            (34052522467 - 8923714800*γ + 8923714800*log2 - 8923714800*logx) * 4.136787164547883e-76, #1//2417334903206923405422864856003508137775015950844947512636211200000000000000
            (34395742267 - 8923714800*γ + 8923714800*log2 - 8923714800*logx) * 1.529876909965933e-79, #1//6536473578271520888263426570633486004543643131084738074168315084800000000000000
            (312536252003 - 80313433200*γ + 80313433200*log2 - 80313433200*logx) * 5.829434956431691e-84, #1//171543212588157794191585366919705206703243370332187866018473261085491200000000000000
            (315404588903 - 80313433200*γ + 80313433200*log2 - 80313433200*logx) * 1.8588759427396975e-87, #1//537959514676462842584811710660195528221371209361741147833932146764100403200000000000000
            (9227046511387 - 2329089562800*γ + 2329089562800*log2 - 2329089562800*logx) * 1.9054450190041593e-92, #1//52481178413777009071203891245166034951164089700494019418087084509718578934579200000000000000 
            (9304682830147 - 2329089562800*γ + 2329089562800*log2 - 2329089562800*logx) * 5.2929028305671086e-96 #1//188932242289597232656334008482597725824190722921778469905113504234986884164485120000000000000000
            )
            r = evalpoly(z*z, P)
        else
            rez, imz = reim(z)
            z = complex(imz, -rez)
            if imz < 4.5
                r = evalpoly(z - 4.0, (-0.39714980986384735, 0.06604332802354913, 0.19031948892898004, -0.02617922741442789, -0.012327254937700382, 0.0013954187466492201, 0.0003383564874916576, -3.2352699991643884e-5, -5.19447498672009e-6, 4.288219153645963e-7, 5.110006887219985e-8, -3.706408095359726e-9, -3.498992638539183e-10, 2.261387420545399e-11, 1.764093969853949e-12, -1.0276029888008532e-13, -6.822065455052696e-15, 3.615771894851861e-16, 2.087643213976906e-17, -1.0147714299511874e-18, -5.180949462623928e-20, 2.325268708649643e-21, 1.0636683330175965e-22, -4.433363402376956e-24, -1.8364438496343838e-25, 7.144077519453622e-27, 2.703432663739215e-28, -9.858964808813052e-30, -3.433416796708309e-31, 1.1783395850758042e-32, 3.8002747615197184e-34))
                r2 = evalpoly(z - 4.0, (-0.016940739325064992, -0.3979257105571002, 0.05821108348217002, 0.057324968651032725, -0.007309236275643215, -0.002132039720950118, 0.00021010807896077566, 4.924704950509383e-5, -5.006687592154388e-6, -3.0093149681690976e-7, -1.3457910817518403e-8, 1.7173192411758812e-8, -3.2636233744996874e-9, 6.646264536833958e-10, -1.5994583734375563e-10, 3.8041073427550295e-11, -8.941754818236665e-12, 2.1110944642482933e-12, -5.003107469953927e-13, 1.1885823025420386e-13, -2.8301100393443645e-14, 6.753165466283357e-15, -1.6146102810126576e-15, 3.867390388035754e-16, -9.278970827415042e-17, 2.2297719204878757e-17, -5.366021041829309e-18, 1.2931015155738067e-18, -3.1200550260941356e-19, 7.537128356690712e-20, -1.8227680742788804e-20, 4.4127508163579844e-21, -1.0693331892932897e-21, 2.5936881233752025e-22, -6.296525282859723e-23, 1.5298254245756635e-23, -3.7198153277926595e-24, 9.05152571446107e-25, -2.2040769261100757e-25, 5.370578832191346e-26, -1.309455542285776e-26, 3.194645307301184e-27, -7.798383171027542e-28, 1.9046937507792232e-28, -4.65450801477231e-29, 1.1379954038230175e-29, -2.783658935012456e-30, 6.812265923313088e-31, -1.667857780095347e-31, 4.085174161348375e-32, -1.0010113785129352e-32, 2.453790517095978e-33, -6.017270382201738e-34, 1.4761110876851368e-34, -3.622348767426364e-35, 8.892169415934248e-36, -2.183565717635759e-36, 5.363656720329155e-37, -1.3179143791442321e-37, 3.239220486364871e-38, -7.963733407537875e-39))
            elseif imz < 8.5
                r = evalpoly(z - 7.0, (0.3000792705195556, 0.004682823482345833, -0.15037412265137393, 0.0063961298978456975, 0.0117901293571031, -0.0005931489739085397, -0.00035284910023739957, 1.7226126084364155e-5, 5.6607461669298285e-6, -2.5797924015210036e-7, -5.7071477461194966e-8, 2.405527610719961e-9, 3.9654842151877684e-10, -1.5448890579256987e-11, -2.0176640921954225e-12, 7.282719360974481e-14, 7.849059918114502e-15, -2.6338529522589484e-16, -2.4114034881541882e-17, 7.550478070536054e-19, 6.00042409963671e-20, -1.7595225055714948e-21, -1.2341622420258064e-22, 3.400866475630264e-24, 2.1334826847338247e-25, -5.542486611309409e-27, -3.14340708324299e-28, 7.721501146318583e-30, 3.994514803313029e-31, -9.303265355325922e-33, -4.42301370247718e-34))
                r2 = evalpoly(z - 7.0, (-0.025949743967209265, 0.30266723702418485, -0.008644216375265714, -0.04900342975478623, 0.002367537445187513, 0.002241340182398351, -0.00011239446153088169, -4.7655503764844296e-5, 2.2462773795795664e-6, 5.996453087136899e-7, -2.641223475876443e-8, -4.938403980128501e-9, 1.9782166254676283e-10, 2.9757090854331695e-11, -1.1580031492000876e-12, -1.2187257230194903e-13, 3.434678546432801e-15, 5.944503400351807e-16, -3.453143127602932e-17, 1.5005650821868992e-18, -3.3625350217014626e-19, 5.392138658802721e-20, -7.089091543697839e-21, 9.57064964202182e-22, -1.3213900410071338e-22, 1.821469122769096e-23, -2.5090711984187393e-24, 3.4610669426905155e-25, -4.780101529873834e-26, 6.608430741159359e-27, -9.144810903930076e-28, 1.2666282357364091e-28, -1.755904904416584e-29, 2.4361887914779248e-30, -3.382682803901389e-31, 4.700406478209497e-32, -6.536087322768435e-33, 9.094818744341463e-34, -1.2663403166258384e-34, 1.764307460213159e-35, -2.4595384727692307e-36, 3.4306568518107046e-37, -4.787785980439381e-38, 6.685225705699862e-39, -9.339250223791987e-40, 1.3053103341924463e-40, -1.8252111333542517e-41, 2.553302191768378e-42, -3.573335103660781e-43, 5.002887535681805e-44, -7.007072731832788e-45, 9.81782252798388e-46, -1.3761015975881949e-46, 1.929465420641253e-47, -2.7062590995458776e-48, 3.797017371135204e-49, -5.329075117245104e-50, 7.481567428238697e-51, -1.0506559093101285e-51, 1.4758813801846077e-52, -2.0737743523218415e-53))
            elseif imz < 11.5
                r = evalpoly(z - 10.0, (-0.24593576445134835, -0.04347274616886144, 0.12514151953411723, 0.003001619133391562, -0.010291308511440292, 4.751612657505903e-5, 0.0003290785427221163, -4.8349530769202e-6, -5.5381943804055925e-6, 1.0238253943478287e-7, 5.769323465195413e-8, -1.1408677083069533e-9, -4.100529494311991e-10, 8.181453300774406e-12, 2.1201821949384996e-12, -4.157966370690044e-14, -8.344937881711169e-15, 1.5879358082667785e-16, 2.58619927010138e-17, -4.743519186210765e-19, -6.478222768805454e-20, 1.1415279905758999e-21, 1.339308120471104e-22, -2.2639457012857495e-24, -2.3246536867152943e-25, 3.768116220091341e-27, 3.4361950006830084e-28, -5.342247232985093e-30, -4.378059507841556e-31, 6.532369171961593e-33, 4.8581428358700575e-34))
                r2 = evalpoly(z - 10.0, (0.055671167283599395, -0.24901542420695388, -0.015384812431452002, 0.04160037207519579, 0.00023716833203926485, -0.002022068008165671, 2.1932149361844378e-5, 4.5699891363312016e-5, -7.795502897411804e-7, -5.958886728493458e-7, 1.1513993372107611e-8, 5.07912516456716e-9, -1.0138288316996662e-10, -3.058083964186157e-11, 6.059700915423715e-13, 1.3734340078394146e-13, -2.6588026874787217e-15, -4.774818424217788e-16, 8.900940702856347e-18, 1.330326410517835e-18, -2.4151420737820336e-20, -2.9865801221207452e-21, 4.9409010977418516e-23, 5.949260997223632e-24, -1.1859946651887276e-25, -6.764721372473432e-27, -8.235622680513344e-29, 3.44614657497597e-29, -2.3193286892848722e-30, 1.9163734750015202e-31, -1.982013656367525e-32, 1.961406249627383e-33, -1.893495350554107e-34, 1.8380681625338814e-35, -1.7900599990205734e-36, 1.7435731745077013e-37, -1.6989507580298607e-38, 1.656523171811071e-39, -1.6161041705913943e-40, 1.5775284621796304e-41, -1.5406812122381703e-42, 1.5054557990345176e-43, -1.4717514071809113e-44, 1.4394750961139824e-45, -1.4085411690774174e-46, 1.3788703709267339e-47, -1.3503893206882379e-48, 1.3230300103541812e-49, -1.2967293430366117e-50, 1.2714287148742005e-51, -1.2470736371996222e-52, 1.22361339443516e-53, -1.2010007340451249e-54, 1.1791915853152485e-55, -1.1581448040474841e-56, 1.1378219405622267e-57, -1.118187028675933e-58, 1.099206393568488e-59, -1.080848476674836e-60, 1.0630836759319859e-61, -1.0458841998878446e-62))
            elseif imz < 14.5
                r = evalpoly(z - 13.0, (0.20692610237706782, 0.07031805212177837, -0.10616759165475616, -0.00892808222232259, 0.008911624226865098, 0.00030633335736580117, -0.00029379837605402224, -4.243985960944521e-6, 5.111264894811495e-6, 2.334320542560391e-8, -5.478056180434088e-8, 4.428644411870052e-11, 3.982782274431613e-10, -1.5518869433024987e-12, -2.0962106963067093e-12, 1.1997655419738472e-14, 8.3663953608489e-15, -5.700114155181254e-17, -2.6216054600879916e-17, 1.9537138326805023e-19, 6.625117044604972e-20, -5.172603589788702e-22, -1.3794951394555679e-22, 1.100756912484797e-24, 2.408449879552199e-25, -1.93423576653241e-27, -3.5773306530412485e-28, 2.8629880731192636e-30, 4.576360714471254e-31, -3.62565930854413e-33, -5.095559260340188e-34))
                r2 = evalpoly(z - 13.0, (-0.07820786452787591, 0.2100814084206935, 0.031023878093911283, -0.03560187124765109, -0.0018780447590556434, 0.0017763429395017047, 4.002020417392779e-5, -4.149293187679105e-5, -3.6188452170330305e-7, 5.582783329202643e-7, 8.349450362329919e-10, -4.880579237680724e-9, 1.2481042755288134e-11, 2.998782785840972e-11, -1.4776721310619938e-13, -1.36762062690769e-13, 8.668621850621434e-16, 4.818318180741763e-16, -3.458155137877304e-18, -1.3518294096613095e-18, 1.0351136762340261e-20, 3.093678160904443e-21, -2.44834227785507e-23, -5.886131989856995e-24, 4.7153032030525766e-26, 9.467050281680743e-27, -7.61850713124713e-29, -1.3009377326883517e-29, 1.0248185965394942e-31, 1.5627245035363336e-32, -1.2956674830778708e-34, -1.5634902534032963e-35, 8.392245873353989e-38, 1.798396587327282e-38, -3.455798893179004e-40, 5.2861239154891304e-42, -1.2189822410970054e-42, 1.0722229515597867e-43, -7.453040641921702e-45, 5.49531978161356e-46, -4.1724403629873716e-47, 3.14578661984596e-48, -2.365398927476533e-49, 1.7807545292937048e-50, -1.3415563205987565e-51, 1.0109925565932276e-52, -7.621453208446864e-54, 5.747620076286422e-55, -4.336025773407635e-56, 3.272225959262748e-57, -2.4702333392202132e-58, 1.8653986578568636e-59, -1.4090944921094982e-60, 1.0647295453636356e-61, -8.04758069877769e-63, 6.084354157252506e-64, -4.601327517608198e-65, 3.4807102190280396e-66, -2.6336940621320167e-67, 1.9932983968961283e-68, -1.5089879429646347e-69))
            else
                r = evalpoly(z - 16.0, (-0.1748990739836292, -0.09039717566130419, 0.09027444873123035, 0.01312662593374557, -0.007667362695010893, -0.0005550708142218287, 0.0002571415573791134, 1.0850297108500103e-5, -4.565690471161262e-6, -1.2026225777846727e-7, 4.9959717576483296e-8, 8.48815248845687e-10, -3.701703923085197e-10, -4.101051708969331e-12, 1.980421966657433e-12, 1.4173962555705988e-14, -8.014281597026939e-15, -3.5742021762369616e-17, 2.540522616136639e-17, 6.485026844702886e-20, -6.482772100803784e-20, -7.615209126543415e-23, 1.3608987282597849e-22, 2.2067196209386653e-26, -2.3923906484884365e-25, 1.4153681120888267e-28, 3.5743243599678463e-28, -4.139824487468731e-31, -4.595455176935322e-31, 7.2929256888043685e-34, 5.138919311361309e-34))
                r2 = evalpoly(z - 16.0, (0.0958109970807124, -0.17797516893941687, -0.042343774510999424, 0.03042882087493703, 0.003029250902296742, -0.0015405792935148154, -8.412990202878025e-5, 3.667960456395296e-5, 1.2158003374348262e-6, -5.039539866235208e-7, -1.063342850444792e-8, 4.494776306242216e-9, 6.165762767016077e-11, -2.8109651754779916e-11, -2.5095027555184863e-13, 1.3014376619478582e-13, 7.402366587975707e-16, -4.643498548292513e-16, -1.5911855984259361e-18, 1.316685883106809e-18, 2.382437599096684e-21, -3.0399945790061975e-21, -1.9165686350953203e-24, 5.828198478802957e-24, -1.4104753305083107e-27, -9.429393045351943e-27, 8.43651729135661e-30, 1.3050006977135365e-29, -1.8108987900262247e-32, -1.5628015887600736e-32, 2.7507053932510895e-35, 1.6357586932973496e-35, -3.350163923317318e-38, -1.5087758421235516e-38, 3.3924254427089285e-41, 1.2378641589060088e-41, -3.0697698225515145e-44, -9.018313764541882e-45, 2.038554081122575e-47, 6.138410827178904e-48, -2.581641654777743e-50, -2.945649677635829e-51, -2.8081847273833064e-53, 4.238759347046062e-54, -1.4675299965724711e-55, 7.713873279422344e-57, -5.287229701244321e-58, 3.3016559715175193e-59, -1.99988603375306e-60, 1.2245417312605611e-61, -7.526297277174194e-63, 4.621657471219175e-64, -2.8377793329899337e-65, 1.7431331319950407e-66, -1.0710565225239064e-67, 6.582654578728787e-69, -4.0466665627666414e-70, 2.488286761082909e-71, -1.5304073094823902e-72, 9.414872460609797e-74, -5.7932214228437515e-75))
            end
            r = -(r - im*r2) * im*pi/2
        end
    elseif imz < (0.65(rez-2.0)+0.1)#abs(angle(z)) < pi/5
        _z = complex(abs(rez), imz)
        s = exp(-_z / 2)
        a = muladd(evalpoly(inv(_z), P3_k0(Float64)), inv(evalpoly(inv(_z), Q3_k0(Float64))), one(Float64)) * s / sqrt(_z)
        r =  a * s
        
    else
             # use rational approximation based on AAA algorithm
            zz = (1.2 + 0.6579604803220245878847549647616688162088im,
            1.2 + 16.99439365044952410244150087237358093262im,
            1.2 + 3.447627683387571639883617535815574228764im,
           2.934364628128498964088066713884472846985 + 0.65im,
           13.93878851250798156513610592810437083244 + 0.65im,
            1.2 + 1.318992142890197172278021753299981355667im,
            1.2 + 6.532516845602102328882665460696443915367im,
           1.606565913616233798322241455025505274534 + 0.65im,
            13.9277457383449512207107545691542327404 + 17.0im,
           5.673369442310413290897486149333417415619 + 0.65im,
            1.2 + 1.958379542779472348712488383171148598194im,
             1.2 + 3.19303838013913354743067429808434098959im,
           1.268760953600300833699066060944460332394 + 0.65im,
           1.460619083673731166683751325763296335936 + 0.65im,
           10.08421542607476339981076307594776153564 + 0.65im)
            w = (-0.09680823500842447582037664233212126418948 + 0.0im,
            0.05423559418525394915100434900523396208882 + 0.1144965073196525795484745913199731148779im,
        - 0.007396939988654569614334732818861084524542 - 0.02778731755861096885951155854854732751846im,
          - 0.04414191750882235504294115457923908252269 - 0.2909178116618649778679639439360471442342im,
          - 0.07841499402757724779267078929478884674609 - 0.1119507888959188901534247406743816100061im,
          - 0.3877782157727173717454149937111651524901 - 0.04974618828993317143360997079071239568293im,
             0.1622645929151690924463480314443586394191 + 0.1460823764263605994973005408610333688557im,
           - 0.020469802336280332188955810579500393942 + 0.03354462595402274427414113233680836856365im,
              0.2534062532247348120684193872875766828656 - 0.171364921953992055403404037861037068069im,
            0.009835262795195880494714124608890415402129 - 0.178079191318303603486228325891715940088im,
           - 0.1656594359598556609469710565463174134493 + 0.4988334748067394519566164490242954343557im,
             0.2380784214921798647157658024298143573105 + 0.3233607593681438463484312251239316537976im,
         - 0.03397428102687283235638915357412770390511 + 0.02550506445688451376274663573440193431452im,
               0.1320100465195262195994274634358589537442 - 0.28990229940480305437233710108557716012im,
         0.0004315595380057664354200852585563552565873 - 0.01549561321829133128669120367248979164287im)

            f = (1.172826382150622182010124561202246695757 + 0.03336141554452320967527612083358690142632im,
            1.252371234797117205417293916980270296335 + 0.009112669468987186216502038860198808833957im,
             1.235526466824824476375965787156019359827 + 0.03601791107741053599156089148891624063253im,
            1.209333297545630792058091174112632870674 + 0.008445981757095136557844661240324057871476im,
            1.242519439909658629517252848017960786819 + 0.000484990485661015394581313531219279866491im,
             1.197189637455772670548981295723933726549 + 0.04560411489851770583392465141514549031854im,
             1.247383532310859299840899439004715532064 + 0.02227599877715117152043866610711120301858im,
             1.185125871280349407754783896962180733681 + 0.02201178588810389843977155521770328050479im,
            1.248767617633458559822656752658076584339 + 0.005339617830697486119961858719307201681659im,
             1.228316149050262540143307887774426490068 + 0.00263570669216164768858257083650187269086im,
              1.21490347831149314572485309327021241188 + 0.04544758993246874462235140867960581090301im,
             1.233298447226847560642681855824775993824 + 0.03766718444981393548731674059126817155629im,
              1.174897700249087906243516954418737441301 + 0.0307157965036025139282660489925547153689im,
              1.18098130606021078925493839051341637969 + 0.02526540273742965136971605488724890165031im,
            1.23862233012449962643586331978440284729 + 0.0009007487014944375600458692510130731534446im)
            f = f.*w
            s1 = 0.0 + 0.0im
            s2 = 0.0 + 0.0im
            @fastmath for ind in eachindex(f)
                C = inv(z - zz[ind])
                s1 += C*f[ind]
                s2 += C*w[ind]
            end
            r = @fastmath s1 / (s2 * sqrt(z) * exp(z))
    end
    isconj && (r = conj(r))
    return r
end

function h02(z)
    #if imag(z) >= 0.0
        r = j0(z) - im*y0(z)
    #else
    #    r = j0(z) + im*y0(z)
    #    r = 2*besseli0(z/(exp(pi*im/2))) - r
        
    #end
    return r
end

#              Modified Bessel functions of the second kind of order nu
#                           besselk(nu, x)
#
#    A numerical routine to compute the modified Bessel function of the second kind K_{ν}(x) [1]
#    for real orders and arguments of positive or negative value. The routine is based on several
#    publications [2-8] that calculate K_{ν}(x) for positive arguments and orders where
#    reflection identities are used to compute negative arguments and orders.
#
#    In particular, the reflectance identities for negative orders I_{−ν}(x) = I_{ν}(x).
#    For negative arguments of integer order, Kn(−x) = (−1)^n Kn(x) − im * π In(x) is used and for
#    noninteger orders, Kν(−x) = exp(−iπν)*Kν(x) − im π Iν(x) is used. For negative orders and arguments the previous identities are combined.
#
#    The identities are computed by calling the `besseli_positive_args(nu, x)` function which computes K_{ν}(x)
#    for positive arguments and orders. For large orders, Debye's uniform asymptotic expansions are used.
#    For large arguments x >> nu, large argument expansion is used [9].
#    For small value and when nu > ~x the power series is used. The rest of the values are computed using slightly different methods.
#    The power series for besseli is modified to give both I_{v} and I_{v-1} where the ratio K_{v+1} / K_{v} is computed using continued fractions [8].
#    The wronskian connection formula is then used to compute K_v.

# [1] http://dlmf.nist.gov/10.27.E4
# [2] Amos, Donald E. "Computation of modified Bessel functions and their ratios." Mathematics of computation 28.125 (1974): 239-251.
# [3] Gatto, M. A., and J. B. Seery. "Numerical evaluation of the modified Bessel functions I and K." 
#     Computers & Mathematics with Applications 7.3 (1981): 203-209.
# [4] Temme, Nico M. "On the numerical evaluation of the modified Bessel function of the third kind." 
#     Journal of Computational Physics 19.3 (1975): 324-337.
# [5] Amos, DEv. "Algorithm 644: A portable package for Bessel functions of a complex argument and nonnegative order." 
#     ACM Transactions on Mathematical Software (TOMS) 12.3 (1986): 265-273.
# [6] Segura, Javier, P. Fernández de Córdoba, and Yu L. Ratis. "A code to evaluate modified bessel functions based on thecontinued fraction method." 
#     Computer physics communications 105.2-3 (1997): 263-272.
# [7] Geoga, Christopher J., et al. "Fitting Mat\'ern Smoothness Parameters Using Automatic Differentiation." 
#     arXiv preprint arXiv:2201.00090 (2022).
# [8] Cuyt, A. A., Petersen, V., Verdonk, B., Waadeland, H., & Jones, W. B. (2008). 
#     Handbook of continued fractions for special functions. Springer Science & Business Media.
# [9] http://dlmf.nist.gov/10.40.E2
#

"""
    besselk(ν::Real, x::Real)
    besselk(ν::AbstractRange, x::Real)

Returns the modified Bessel function, ``K_ν(x)``, of the second kind and order `ν`.

```math
K_{\\nu}(x) = \\frac{\\pi}{2} \\frac{I_{-\\nu}(x) - I_{\\nu}(x)}{\\sin(\\nu \\pi)}
```

Routine supports single and double precision (e.g., `Float32` or `Float64`) real arguments.

For `ν` isa `AbstractRange`, returns a vector of type `float(x)` using recurrence to compute ``K_ν(x)`` at many orders
as long as the conditions `ν[1] >= 0` and `step(ν) == 1` are met. Consider the in-place version [`besselk!`](@ref Bessels.besselk!)
to avoid allocation.

# Examples

```
julia> besselk(2, 1.5)
0.5836559632566508

julia> besselk(3.2, 2.5)
0.3244950563641161

julia> besselk(1:3, 2.5)
3-element Vector{Float64}:
 0.07389081634774707
 0.12146020627856384
 0.26822714639344925
```

External links: [DLMF](https://dlmf.nist.gov/10.27.4), [Wikipedia](https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1)

See also: [`besselk!`](@ref Bessels.besselk!(out, ν, x)), [`besselk0(x)`](@ref Bessels.besselk0), [`besselk1(x)`](@ref Bessels.besselk1), [`besselkx(nu,x)`](@ref Bessels.besselkx))
"""
besselk(nu, x::Real) = _besselk(nu, float(x))

_besselk(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_besselk(Float32(nu), Float32(x)))

_besselk(nu::AbstractRange, x::T) where T = besselk!(zeros(T, length(nu)), nu, x)

function _besselk(nu::T, x::T) where T <: Union{Float32, Float64}
    isinteger(nu) && return _besselk(Int(nu), x)
    abs_nu = abs(nu)
    abs_x = abs(x)

    if x >= 0
        return besselk_positive_args(abs_nu, abs_x)
    else
        return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
        #return cispi(-abs_nu)*besselk_positive_args(abs_nu, abs_x) - besseli_positive_args(abs_nu, abs_x) * im * π
    end
end
function _besselk(nu::Integer, x::T) where T <: Union{Float32, Float64}
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    if x >= 0
        return besselk_positive_args(abs_nu, abs_x)
    else
        return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
        #return sg * besselk_positive_args(abs_nu, abs_x) - im * π * besseli_positive_args(abs_nu, abs_x)
    end
end

"""
    Bessels.besselk!(out::DenseVector{T}, ν::AbstractRange, x::T)

Computes the modified Bessel function, ``K_ν(x)``, of the second kind at many orders `ν` in-place using recurrence.
The conditions `ν[1] >= 0` and `step(ν) == 1` must be met.

# Examples

```
julia> nu = 1:3; x = 1.5; out = zeros(typeof(x), length(nu));

julia> Bessels.besselk!(out, nu, x)
3-element Vector{Float64}:
 0.2773878004568438
 0.5836559632566508
 1.8338037024745792
```

See also: [`besselk(ν, x)`](@ref Bessels.besselk(ν, x))
"""
besselk!(out::DenseVector, nu::AbstractRange, x) = _besselk!(out, nu, float(x))

function _besselk!(out::DenseVector{T}, nu::AbstractRange, x::T) where T
    (nu[1] >= 0 && step(nu) == 1) || throw(ArgumentError("nu must be >= 0 with step(nu)=1"))
    len = length(out)
    !isequal(len, length(nu)) && throw(ArgumentError("out and nu must have the same length"))
    isone(len) && return [besselk(nu[1], x)]

    k = 1
    knu = zero(T)
    while abs(knu) < floatmin(T)
        if besselk_underflow_check(nu[k], x)
            knu = zero(T)
        else
            knu = _besselk(nu[k], x)
        end
        out[k] = knu
        k += 1
        k == len && break
    end
    if k < len
        out[k] = _besselk(nu[k], x)
        tmp = @view out[k-1:end]
        besselk_up_recurrence!(tmp, x, nu[k-1:end])
        return out
    else
        return out
    end
end

besselk_underflow_check(nu, x::T) where T = nu < T(1.45)*(x - 780) + 45*Base.Math._approx_cbrt(x - 780)

"""
    besselk_positive_args(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order nu, ``K_{nu}(x)`` valid for postive arguments and orders.
"""
function besselk_positive_args(nu, x::T) where T <: Union{Float32, Float64}
    iszero(x) && return T(Inf)
    isinf(x) && return zero(T)

    # dispatch to avoid uniform expansion when nu = 0 
    iszero(nu) && return besselk0(x)
    
    # check if nu is a half-integer
    (isinteger(nu-1/2) && sphericalbesselk_cutoff(nu)) && return sphericalbesselk_int(Int(nu-1/2), x)*SQPIO2(T)*sqrt(x)

    # check if the standard asymptotic expansion can be used
    besseli_large_argument_cutoff(nu, x) && return besselk_large_argument(nu, x)

    # use uniform debye expansion if x or nu is large
    besselik_debye_cutoff(nu, x) && return besselk_large_orders(nu, x)

    # for integer nu use forward recurrence starting with K_0 and K_1
    isinteger(nu) && return besselk_up_recurrence(x, besselk1(x), besselk0(x), 1, nu)[1]

    # for small x and nu > x use power series
    besselk_power_series_cutoff(nu, x) && return besselk_power_series(nu, x)

    # for rest of values use the continued fraction approach
    return besselk_continued_fraction(nu, x)
end
"""
    besselkx(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the second kind of order nu, ``K_{nu}(x)*e^{x}``.
"""
besselkx(nu::Real, x::Real) = _besselkx(nu, float(x))

_besselkx(nu, x::Float16) = Float16(_besselkx(nu, Float32(x)))

function _besselkx(nu, x::T) where T <: Union{Float32, Float64}
    # dispatch to avoid uniform expansion when nu = 0 
    iszero(nu) && return besselk0x(x)

    # check if the standard asymptotic expansion can be used
    besseli_large_argument_cutoff(nu, x) && return besselk_large_argument_scaled(nu, x)

    # use uniform debye expansion if x or nu is large
    besselik_debye_cutoff(nu, x) && return besselk_large_orders_scaled(nu, x)

    # for integer nu use forward recurrence starting with K_0 and K_1
    isinteger(nu) && return besselk_up_recurrence(x, besselk1x(x), besselk0x(x), 1, nu)[1]

    # for small x and nu > x use power series
    besselk_power_series_cutoff(nu, x) && return besselk_power_series(nu, x) * exp(x)

    # for rest of values use the continued fraction approach
    return besselk_continued_fraction(nu, x) * exp(x)
end

#####
#####  Debye's uniform asymptotic for K_{nu}(x)
#####

# Implements the uniform asymptotic expansion https://dlmf.nist.gov/10.41
# In general this is valid when either x or nu is gets large
# see the file src/U_polynomials.jl for more details
"""
    besselk_large_orders(nu, x::T)

Debey's uniform asymptotic expansion for large order valid when v-> ∞ or x -> ∞
"""
function besselk_large_orders(v, x::T) where T
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQPIO2(S) * sqrt(inv(v)) * exp(-v*n) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_Kn(p, v, p2, T))
end
function besselk_large_orders_scaled(v, x::T) where T
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQPIO2(S) * sqrt(inv(v)) * exp(-v*n + x) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_Kn(p, v, p2, T))
end
besselik_debye_cutoff(nu, x::Float64) = nu > 25.0 || x > 35.0
besselik_debye_cutoff(nu, x::Float32) = nu > 15.0f0 || x > 20.0f0

#####
#####  Continued fraction with Wronskian for K_{nu}(x)
#####

# Use the ratio K_{nu+1}/K_{nu} and I_{nu-1}, I_{nu}
# along with the Wronskian (NIST https://dlmf.nist.gov/10.28.E2) to calculate K_{nu}
# Inu and Inum1 are generated from the power series form where K_{nu_1}/K_{nu}
# is calculated with continued fractions. 
# The continued fraction K_{nu_1}/K_{nu} method is a slightly modified form
# https://github.com/heltonmc/Bessels.jl/issues/17#issuecomment-1195726642 by @cgeoga  
# 
# It is also possible to use continued fraction to calculate inu/inmu1 such as
# inum1 = besseli_power_series(nu-1, x)
# H_inu = steed(nu, x)
# inu = besseli_power_series(nu, x)#inum1 * H_inu
# but it appears to be faster to modify the series to calculate both Inu and Inum1

function besselk_continued_fraction(nu, x)
    inu, inum1 = besseli_power_series_inu_inum1(nu, x)
    H_knu = besselk_ratio_knu_knup1(nu-1, x)
    return 1 / (x * (inum1 + inu / H_knu))
end

# a modified version of the I_{nu} power series to compute both I_{nu} and I_{nu-1}
# use this along with the continued fractions for besselk
function besseli_power_series_inu_inum1(v, x::ComplexOrReal{T}) where T
    MaxIter = 3000
    S = eltype(x)
    out = zero(S)
    out2 = zero(S)
    x2 = x / 2
    xs = x2^v
    gmx = xs / gamma(v)
    a = gmx / v
    b = gmx / x2
    t2 = x2 * x2
    for i in 0:MaxIter
        out += a
        out2 += b
        abs(a) < eps(T) * abs(out) && break
        a *= inv((v + i + one(T)) * (i + one(T))) * t2
        b *= inv((v + i) * (i + one(T))) * t2
    end
    return out, out2
end

# computes K_{nu+1}/K_{nu} using continued fractions and the modified Lentz method
# generally slow to converge for small x
besselk_ratio_knu_knup1(v, x::Float32) = Float32(besselk_ratio_knu_knup1(v, Float64(x)))
besselk_ratio_knu_knup1(v, x::ComplexF32) = ComplexF32(besselk_ratio_knu_knup1(v, ComplexF64(x)))
function besselk_ratio_knu_knup1(v, x::ComplexOrReal{T}) where T
    MaxIter = 1000
    S = eltype(x)
    (hn, Dn, Cn) = (S(1e-50), zero(S), S(1e-50))

    jf = 1.0#one(S)
    #vv = v * v
    @fastmath for _ in 1:MaxIter
        an = ( - ((2*jf - 1)^2) * (0.25))
        bn = 2 * (x + jf)
        Cn = an / Cn + bn
        Dn = inv(muladd(an, Dn, bn))
        del = Dn * Cn
        hn *= del
        abs(del - 1) < eps(T) && break
        jf += one(T)
    end
    xinv = inv(x)
    return xinv * (v + x + 1/2) + xinv * hn
end

#####
#####  Power series for K_{nu}(x)
#####

# Use power series form of K_v(x) which is accurate for small x (x<2) or when nu > x
# We use the form as described by Equation 3.2 from reference [7].
# This method was originally contributed by @cgeoga https://github.com/cgeoga/BesselK.jl/blob/main/src/besk_ser.jl
# A modified form appears below. See more discussion at https://github.com/heltonmc/Bessels.jl/pull/29
# This is only valid for noninteger orders (nu) and no checks are performed. 
#
"""
    besselk_power_series(nu, x::T) where T <: Float64

Computes ``K_{nu}(x)`` using the power series when nu is not an integer.
In general, this is most accurate for small arguments and when nu > x.
No checks are performed on nu so this is not accurate when nu is an integer.
"""
besselk_power_series(v, x::Float32) = Float32(besselk_power_series(v, Float64(x)))
besselk_power_series(v, x::ComplexF32) = ComplexF32(besselk_power_series(v, ComplexF64(x)))

function besselk_power_series(v, x::ComplexOrReal{T}) where T
    MaxIter = 1000
    S = eltype(x)
    v, x = S(v), S(x)

    z  = x / 2
    zz = z * z
    logz = log(z)
    xd2_v = exp(v*logz)
    xd2_nv = inv(xd2_v)

    # use the reflection identify to calculate gamma(-v)
    # use relation gamma(v)*v = gamma(v+1) to avoid two gamma calls
    gam_v = gamma(v)
    gam_nv = π / (sinpi(-abs(v)) * gam_v * v)
    gam_1mv = -gam_nv * v
    gam_1mnv = gam_v * v

    _t1 = gam_v * xd2_nv * gam_1mv
    _t2 = gam_nv * xd2_v * gam_1mnv
    (xd2_pow, fact_k, out) = (one(S), one(S), zero(S))
    for k in 0:MaxIter
        t1 = xd2_pow * T(0.5)
        tmp = muladd(_t1, gam_1mnv, _t2 * gam_1mv)
        tmp *= inv(gam_1mv * gam_1mnv * fact_k)
        term = t1 * tmp
        out += term
        abs(term / out) < eps(T) && break
        (gam_1mnv, gam_1mv) = (gam_1mnv*(one(S) + v + k), gam_1mv*(one(S) - v + k)) 
        xd2_pow *= zz
        fact_k *= k + one(S)
    end
    return out
end
besselk_power_series_cutoff(nu, x::Float64) = x < 2.0 || nu > 1.6x - 1.0
besselk_power_series_cutoff(nu, x::Float32) = x < 10.0f0 || nu > 1.65f0*x - 8.0f0

#####
#####  Large argument expansion for K_{nu}(x)
#####

# Computes the asymptotic expansion of K_ν w.r.t. argument. 
# Accurate for large x, and faster than uniform asymptotic expansion for small to small-ish orders
# See http://dlmf.nist.gov/10.40.E2

function besselk_large_argument(v, x::T) where T
    a = exp(-x / 2)
    coef = a * sqrt(pi / 2x)
    return T(_besselk_large_argument(v, x) * coef * a)
end

besselk_large_argument_scaled(v, x::T) where T =  T(_besselk_large_argument(v, x) * sqrt(pi / 2x))

_besselk_large_argument(v, x::Float32) = Float32(_besselk_large_argument(v, Float64(x)))
_besselk_large_argument(v, x::ComplexF32) = ComplexF32(_besselk_large_argument(v, ComplexF64(x)))
function _besselk_large_argument(v, x::ComplexOrReal{T}) where T
    MaxIter = 5000 
    S = eltype(x)
    v, x = S(v), S(x) 
 
    fv2 = 4 * v^2 
    term = one(S) 
    res = term 
    s = term 
    for i in 1:MaxIter 
        offset = muladd(2, i, -1) 
        term *= muladd(offset, -offset, fv2) / (8 * x * i) 
        res = muladd(term, s, res) 
        abs(term) <= eps(T) && break 
    end 
    return res 
end
