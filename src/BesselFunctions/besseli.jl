#    Modified Bessel functions of the first kind of order zero and one
#                       besseli0, besseli1
#
#    Scaled modified Bessel functions of the first kind of order zero and one
#                       besseli0x, besselix
#
#    (Scaled) Modified Bessel functions of the first kind of order nu
#                       besseli, besselix
#
#    Calculation of besseli0 is done in two branches using polynomial approximations [1]
#
#    Branch 1: x < 7.75 
#              besseli0 = [x/2]^2 P16([x/2]^2)
#    Branch 2: x >= 7.75
#              sqrt(x) * exp(-x) * besseli0(x) = P22(1/x)
#    where P16 and P22 are a 16 and 22 degree polynomial respectively.
#
#    Remez.jl is then used to approximate the polynomial coefficients of
#    P22(y) = sqrt(1/y) * exp(-inv(y)) * besseli0(inv(y))
#    N,D,E,X = ratfn_minimax(g, (1/1e6, 1/7.75), 21, 0)
#
#    A third branch is used for scaled functions for large values
#
#
#    Calculation of besseli1 is done in two branches using polynomial approximations [2]
#
#    Branch 1: x < 7.75 
#              besseli1 = x / 2 * (1 + 1/2 * (x/2)^2 + (x/2)^4 * P13([x/2]^2)
#    Branch 2: x >= 7.75
#              sqrt(x) * exp(-x) * besseli1(x) = P22(1/x)
#    where P13 and P22 are a 16 and 22 degree polynomial respectively.
#
#    Remez.jl is then used to approximate the polynomial coefficients of
#    P13(y) = (besseli1(2 * sqrt(y)) / sqrt(y) - 1 - y/2) / y^2
#    N,D,E,X = ratfn_minimax(g, (1/1e6, 1/7.75), 21, 0)
#
#    A third branch is used for scaled functions for large values
#
#    Horner's scheme is then used to evaluate all polynomials.
#    ArbNumerics.jl is used as the reference bessel implementations with 75 digits.
#
#    Calculation of besseli and besselkx can be done with downward recursion starting with
#    besseli_{nu+1} and besseli_{nu}. Higher orders are determined by a uniform asymptotic
#    expansion similar to besselk (see notes there) using Equation 10.41.3 [3].
#
# 
# [1] "Rational Approximations for the Modified Bessel Function of the First Kind 
#     - I0(x) for Computations with Double Precision" by Pavel Holoborodko     
# [2] "Rational Approximations for the Modified Bessel Function of the First Kind 
#     - I1(x) for Computations with Double Precision" by Pavel Holoborodko
# [3] https://dlmf.nist.gov/10.41

"""
    besseli0(x::T) where T <: Union{Float32, Float64, ComplexF32, ComplexF64}

Modified Bessel function of the first kind of order zero, ``I_0(x)``.

See also: [`besseli0x(x)`](@ref Bessels.besseli0x), [`besseli1(x)`](@ref Bessels.besseli1), [`besseli(nu,x)`](@ref Bessels.besseli))
"""
function besseli0(x::T) where T <: Union{Float32, Float64}
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, besseli0_small_coefs(T)), 1)
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), besseli0_med_coefs(T)) / sqrt(x)
        return a * s
    end
end

"""
    besseli0x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the first kind of order zero, ``I_0(x)*e^{-x}``.

See also: [`besseli0(x)`](@ref Bessels.besseli0), [`besseli1x(x)`](@ref Bessels.besseli1x), [`besseli(nu,x)`](@ref Bessels.besseli))
"""
function besseli0x(x::T) where T <: Union{Float32, Float64}
    T == Float32 ? branch = 50 : branch = 500
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, besseli0_small_coefs(T)), 1) * exp(-x)
    elseif x < branch
        return evalpoly(inv(x), besseli0_med_coefs(T)) / sqrt(x)
    else
        return evalpoly(inv(x), besseli0_large_coefs(T)) / sqrt(x)
    end
end

"""
    besseli1(x::T) where T <: Union{Float32, Float64, ComplexF32, ComplexF64}

Modified Bessel function of the first kind of order one, ``I_1(x)``.

See also: [`besseli0(x)`](@ref Bessels.besseli0), [`besseli1x(x)`](@ref Bessels.besseli1x), [`besseli(nu,x)`](@ref Bessels.besseli))
"""
function besseli1(x::T) where T <: Union{Float32, Float64}
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2
    else
        a = exp(z / 2)
        s = a * evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
        z =  a * s
    end
    if x < zero(x)
        z = -z
    end
    return z
end

"""
    besseli1x(x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the first kind of order one, ``I_1(x)*e^{-x}``.

See also: [`besseli1(x)`](@ref Bessels.besseli1), [`besseli(nu,x)`](@ref Bessels.besseli))
"""
function besseli1x(x::T) where T <: Union{Float32, Float64}
    T == Float32 ? branch = 50 : branch = 500
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2 * exp(-z)
    elseif z < branch
        z = evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
    else
        z = evalpoly(inv(z), besseli1_large_coefs(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end

#####
#####  Implementation for complex arguments
#####

function besseli0(z::ComplexF64)
    isconj = false
    if real(z) < 0.0
        z = -z
    end
    if imag(z) < 0.0
        z = conj(z)
        isconj = true
    end
    rez, imz = reim(z)
    if abs2(z) > 306.25
        # use asymptotic expansion for large arguments |z| > 17.5
        zinv = 1 / z
        e = exp(z)
        sinv = sqrt(zinv) * SQ1O2PI(Float64)
        p = evalpoly(zinv*zinv, (1.0, 0.0703125, 0.112152099609375, 0.5725014209747314, 6.074042001273483, 110.01714026924674, 3038.090510922384, 118838.42625678325, 6.252951493434797e6, 4.259392165047669e8, 3.646840080706556e10, 3.8335346613939443e12, 4.8540146868529006e14, 7.286857349377656e16))
        p2 = evalpoly(zinv*zinv, (0.125, 0.0732421875, 0.22710800170898438, 1.7277275025844574, 24.380529699556064, 551.3358961220206, 18257.755474293175, 832859.3040162893, 5.0069589531988926e7, 3.8362551802304335e9, 3.6490108188498334e11, 4.218971570284097e13, 5.827244631566907e15, 9.47628809926011e17))
        r = e * sinv * muladd(p2, zinv, p) + im * muladd(p2, -zinv, p) * sinv / e
    elseif rez < 4.5
        # use taylor series around the roots (slight offset) of J0(z)
        # use relation I0(z) = J0(im*z)
        _z = complex(imz, rez)
        if imz < 2.2
            # use power series for I0
            r = evalpoly(z*z, (1.0, 0.25, 0.015625, 0.00043402777777777775, 6.781684027777777e-6, 6.781684027777778e-8, 4.709502797067901e-10, 2.4028075495244395e-12, 9.385966990329842e-15, 2.896903392077112e-17, 7.242258480192779e-20, 1.4963343967340453e-22, 2.5978027721077174e-25, 3.842903509035085e-28, 4.9016626390753635e-31, 5.4462918211948485e-34))
        else
            # use taylor series along imaginary axis to compute J0 then use connection to I0 
            if imz < 5.5
                r = evalpoly(_z - 4.0, (-0.39714980986384735, 0.06604332802354913, 0.19031948892898004, -0.02617922741442789, -0.012327254937700382, 0.0013954187466492201, 0.0003383564874916576, -3.2352699991643884e-5, -5.19447498672009e-6, 4.288219153645963e-7, 5.110006887219985e-8, -3.706408095359726e-9, -3.498992638539183e-10, 2.261387420545399e-11, 1.764093969853949e-12, -1.0276029888008532e-13, -6.822065455052696e-15, 3.615771894851861e-16, 2.087643213976906e-17, -1.0147714299511874e-18, -5.180949462623928e-20, 2.325268708649643e-21, 1.0636683330175965e-22, -4.433363402376956e-24, -1.8364438496343838e-25, 7.144077519453622e-27, 2.703432663739215e-28, -9.858964808813052e-30, -3.433416796708309e-31, 1.1783395850758042e-32, 3.8002747615197184e-34))
            elseif imz < 8.5
                r = evalpoly(_z - 7.0, (0.3000792705195556, 0.004682823482345833, -0.15037412265137393, 0.0063961298978456975, 0.0117901293571031, -0.0005931489739085397, -0.00035284910023739957, 1.7226126084364155e-5, 5.6607461669298285e-6, -2.5797924015210036e-7, -5.7071477461194966e-8, 2.405527610719961e-9, 3.9654842151877684e-10, -1.5448890579256987e-11, -2.0176640921954225e-12, 7.282719360974481e-14, 7.849059918114502e-15, -2.6338529522589484e-16, -2.4114034881541882e-17, 7.550478070536054e-19, 6.00042409963671e-20, -1.7595225055714948e-21, -1.2341622420258064e-22, 3.400866475630264e-24, 2.1334826847338247e-25, -5.542486611309409e-27, -3.14340708324299e-28, 7.721501146318583e-30, 3.994514803313029e-31, -9.303265355325922e-33, -4.42301370247718e-34))
            elseif imz < 11.5
                r = evalpoly(_z - 10.0, (-0.24593576445134835, -0.04347274616886144, 0.12514151953411723, 0.003001619133391562, -0.010291308511440292, 4.751612657505903e-5, 0.0003290785427221163, -4.8349530769202e-6, -5.5381943804055925e-6, 1.0238253943478287e-7, 5.769323465195413e-8, -1.1408677083069533e-9, -4.100529494311991e-10, 8.181453300774406e-12, 2.1201821949384996e-12, -4.157966370690044e-14, -8.344937881711169e-15, 1.5879358082667785e-16, 2.58619927010138e-17, -4.743519186210765e-19, -6.478222768805454e-20, 1.1415279905758999e-21, 1.339308120471104e-22, -2.2639457012857495e-24, -2.3246536867152943e-25, 3.768116220091341e-27, 3.4361950006830084e-28, -5.342247232985093e-30, -4.378059507841556e-31, 6.532369171961593e-33, 4.8581428358700575e-34))
            elseif imz < 14.5
                r = evalpoly(_z - 13.0, (0.20692610237706782, 0.07031805212177837, -0.10616759165475616, -0.00892808222232259, 0.008911624226865098, 0.00030633335736580117, -0.00029379837605402224, -4.243985960944521e-6, 5.111264894811495e-6, 2.334320542560391e-8, -5.478056180434088e-8, 4.428644411870052e-11, 3.982782274431613e-10, -1.5518869433024987e-12, -2.0962106963067093e-12, 1.1997655419738472e-14, 8.3663953608489e-15, -5.700114155181254e-17, -2.6216054600879916e-17, 1.9537138326805023e-19, 6.625117044604972e-20, -5.172603589788702e-22, -1.3794951394555679e-22, 1.100756912484797e-24, 2.408449879552199e-25, -1.93423576653241e-27, -3.5773306530412485e-28, 2.8629880731192636e-30, 4.576360714471254e-31, -3.62565930854413e-33, -5.095559260340188e-34))
            else
                r = evalpoly(_z - 16.0, (-0.1748990739836292, -0.09039717566130419, 0.09027444873123035, 0.01312662593374557, -0.007667362695010893, -0.0005550708142218287, 0.0002571415573791134, 1.0850297108500103e-5, -4.565690471161262e-6, -1.2026225777846727e-7, 4.9959717576483296e-8, 8.48815248845687e-10, -3.701703923085197e-10, -4.101051708969331e-12, 1.980421966657433e-12, 1.4173962555705988e-14, -8.014281597026939e-15, -3.5742021762369616e-17, 2.540522616136639e-17, 6.485026844702886e-20, -6.482772100803784e-20, -7.615209126543415e-23, 1.3608987282597849e-22, 2.2067196209386653e-26, -2.3923906484884365e-25, 1.4153681120888267e-28, 3.5743243599678463e-28, -4.139824487468731e-31, -4.595455176935322e-31, 7.2929256888043685e-34, 5.138919311361309e-34))
            end
            r = conj(r)
        end
    else
        if imz <= tan(π/5) * rez
            # angle(z) <= π / 5.0
            # use power series but evaluated using second order horner scheme
            zz = z*z
            z4 = zz*zz
            r = evalpoly(z4, (1.0, 0.015625, 6.781684027777777e-6, 4.709502797067901e-10, 9.385966990329842e-15, 7.242258480192779e-20, 2.5978027721077174e-25, 4.9016626390753635e-31, 5.318644356635594e-37, 3.5500798014623073e-43, 1.5365650110207356e-49, 4.499321282809354e-56, 9.228877211181495e-63, 1.3652185223641266e-69, 1.4929270885431173e-76, 1.232764473958843e-83, 7.829549665715613e-91, 3.887148093924665e-98))
            r += zz*evalpoly(z4, (0.25, 0.00043402777777777775, 6.781684027777778e-8, 2.4028075495244395e-12, 2.896903392077112e-17, 1.4963343967340453e-22, 3.842903509035085e-28, 5.4462918211948485e-34, 4.60090342269515e-40, 2.458504017633177e-46, 8.71068600351891e-53, 2.1263333094562166e-59, 3.691550884472598e-66, 4.681819349671216e-73, 4.4379521062518346e-80, 3.206983543077115e-87, 1.7974172786307652e-94, 7.932955293723807e-102))
        else
            # use rational approximation based on AAA algorithm
            zz = (4.45 + 3.2052728679295117im, 4.455420677104106 + 17.0im, 4.45 + 15.382484426852667im, 4.45 + 5.591125152996643im, 4.45 + 8.721074610794474im, 4.45 + 12.593761762936332im, 13.99545555483758 + 3.2im, 4.45 + 10.449153300661845im, 13.99448098692729 + 17.0im, 4.45 + 4.364295914717135im, 5.74410737590207 + 17.0im, 7.015857771516533 + 3.2im, 8.431271890243135 + 17.0im, 4.45 + 14.193575953103643im, 10.028155979725057 + 3.2im, 14.0 + 12.056471942775644im, 5.0784076766983475 + 3.2im)
            w = (-0.08402970424774468 + 0.0im, -0.025416959445504525 + 0.008064492555082617im, -0.12436237574219852 + 0.09773493772114668im, -0.10773790109082547 + 0.09931091918808033im, -0.04136192882523951 - 0.10845822138428568im, -0.12376621749712938 - 0.1560380316634484im, 0.1440093676081139 + 0.07630216514159634im, -0.11981155913520558 - 0.10502286007991138im, -0.09732103082699156 + 0.12394089091215946im, -0.2024491672690207 - 0.008074664453662488im, -0.032009598780432726 + 0.1399683216091626im, 0.1635093206102166 - 0.2625104018116356im, -0.0069195515849893256 + 0.23685845201445335im, -0.22780106527284 - 0.08591498944175457im, 0.31071127502007534 - 0.15037952246380232im, 0.5169717201599435 + 0.3199985816199036im, 0.057776769164925704 - 0.2256791564859297im)
            f = (0.40650720354357883 - 0.006334133391405162im, 0.3996077953677851 - 0.0028305511728273568im, 0.39967704085612976 - 0.0030016547219227385im, 0.4030249334706483 - 0.006007622393381157im, 0.40100073574830397 - 0.004755573523837742im, 0.40005639633772716 - 0.003557024462101866im, 0.4024594080667488 - 0.0008405204317908334im, 0.40054207447089496 - 0.004211100015366804im, 0.40036795014598564 - 0.0018064878102966003im, 0.4045859012569077 - 0.00644263857738687im, 0.3997607686595453 - 0.00268586039668089im, 0.4051483029145814 - 0.0031164474364945016im, 0.4000585212223849 - 0.0024151743246614353im, 0.39982888182793785 - 0.003319036953780421im, 0.40367802519983587 - 0.0016111929572418197im, 0.4009968019331028 - 0.0018465588128410255im, 0.4062550837386149 - 0.005276094572860034im)
            f = f.*w
            s1 = 0.0 + 0.0im
            s2 = 0.0 + 0.0im
            @fastmath for ind in eachindex(f)
                C = inv(z - zz[ind])
                s1 += C*f[ind]
                s2 += C*w[ind]
            end
            r = @fastmath s1 / (s2 * sqrt(z) * exp(-z))
        end
    end
    isconj && (r = conj(r))
    return r
end

function besseli0(z::ComplexF32)
    z = ComplexF64(z)
    isconj = false
    if real(z) < 0.0
        z = -z
    end
    if imag(z) < 0.0
        z = conj(z)
        isconj = true
    end
    if abs2(z) > 100.0
        zinv = 1 / z
        e = exp(z)
        sinv = sqrt(zinv) * SQ1O2PI(Float64)
        p = evalpoly(zinv*zinv, (1.0, 0.0703125, 0.112152099609375, 0.5725014209747314, 6.074042001273483, 110.01714026924674))
        p2 = zinv*evalpoly(zinv*zinv, (0.125, 0.0732421875, 0.22710800170898438, 1.7277275025844574, 24.380529699556064, 551.3358961220206))
        r = e * sinv * (p + p2) + im * sinv * (p - p2) / e
    else
        zz = z*z
        z4 = zz*zz
        r = evalpoly(z4, (1.0, 0.015625, 6.781684027777777e-6, 4.709502797067901e-10, 9.385966990329842e-15, 7.242258480192779e-20, 2.5978027721077174e-25, 4.9016626390753635e-31, 5.318644356635594e-37, 3.5500798014623073e-43))
        r += zz*evalpoly(z4, (0.25, 0.00043402777777777775, 6.781684027777778e-8, 2.4028075495244395e-12, 2.896903392077112e-17, 1.4963343967340453e-22, 3.842903509035085e-28, 5.4462918211948485e-34, 4.60090342269515e-40, 2.458504017633177e-46))
    end
    isconj && (r = conj(r))
    return ComplexF32(r)
end

function besseli1(z::ComplexF64)
    c = one(z)
    # shift phase to 0 < angle(z) < pi/2
    if real(z) < 0.0
        z = -z
        c = -c
    end
    if imag(z) < 0.0
        z = conj(z)
        isconj = true
    else
        isconj = false
    end
    rez, imz = reim(z)
    if abs2(z) > 306.25
        # use asymptotic expansion for large arguments |z| > 17.5
        zinv = 1 / z
        e = exp(z)
        sinv = sqrt(zinv) * SQ1O2PI(Float64)
        p = evalpoly(zinv*zinv, (1.0, -0.1171875, -0.144195556640625, -0.6765925884246826, -6.883914268109947, -121.59789187653587, -3302.2722944808525, -127641.2726461746, -6.656367718817688e6, -4.502786003050393e8, -3.8338575207427895e10, -4.0118385991331978e12, -5.060568503314726e14, -7.572616461117957e16, -1.3262572853205555e19))
        p2 = evalpoly(zinv*zinv, (0.375, 0.1025390625, 0.2775764465332031, 1.993531733751297, 27.248827311268542, 603.8440767050702, 19718.37591223663, 890297.8767070678, 5.310411010968523e7, 4.043620325107754e9, 3.827011346598606e11, 4.406481417852279e13, 6.065091351222699e15, 9.83388387659068e17))
        r = e * sinv * muladd(-p2, zinv, p) - im * muladd(p2, zinv, p) * sinv / e
    elseif rez < 4.5
        # taylor series around the roots (slight offset) of J1(z)
        # use relation I1(z) = J1(im*z) / im
        _z = complex(imz, rez)
        if imz < 2.2
            # power series for I0
            r = z*evalpoly(z*z, (0.5, 0.0625, 0.0026041666666666665, 5.425347222222222e-5, 6.781684027777778e-7, 5.651403356481481e-9, 3.363930569334215e-11, 1.5017547184527747e-13, 5.214426105738801e-16, 1.4484516960385557e-18, 3.2919356728148996e-21, 6.234726653058522e-24, 9.991549123491221e-27, 1.3724655389411017e-29, 1.6338875463584545e-32, 1.7019661941233902e-35, 1.564307163716351e-38))
        else
            if imz < 5.5
                r = evalpoly(_z - 4.0, (-0.06604332802354913, -0.3806389778579601, 0.07853768224328367, 0.04930901975080153, -0.006977093733246101, -0.0020301389249499455, 0.00022646889994150718, 4.155579989376072e-5, -3.859397238281367e-6, -5.110006887219984e-7, 4.077048904895698e-8, 4.19879116624702e-9, -2.939803646709019e-10, -2.4697315577955285e-11, 1.5414044832012797e-12, 1.0915304728084314e-13, -6.146812221248163e-15, -3.757757785158431e-16, 1.9280657169072558e-17, 1.0361898925247856e-18, -4.88306428816425e-20, -2.340070332638712e-21, 1.0196735825466999e-22, 4.407465239122521e-24, -1.7860193798634053e-25, -7.02892492572196e-27, 2.661920498379524e-28, 9.613567030783265e-30, -3.417184796719832e-31, -1.1400824284559156e-32, 3.818052689081327e-34))
            elseif imz < 8.5
                r = evalpoly(_z - 7.0, (-0.004682823482345833, 0.30074824530274785, -0.019188389693537092, -0.0471605174284124, 0.0029657448695426985, 0.002117094601424397, -0.00012058288259054909, -4.528596933543863e-5, 2.3218131613689033e-6, 5.707147746119497e-7, -2.6460803717919574e-8, -4.758581058225322e-9, 2.0083557753034083e-10, 2.8247297290735913e-11, -1.0924079041461721e-12, -1.2558495868983204e-13, 4.477550018840212e-15, 4.340526278677539e-16, -1.43459083340185e-17, -1.200084819927342e-18, 3.6949972617001396e-20, 2.715156932456774e-21, -7.821992893949606e-23, -5.12035844336118e-24, 1.3856216528273523e-25, 8.172858416431775e-27, -2.0848053095060175e-28, -1.1184641449276482e-29, 2.6979469530445177e-31, 1.326904110743154e-32, -3.035362398996416e-34))
            elseif imz < 11.5
                r = evalpoly(_z - 10.0, (0.04347274616886144, -0.25028303906823446, -0.009004857400174687, 0.04116523404576117, -0.00023758063287529515, -0.0019744712563326975, 3.38446715384414e-5, 4.430555504324474e-5, -9.214428549130458e-7, -5.769323465195413e-7, 1.2549544791376487e-8, 4.920635393174389e-9, -1.0635889291006728e-10, -2.9682550729139e-11, 6.236949556035066e-13, 1.335190061073787e-13, -2.6994908740535234e-15, -4.655158686182484e-16, 9.012686453800453e-18, 1.2956445537610907e-18, -2.3972087802093897e-20, -2.9464778650364287e-21, 5.2070751129572234e-23, 5.579168848116706e-24, -9.420290550228352e-26, -8.934107001775822e-27, 1.442406752905975e-28, 1.2258566621956357e-29, -1.8943870598688617e-31, -1.4574428507610173e-32, 2.158353205458848e-34))
            elseif imz < 14.5
                r = evalpoly(_z - 13.0, (-0.07031805212177837, 0.2123351833095123, 0.02678424666696777, -0.035646496907460384, -0.0015316667868290057, 0.0017627902563241335, 2.9707901726611656e-5, -4.0890119158491976e-5, -2.10088848830435e-7, 5.47805618043409e-7, -4.871508853057117e-10, -4.779338729317937e-9, 2.017453026293242e-11, 2.934694974829395e-11, -1.799648312960788e-13, -1.3386232577358232e-13, 9.690194063808074e-16, 4.718889828158393e-16, -3.7120562820930424e-18, -1.3250234089209876e-18, 1.086246753855577e-20, 3.034889306802292e-21, -2.531740898715398e-23, -5.7802797109249766e-24, 4.83558941632857e-26, 9.301059697909257e-27, -7.730067797438434e-29, -1.281381000050621e-29, 1.0514411994670651e-31, 1.5286677781106927e-32, -1.2332777691576932e-34))
            else
                r = evalpoly(_z - 16.0, (0.09039717566130419, -0.1805488974624607, -0.03937987780123671, 0.030669450780043572, 0.0027753540711091436, -0.0015428493442746806, -7.595207975950073e-5, 3.65255237692901e-5, 1.0823603200062054e-6, -4.995971757648329e-7, -9.336967737302558e-9, 4.442044707702236e-9, 5.331367221660131e-11, -2.772590753320406e-11, -2.126094383355898e-13, 1.2822850555243102e-13, 6.076143699602834e-16, -4.57294070904595e-16, -1.2321551004935482e-18, 1.2965544201607567e-18, 1.599193916574117e-21, -2.993977202171527e-21, -5.0754551281589305e-25, 5.7417375563722476e-24, -3.5384202802220665e-27, -9.2932433359164e-27, 1.1177526116165572e-29, 1.2867274495418905e-29, -2.114948449753267e-32, -1.5416757934083927e-32, 3.0470627981401083e-35))
            end
            r = conj(r) * im
        end 
    else
        if imz <= tan(π/5) * rez
            # angle(z) <= π / 5.0
            # power series for I1 with second order Horner scheme
            zz = z*z
            z4 = zz*zz
            r = evalpoly(z4, (0.5, 0.0026041666666666665, 6.781684027777778e-7, 3.363930569334215e-11, 5.214426105738801e-16, 3.2919356728148996e-21, 9.991549123491221e-27, 1.6338875463584545e-32, 1.564307163716351e-38, 9.342315267006072e-45, 3.658488121477942e-51, 9.781133223498595e-58, 1.845775442236299e-64, 2.5281824488224564e-71, 2.574012221626064e-78, 1.9883297967078112e-85, 1.1862954038963049e-92, 5.553068705606664e-100))
            r += zz*evalpoly(z4, (0.0625, 5.425347222222222e-5, 5.651403356481481e-9, 1.5017547184527747e-13, 1.4484516960385557e-18, 6.234726653058522e-24, 1.3724655389411017e-29, 1.7019661941233902e-35, 1.2780287285264307e-41, 6.146260044082942e-48, 1.9797013644361157e-54, 4.4298610613671176e-61, 7.099136316293458e-68, 8.360391695841456e-75, 7.396586843753058e-82, 5.010911786057992e-89, 2.6432607038687723e-96))
            r *= z    
        else
            # rational approximation using the AAA algorithm
            zz = (4.458154894695153 + 3.2im, 4.480772673369468 + 17.0im, 4.45 + 15.378794232077947im, 4.45 + 5.581079184404986im, 4.45 + 8.671611451329097im, 4.45 + 12.629949204612991im, 4.45 + 10.182155482531126im, 13.977243200035922 + 3.2im, 13.98901544675163 + 17.0im, 5.970053344910512 + 17.0im, 4.45 + 4.1927689631418055im, 6.629237081151842 + 3.2im, 4.45 + 14.033578808776824im, 9.509818960272973 + 17.0im, 10.131161524873981 + 3.2im, 4.45 + 16.51093571890095im, 14.0 + 8.519731558335298im)
            w = (0.011926786938722102 + 0.0im, 0.14525133653183686 + 0.1369374112869543im, -0.41873672964597924 - 0.268074354897585im, -0.019987497679825043 - 0.03974063361067378im, 0.07358389994286887 - 0.02916991816667183im, 0.17507220142869737 - 0.16370313407526862im, 0.03633350478070306 - 0.1177109043598989im, 0.06729582219784123 + 0.02441163867281959im, 0.02876576903699818 + 0.057608388543011514im, 0.13954109367802586 + 0.2979830326037936im, 0.01080106526525797 - 0.038043278614608475im, 0.06278114442055713 - 0.02128826340709847im, -0.056408120884862564 - 0.41889557661864824im, -0.007477086174182513 + 0.19537142411011837im, 0.12032774661776884 - 0.032150889528166825im, -0.4232516462448531 + 0.14804062532952536im, 0.05419012107716023 + 0.26842321386444234im)
            f = (0.3764410461504519 + 0.017574599817003107im, 0.3968835991093494 + 0.008341444680995272im, 0.39653865084627243 + 0.009022274197379459im, 0.3862559779998985 + 0.017302638395118457im, 0.3923211952955922 + 0.014037050481033892im, 0.3954382847083531 + 0.010635452221890032im, 0.39378056504862274 + 0.012599084513554796im, 0.38855550829520036 + 0.002437292977339643im, 0.39464665779628927 + 0.00534351158603366im, 0.3963079213294322 + 0.007922106437844278im, 0.38117491105807355 + 0.018186857840720456im, 0.3800905576242954 + 0.009633046119032825im, 0.3960466628986995 + 0.009855782989032044im, 0.3952629328354993 + 0.006807100956293367im, 0.38515397652762173 + 0.004511333412380529im, 0.39675455548646676 + 0.008524349289970855im, 0.3910654496700234 + 0.004910078696571082im)
            f = f.*w
            s1 = 0.0 + 0.0im
            s2 = 0.0 + 0.0im
            @fastmath for ind in eachindex(f)
                C = inv(z - zz[ind])
                s1 += C*f[ind]
                s2 += C*w[ind]
            end

        r = @fastmath s1 / (s2 * sqrt(z) * exp(-z))
        end
    end
    isconj && (r = conj(r))
    return r*c
end

function besseli1(z::ComplexF32)
    z = ComplexF64(z)
    c = one(z)
    # shift phase to 0 < angle(z) < pi/2
    if real(z) < 0.0
        z = -z
        c = -c
    end
    if imag(z) < 0.0
        z = conj(z)
        isconj = true
    else
        isconj = false
    end
    if abs2(z) > 100.0
        zinv = 1 / z
        e = exp(z)
        sinv = sqrt(zinv) * SQ1O2PI(Float64)
        p = evalpoly(zinv*zinv, (1.0, -0.1171875, -0.144195556640625, -0.6765925884246826, -6.883914268109947, -121.59789187653587, -3302.2722944808525))
        p2 = zinv*evalpoly(zinv*zinv, (0.375, 0.1025390625, 0.2775764465332031, 1.993531733751297, 27.248827311268542, 603.8440767050702, 19718.37591223663))
        r = e * sinv * (p - p2) - im * sinv * (p + p2) / e
    else
        zz = z*z
        z4 = zz*zz
        r = evalpoly(z4, (0.5, 0.0026041666666666665, 6.781684027777778e-7, 3.363930569334215e-11, 5.214426105738801e-16, 3.2919356728148996e-21, 9.991549123491221e-27, 1.6338875463584545e-32, 1.564307163716351e-38, 9.342315267006072e-45))
        r += zz*evalpoly(z4, (0.0625, 5.425347222222222e-5, 5.651403356481481e-9, 1.5017547184527747e-13, 1.4484516960385557e-18, 6.234726653058522e-24, 1.3724655389411017e-29, 1.7019661941233902e-35, 1.2780287285264307e-41, 6.146260044082942e-48))
        r *= z
    end
    isconj && (r = conj(r))
    return ComplexF32(r*c)
end

#              Modified Bessel functions of the first kind of order nu
#                           besseli(nu, x)
#
#    A numerical routine to compute the modified Bessel function of the first kind I_{ν}(x) [1]
#    for real orders and arguments of positive or negative value. The routine is based on several
#    publications [2-6] that calculate I_{ν}(x) for positive arguments and orders where
#    reflection identities are used to compute negative arguments and orders.
#
#    In particular, the reflectance identities for negative noninteger orders I_{−ν}(x) = I_{ν}(x) + 2 / πsin(πν)*Kν(x)
#    and for negative integer orders I_{−n}(x) = I_n(x) are used.
#    For negative arguments of integer order, In(−x) = (−1)^n In(x) is used and for
#    noninteger orders, Iν(−x) = exp(iπν) Iν(x) is used. For negative orders and arguments the previous identities are combined.
#
#    The identities are computed by calling the `besseli_positive_args(nu, x)` function which computes I_{ν}(x)
#    for positive arguments and orders. For large orders, Debye's uniform asymptotic expansions are used where large arguments (x>>nu)
#    a large argument expansion is used. The rest of the values are computed using the power series.

# [1] https://dlmf.nist.gov/10.40.E1
# [2] Amos, Donald E. "Computation of modified Bessel functions and their ratios." Mathematics of computation 28.125 (1974): 239-251.
# [3] Gatto, M. A., and J. B. Seery. "Numerical evaluation of the modified Bessel functions I and K." 
#     Computers & Mathematics with Applications 7.3 (1981): 203-209.
# [4] Temme, Nico M. "On the numerical evaluation of the modified Bessel function of the third kind." 
#     Journal of Computational Physics 19.3 (1975): 324-337.
# [5] Amos, DEv. "Algorithm 644: A portable package for Bessel functions of a complex argument and nonnegative order." 
#     ACM Transactions on Mathematical Software (TOMS) 12.3 (1986): 265-273.
# [6] Segura, Javier, P. Fernández de Córdoba, and Yu L. Ratis. "A code to evaluate modified bessel functions based on thecontinued fraction method." 
#     Computer physics communications 105.2-3 (1997): 263-272.
#

"""
    besseli(ν::Real, x::Real)
    besseli(ν::AbstractRange, x::Real)

Returns the modified Bessel function, ``I_ν(x)``, of the first kind and order `ν`.

```math
I_{\\nu}(x) = \\sum_{m=0}^{\\infty} \\frac{1}{m!\\Gamma(m+\\nu+1)}(\\frac{x}{2})^{2m+\\nu}
```

Routine supports single and double precision (e.g., `Float32` or `Float64`) real arguments.

For `ν` isa `AbstractRange`, returns a vector of type `float(x)` using recurrence to compute ``I_ν(x)`` at many orders
as long as the conditions `ν[1] >= 0` and `step(ν) == 1` are met. Consider the in-place version [`besseli!`](@ref Bessels.besseli!)
to avoid allocation.

# Examples

```
julia> besseli(2, 1.5)
0.3378346183356807

julia> besseli(3.2, 2.5)
0.3772632469352918

julia> besseli(1:3, 2.5)
3-element Vector{Float64}:
 2.5167162452886984
 1.2764661478191641
 0.47437040877803555
```

External links: [DLMF](https://dlmf.nist.gov/10.25.2), [Wikipedia](https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1)

See also: [`besseli!`](@ref Bessels.besseli!(out, ν, x)), [`besseli0(x)`](@ref Bessels.besseli0), [`besseli1(x)`](@ref Bessels.besseli1), [`besselix(nu,x)`](@ref Bessels.besselix))
"""
besseli(nu, x::Real) = _besseli(nu, float(x))

_besseli(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_besseli(Float32(nu), Float32(x)))

_besseli(nu::AbstractRange, x::T) where T = besseli!(zeros(T, length(nu)), nu, x)

function _besseli(nu::T, x::T) where T <: Union{Float32, Float64}
    isinteger(nu) && return _besseli(Int(nu), x)
    ~isfinite(x) && return x
    abs_nu = abs(nu)
    abs_x = abs(x)

    if nu >= 0
        if x >= 0
            return besseli_positive_args(abs_nu, abs_x)
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return cispi(abs_nu) * besseli_positive_args(abs_nu, abs_x)
        end
    else
        if x >= 0
            return besseli_positive_args(abs_nu, abs_x) + 2 / π * sinpi(abs_nu) * besselk_positive_args(abs_nu, abs_x)
        else
            #Iv = besseli_positive_args(abs_nu, abs_x)
            #Kv = besselk_positive_args(abs_nu, abs_x)
            #return cispi(abs_nu) * Iv + 2 / π * sinpi(abs_nu) * (cispi(-abs_nu) * Kv - im * π * Iv)
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
        end
    end
end
function _besseli(nu::Integer, x::T) where T <: Union{Float32, Float64}
    ~isfinite(x) && return x
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    if x >= 0
        return besseli_positive_args(abs_nu, abs_x)
    else
        return sg * besseli_positive_args(abs_nu, abs_x)
    end
end

"""
    Bessels.besseli!(out::AbstractVector{T}, ν::AbstractRange, x::T)

Computes the modified Bessel function, ``I_ν(x)``, of the first kind at many orders `ν` in-place using recurrence.
The conditions `ν[1] >= 0` and `step(ν) == 1` must be met.

# Examples

```
julia> nu = 1:3; x = 1.5; out = zeros(typeof(x), length(nu));

julia> Bessels.besseli!(out, nu, x)
3-element Vector{Float64}:
 0.9816664285779074
 0.3378346183356807
 0.0807741130160923
```

See also: [`besseli(ν, x)`](@ref Bessels.besseli(ν, x))
"""
besseli!(out::AbstractVector, nu::AbstractRange, x) = _besseli!(out, nu, float(x))

function _besseli!(out::AbstractVector{T}, nu::AbstractRange, x::T) where T
    (nu[1] >= 0 && step(nu) == 1) || throw(ArgumentError("nu must be >= 0 with step(nu)=1"))
    len = length(out)
    !isequal(len, length(nu)) && throw(ArgumentError("out and nu must have the same length"))

    k = len
    inu = zero(T)
    while abs(inu) < floatmin(T)
        if besseli_underflow_check(nu[k], x)
            inu = zero(T)
        else
            inu = _besseli(nu[k], x)
        end
        out[k] = inu
        k -= 1
        k < 1 && break
    end
    if k >= 1
        out[k] = _besseli(nu[k], x)
        tmp = @view out[begin:k+1]
        besselk_down_recurrence!(tmp, x, nu[begin:k+1])
        return out
    else
        return out
    end
end

besseli_underflow_check(nu, x::T) where T = nu > 140 + T(1.45)*x + 53*Base.Math._approx_cbrt(x)

"""
    besseli_positive_args(nu, x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the first kind of order nu, ``I_{nu}(x)`` for positive arguments.
"""
function besseli_positive_args(nu, x::T) where T <: Union{Float32, Float64}
    iszero(nu) && return besseli0(x)
    isone(nu) && return besseli1(x)

    # use large argument expansion if x >> nu
    besseli_large_argument_cutoff(nu, x) && return besseli_large_argument(nu, x)

    # use uniform debye expansion if x or nu is large
    besselik_debye_cutoff(nu, x) && return besseli_large_orders(nu, x)

    # for rest of values use the power series
    return besseli_power_series(nu, x)
end

"""
    besselix(nu, x::T) where T <: Union{Float32, Float64}

Scaled modified Bessel function of the first kind of order nu, ``I_{nu}(x)*e^{-x}``.
Nu must be real.
"""
besselix(nu::Real, x::Real) = _besselix(nu, float(x))

_besselix(nu::Union{Int16, Float16}, x::Union{Int16, Float16}) = Float16(_besselix(Float32(nu), Float32(x)))

function _besselix(nu, x::T) where T <: Union{Float32, Float64}
    iszero(nu) && return besseli0x(x)
    isone(nu) && return besseli1x(x)
    isinf(x) && return T(Inf)

    # use large argument expansion if x >> nu
    besseli_large_argument_cutoff(nu, x) && return besseli_large_argument_scaled(nu, x)

    # use uniform debye expansion if x or nu is large
    besselik_debye_cutoff(nu, x) && return besseli_large_orders_scaled(nu, x)

    # for rest of values use the power series
    return besseli_power_series(nu, x) * exp(-x)
end

#####
#####  Debye's uniform asymptotic for I_{nu}(x)
#####

# Implements the uniform asymptotic expansion https://dlmf.nist.gov/10.41
# In general this is valid when either x or nu is gets large
# see the file src/U_polynomials.jl for more details
"""
    besseli_large_orders(nu, x::T)

Debey's uniform asymptotic expansion for large order valid when v-> ∞ or x -> ∞
"""
function besseli_large_orders(v, x::T) where T
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQ1O2PI(S) * sqrt(inv(v)) * exp(v*n) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_In(p, v, p2, T))
end

function besseli_large_orders_scaled(v, x::T) where T
    S = promote_type(T, Float64)
    x = S(x)
    z = x / v
    zs = hypot(1, z)
    n = zs + log(z) - log1p(zs)
    coef = SQ1O2PI(S) * sqrt(inv(v)) * exp(v*n - x) / sqrt(zs)
    p = inv(zs)
    p2  = v^2/fma(max(v,x), max(v,x), min(v,x)^2)

    return T(coef*Uk_poly_In(p, v, p2, T))
end

#####
#####  Large argument expansion (x>>nu) for I_{nu}(x)
#####

# Implements the uniform asymptotic expansion https://dlmf.nist.gov/10.40.E1
# In general this is valid when x > nu^2
"""
    besseli_large_orders(nu, x::T)

Debey's uniform asymptotic expansion for large order valid when v-> ∞ or x -> ∞
"""
function besseli_large_argument(v, x::T) where T
    a = exp(x / 2)
    coef = a / sqrt(2 * (π * x))
    return T(_besseli_large_argument(v, x) * coef * a)
end

besseli_large_argument_scaled(v, x::T) where T =  T(_besseli_large_argument(v, x) / sqrt(2 * (π * x)))

function _besseli_large_argument(v, x::T) where T
    MaxIter = 5000
    S = promote_type(T, Float64)
    v, x = S(v), S(x)

    fv2 = 4 * v^2
    term = one(S)
    res = term
    s = -term
    for i in 1:MaxIter
        offset = muladd(2, i, -1)
        term *= muladd(offset, -offset, fv2) / (8 * x * i)
        res = muladd(term, s, res)
        s = -s
        abs(term) <= eps(T) && break
    end
    return res
end

besseli_large_argument_cutoff(nu, x::Float64) = x > 23.0 && x > nu^2 / 1.8 + 23.0
besseli_large_argument_cutoff(nu, x::Float32) = x > 18.0f0 && x > nu^2 / 19.5f0 + 18.0f0

#####
#####  Power series for I_{nu}(x)
#####

# Use power series form of I_v(x) which is generally accurate across all values though slower for larger x
# https://dlmf.nist.gov/10.25.E2
"""
    besseli_power_series(nu, x::T) where T <: Float64

Computes ``I_{nu}(x)`` using the power series for any value of nu.
"""
function besseli_power_series(v, x::ComplexOrReal{T}) where T
    MaxIter = 3000
    S = eltype(x)
    out = zero(S)
    xs = (x/2)^v
    a = xs / gamma(v + one(T))
    t2 = (x/2)^2
    for i in 0:MaxIter
        out += a
        abs(a) < eps(T) * abs(out) && break
        a *= inv((v + i + one(T)) * (i + one(T))) * t2
    end
    return out
end

#=
# the following is a deprecated version of the continued fraction approach
# using K0 and K1 as starting values then forward recurrence up till nu
# then using the wronskian to getting I_{nu}
# in general this method is slow and depends on starting values of K0 and K1
# which is not very flexible for arbitrary orders

function _besseli_continued_fractions(nu, x::T) where T
    S = promote_type(T, Float64)
    xx = S(x)
    knum1, knu = besselk_up_recurrence(xx, besselk1(xx), besselk0(xx), 1, nu-1)
    # if knu or knum1 is zero then besseli will likely overflow
    (iszero(knu) || iszero(knum1)) && return throw(DomainError(x, "Overflow error"))
    return 1 / (x * (knum1 + knu / steed(nu, x)))
end
function _besseli_continued_fractions_scaled(nu, x::T) where T
    S = promote_type(T, Float64)
    xx = S(x)
    knum1, knu = besselk_up_recurrence(xx, besselk1x(xx), besselk0x(xx), 1, nu-1)
    # if knu or knum1 is zero then besseli will likely overflow
    (iszero(knu) || iszero(knum1)) && return throw(DomainError(x, "Overflow error"))
    return 1 / (x * (knum1 + knu / steed(nu, x)))
end
function steed(n, x::T) where T
    MaxIter = 1000
    xinv = inv(x)
    xinv2 = 2 * xinv
    d = x / (n + n)
    a = d
    h = a
    b = muladd(2, n, 2) * xinv
    for _ in 1:MaxIter
        d = inv(b + d)
        a *= muladd(b, d, -1)
        h = h + a
        b = b + xinv2
        abs(a / h) <= eps(T) && break
    end
    return h
end
=#
