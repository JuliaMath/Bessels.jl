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
    besseli1(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the first kind of order one, ``I_1(x)``.
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
    if abs(z) > 17.5
        # use asymptotic expansion for large arguments
        zinv = 1 / z
        e = exp(z)
        sinv =sqrt(zinv)/sqrt(2*pi)
        p = evalpoly(zinv*zinv, (1.0, 0.0703125, 0.112152099609375, 0.5725014209747314, 6.074042001273483, 110.01714026924674, 3038.090510922384, 118838.42625678325, 6.252951493434797e6, 4.259392165047669e8, 3.646840080706556e10, 3.8335346613939443e12, 4.8540146868529006e14, 7.286857349377656e16))
        p2 = evalpoly(zinv*zinv, (0.125, 0.0732421875, 0.22710800170898438, 1.7277275025844574, 24.380529699556064, 551.3358961220206, 18257.755474293175, 832859.3040162893, 5.0069589531988926e7, 3.8362551802304335e9, 3.6490108188498334e11, 4.218971570284097e13, 5.827244631566907e15, 9.47628809926011e17))
        r = e * sinv * muladd(p2, zinv, p) + im * muladd(p2, -zinv, p) * sinv / e
    elseif real(z) < 4.5
        # use taylor series around the roots (slight offset) of J0(z)
        # use relation I0(z) = J0(im*z)
        _z = imag(z) + abs(real(z))*im
        if real(_z) < 2.2
            # use power series for I0
            r = evalpoly(z*z, (1.0, 0.25, 0.015625, 0.00043402777777777775, 6.781684027777777e-6, 6.781684027777778e-8, 4.709502797067901e-10, 2.4028075495244395e-12, 9.385966990329842e-15, 2.896903392077112e-17, 7.242258480192779e-20, 1.4963343967340453e-22, 2.5978027721077174e-25, 3.842903509035085e-28, 4.9016626390753635e-31, 5.4462918211948485e-34))
            r = conj(r) # the following taylor series compute J0 then use relation formula to I0
        elseif real(_z) < 5.5
            r = evalpoly(_z - 4.0, (-0.39714980986384735, 0.06604332802354913, 0.19031948892898004, -0.02617922741442789, -0.012327254937700382, 0.0013954187466492201, 0.0003383564874916576, -3.2352699991643884e-5, -5.19447498672009e-6, 4.288219153645963e-7, 5.110006887219985e-8, -3.706408095359726e-9, -3.498992638539183e-10, 2.261387420545399e-11, 1.764093969853949e-12, -1.0276029888008532e-13, -6.822065455052696e-15, 3.615771894851861e-16, 2.087643213976906e-17, -1.0147714299511874e-18, -5.180949462623928e-20, 2.325268708649643e-21, 1.0636683330175965e-22, -4.433363402376956e-24, -1.8364438496343838e-25, 7.144077519453622e-27, 2.703432663739215e-28, -9.858964808813052e-30, -3.433416796708309e-31, 1.1783395850758042e-32, 3.8002747615197184e-34))
        elseif real(_z) < 8.5
            r = evalpoly(_z - 7.0, (0.3000792705195556, 0.004682823482345833, -0.15037412265137393, 0.0063961298978456975, 0.0117901293571031, -0.0005931489739085397, -0.00035284910023739957, 1.7226126084364155e-5, 5.6607461669298285e-6, -2.5797924015210036e-7, -5.7071477461194966e-8, 2.405527610719961e-9, 3.9654842151877684e-10, -1.5448890579256987e-11, -2.0176640921954225e-12, 7.282719360974481e-14, 7.849059918114502e-15, -2.6338529522589484e-16, -2.4114034881541882e-17, 7.550478070536054e-19, 6.00042409963671e-20, -1.7595225055714948e-21, -1.2341622420258064e-22, 3.400866475630264e-24, 2.1334826847338247e-25, -5.542486611309409e-27, -3.14340708324299e-28, 7.721501146318583e-30, 3.994514803313029e-31, -9.303265355325922e-33, -4.42301370247718e-34))
        elseif real(_z) < 11.5
            r = evalpoly(_z - 10.0, (-0.24593576445134835, -0.04347274616886144, 0.12514151953411723, 0.003001619133391562, -0.010291308511440292, 4.751612657505903e-5, 0.0003290785427221163, -4.8349530769202e-6, -5.5381943804055925e-6, 1.0238253943478287e-7, 5.769323465195413e-8, -1.1408677083069533e-9, -4.100529494311991e-10, 8.181453300774406e-12, 2.1201821949384996e-12, -4.157966370690044e-14, -8.344937881711169e-15, 1.5879358082667785e-16, 2.58619927010138e-17, -4.743519186210765e-19, -6.478222768805454e-20, 1.1415279905758999e-21, 1.339308120471104e-22, -2.2639457012857495e-24, -2.3246536867152943e-25, 3.768116220091341e-27, 3.4361950006830084e-28, -5.342247232985093e-30, -4.378059507841556e-31, 6.532369171961593e-33, 4.8581428358700575e-34))
        elseif real(_z) < 14.5
            r = evalpoly(_z - 13.0, (0.20692610237706782, 0.07031805212177837, -0.10616759165475616, -0.00892808222232259, 0.008911624226865098, 0.00030633335736580117, -0.00029379837605402224, -4.243985960944521e-6, 5.111264894811495e-6, 2.334320542560391e-8, -5.478056180434088e-8, 4.428644411870052e-11, 3.982782274431613e-10, -1.5518869433024987e-12, -2.0962106963067093e-12, 1.1997655419738472e-14, 8.3663953608489e-15, -5.700114155181254e-17, -2.6216054600879916e-17, 1.9537138326805023e-19, 6.625117044604972e-20, -5.172603589788702e-22, -1.3794951394555679e-22, 1.100756912484797e-24, 2.408449879552199e-25, -1.93423576653241e-27, -3.5773306530412485e-28, 2.8629880731192636e-30, 4.576360714471254e-31, -3.62565930854413e-33, -5.095559260340188e-34, 3.978315384156111e-36))
        else
            r = evalpoly(_z - 16.0, (-0.1748990739836292, -0.09039717566130419, 0.09027444873123035, 0.01312662593374557, -0.007667362695010893, -0.0005550708142218287, 0.0002571415573791134, 1.0850297108500103e-5, -4.565690471161262e-6, -1.2026225777846727e-7, 4.9959717576483296e-8, 8.48815248845687e-10, -3.701703923085197e-10, -4.101051708969331e-12, 1.980421966657433e-12, 1.4173962555705988e-14, -8.014281597026939e-15, -3.5742021762369616e-17, 2.540522616136639e-17, 6.485026844702886e-20, -6.482772100803784e-20, -7.615209126543415e-23, 1.3608987282597849e-22, 2.2067196209386653e-26, -2.3923906484884365e-25, 1.4153681120888267e-28, 3.5743243599678463e-28, -4.139824487468731e-31, -4.595455176935322e-31, 7.2929256888043685e-34, 5.138919311361309e-34))
        end
        r = conj(r) 
    elseif abs(z) < 5.5
        # use power series for I0
        r = evalpoly(z*z, (1.0, 0.25, 0.015625, 0.00043402777777777775, 6.781684027777777e-6, 6.781684027777778e-8, 4.709502797067901e-10, 2.4028075495244395e-12, 9.385966990329842e-15, 2.896903392077112e-17, 7.242258480192779e-20, 1.4963343967340453e-22, 2.5978027721077174e-25, 3.842903509035085e-28, 4.9016626390753635e-31, 5.4462918211948485e-34, 5.318644356635594e-37))
    else
        if angle(z) <= π / 4.4
            # use power series but evaluated using second order horner scheme
            zz = z*z
            z4 = zz*zz
            r = evalpoly(z4, (1.0, 0.015625, 6.781684027777777e-6, 4.709502797067901e-10, 9.385966990329842e-15, 7.242258480192779e-20, 2.5978027721077174e-25, 4.9016626390753635e-31, 5.318644356635594e-37, 3.5500798014623073e-43, 1.5365650110207356e-49, 4.499321282809354e-56, 9.228877211181495e-63, 1.3652185223641266e-69, 1.4929270885431173e-76, 1.232764473958843e-83, 7.829549665715613e-91, 3.887148093924665e-98))
            r += zz*evalpoly(z4, (0.25, 0.00043402777777777775, 6.781684027777778e-8, 2.4028075495244395e-12, 2.896903392077112e-17, 1.4963343967340453e-22, 3.842903509035085e-28, 5.4462918211948485e-34, 4.60090342269515e-40, 2.458504017633177e-46, 8.71068600351891e-53, 2.1263333094562166e-59, 3.691550884472598e-66, 4.681819349671216e-73, 4.4379521062518346e-80, 3.206983543077115e-87, 1.7974172786307652e-94, 7.932955293723807e-102))
        
        else
            # use rational approximation based on AAA algorithm
            zz = (-1.3144540816651462 + 20.054693865816724im, -1.2152720220184496 + 18.54146805523844im, -1.0758582968968085 + 16.41442564500392im, -0.9186088936209557 + 14.015263371275323im, -0.7533475204960969 + 11.49386205943562im, -0.5838248556039682 + 8.907445998843897im, -0.41092161465730614 + 6.269452314652046im, -0.32055227293065935 + 4.890682596894067im, 16.906716750808087 + 10.865287107782516im, 0.26715371754140166 + 20.098224520867603im, 4.121994532242284 + 2.6493699394695174im, 4.053028754723365 + 19.68712670537236im, -0.5018982035099 + 7.657486833198153im, 9.416081350634247 + 17.758023876497013im, 1.9674895590143007 + 4.487648029332259im, -1.275759132467892 + 19.46432302583941im, 7.709313889965882 + 4.954475197822861im, 13.72022229845232 + 14.68895844098729im, -0.3541218766747901 + 5.402855776372861im, -0.9973883749616251 + 15.217205990064667im, 5.067285183070908 + 3.256546447342955im, 2.1309472708925585 + 19.986722185708082im, -0.8162780120038937 + 12.453995821138065im, -0.6642234573493526 + 10.134091621337502im)
            w  = (0.023435912479136792 + 0.0im, -0.07518141047674767 + 0.0475948197731169im, 0.02099862365431344 + 0.10227799317706024im, -0.044790606134137025 - 0.0019900581421525287im, 0.05092855107420865 - 0.030284027691088355im, 0.11876822116350058 + 0.3545737241031229im, -0.2800428715703842 + 0.1385502878368521im, 0.07056156978707125 - 0.015640049217372977im, -0.06946091028316168 + 0.016066454632345423im, 0.08524646892263366 + 0.0953622756709394im, -0.02790219158045262 + 0.07064926802304639im, 0.1422314233425249 - 0.2536760604980481im, -0.20586363442913913 + 0.3014383126557577im, 0.20547145538054265 - 0.17388658146749558im, -0.13839690210608482 - 0.07997812132261209im, -0.005116954351018287 + 0.07859682420061695im, 0.0409490985507271 - 0.2642881024594653im, -0.04174610452138103 - 0.23810783641445965im, -0.14078210159561536 - 0.17647153499232213im, -0.07672294420555978 + 0.08789101702567167im, -0.15544491331665283 - 0.17872134246439533im, 0.2746212867658749 + 0.0633913060138901im, 0.003561860807958012 + 0.005781000943183686im, 0.22467614820711904 + 0.050873741433048215im)
            f = (4.11757504005866 - 4.095512734321109im, -2.247112620034342 + 3.680759784557969im, 3.7834318298173315 + 0.562243565983836im, 1.025044899237516 - 2.4295763517152302im, -1.1024728966418114 - 0.9985456859791928im, -0.7134052191604513 + 0.634168698676205im, 0.35489432819051814 + 0.8984965319188546im, 0.15105140195420463 - 0.7254114647779558im, 0.401057852147248 - 0.0014084695668244449im, 0.5404571564588408 - 0.1885021348797114im, 0.407770977742567 - 0.00680290179701475im, 0.3994965367795384 - 0.0024661141779376626im, 0.8312271064319111 - 1.0050503936637845im, 0.40006191058465196 - 0.002248719754950748im, 0.4055260568863696 - 0.01689191195791546im, 5.210832430161148 + 1.7413278357835171im, 0.40364747476765517 - 0.003288380266379im, 0.40062798907705705 - 0.0018845654535970921im, -0.39383201186515915 - 0.17987628384333437im, -2.053184226968296 + 1.6069066509951657im, 0.40617427977861376 - 0.005320233415147144im, 0.4034434765080089 - 0.006086635360116373im, -0.07645988937623527 + 1.981923553555365im, 1.8844134122620073 + 0.24179792442042025im)
            f = f.*w
            s1 = 0.0 + 0.0im
            s2 = 0.0 + 0.0im
            @fastmath for ind in eachindex(f)
                C = inv(z - zz[ind])
                s1 += C*f[ind]
                s2 += C*w[ind]
            end
            r = s1 / (s2 * sqrt(z) * exp(-z))
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

    if abs(z) > 10.0
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

    if abs(z) > 17.5
        zinv = 1 / z
        e = exp(z)
        sinv = sqrt(zinv) * SQ1O2PI(Float64)
        p = evalpoly(zinv*zinv, (1.0, -0.1171875, -0.144195556640625, -0.6765925884246826, -6.883914268109947, -121.59789187653587, -3302.2722944808525, -127641.2726461746, -6.656367718817688e6, -4.502786003050393e8, -3.8338575207427895e10, -4.0118385991331978e12, -5.060568503314726e14, -7.572616461117957e16, -1.3262572853205555e19))
        p2 = evalpoly(zinv*zinv, (0.375, 0.1025390625, 0.2775764465332031, 1.993531733751297, 27.248827311268542, 603.8440767050702, 19718.37591223663, 890297.8767070678, 5.310411010968523e7, 4.043620325107754e9, 3.827011346598606e11, 4.406481417852279e13, 6.065091351222699e15, 9.83388387659068e17))
        r = e * sinv * muladd(-p2, zinv, p) - im * muladd(p2, zinv, p) * sinv / e
    elseif abs(z) < 5.2
        r = z*evalpoly(z*z, (0.5, 0.0625, 0.0026041666666666665, 5.425347222222222e-5, 6.781684027777778e-7, 5.651403356481481e-9, 3.363930569334215e-11, 1.5017547184527747e-13, 5.214426105738801e-16, 1.4484516960385557e-18, 3.2919356728148996e-21, 6.234726653058522e-24, 9.991549123491221e-27, 1.3724655389411017e-29, 1.6338875463584545e-32, 1.7019661941233902e-35, 1.564307163716351e-38))
    else
        if angle(z) <= π / 4.4
            zz = z*z
            z4 = zz*zz
            r = evalpoly(z4, (0.5, 0.0026041666666666665, 6.781684027777778e-7, 3.363930569334215e-11, 5.214426105738801e-16, 3.2919356728148996e-21, 9.991549123491221e-27, 1.6338875463584545e-32, 1.564307163716351e-38, 9.342315267006072e-45, 3.658488121477942e-51, 9.781133223498595e-58, 1.845775442236299e-64, 2.5281824488224564e-71, 2.574012221626064e-78, 1.9883297967078112e-85, 1.1862954038963049e-92, 5.553068705606664e-100))
            r += zz*evalpoly(z4, (0.0625, 5.425347222222222e-5, 5.651403356481481e-9, 1.5017547184527747e-13, 1.4484516960385557e-18, 6.234726653058522e-24, 1.3724655389411017e-29, 1.7019661941233902e-35, 1.2780287285264307e-41, 6.146260044082942e-48, 1.9797013644361157e-54, 4.4298610613671176e-61, 7.099136316293458e-68, 8.360391695841456e-75, 7.396586843753058e-82, 5.010911786057992e-89, 2.6432607038687723e-96))
            r *= z    
        else
            zz = (-1.3144285719254842 + 20.054304662400146im, -1.2128746095534062 + 18.50489060934118im,
                  -1.0886579984944276 + 16.60971135387317im, -0.9427507983505322 + 14.383597659587654im, 
                  -0.7876981565498402 + 12.017951489232377im, -0.6255235989509672 + 9.543645881424778im, 
                  -0.4528832294653219 + 6.9096628407010305im, -0.321046242530308 + 4.898219116608337im, 
                  16.905702128729647 + 10.872315095446137im, 4.119703326027316 + 2.6529313043347877im, 
                  -1.2665282079632723 + 19.323486333541947im, -0.3857883819775798 + 5.8859932845642575im, 
                  5.040563157078601 + 19.45771628582095im, 8.609436758553485 + 5.532949040120932im, 
                  0.694001756513151 + 20.08801537140881im, -0.5457083486368534 + 8.325900481870493im, 
                  2.790091405909159 + 4.028075216114001im, 12.479613762034912 + 15.75656181882421im, 
                  -1.0195312436724646 + 15.555040882512422im, 13.06890672928664 + 8.39887636916513im, 
                  5.1276393456482685 + 3.295333712441098im, 2.7253844460913528 + 19.914373693917753im, 
                  14.171480465360736 + 9.107457483790645im, 0.8038847904678548 + 4.833608304740307im
                 )
            w  = (0.013404709995345458 + 0.0im, -0.10348199668238728 + 0.03550804014042965im, 
                  -0.11679300367073918 + 0.16207579437860672im, -0.03568356238246015 - 0.16027459313035955im, 
                  -0.060434459186566966 - 0.02767438331782018im, -0.07086407749919574 + 0.06788442528741509im,
                  -0.08650269086297906 - 0.029416238653716714im, -0.00415412382674645 - 0.00607269307464697im, 
                  -0.07587326685881253 + 0.19888070181762688im, -0.08774917184927954 - 0.014101104643344704im, 
                  0.007193185558007104 + 0.06995793660348812im, -0.02672800181750328 - 0.03807797543563465im, 
                  0.19776072630033434 + 0.05342203578372187im, 0.3489984378687633 - 0.29126891724262455im, 
                  -0.010279561959668174 + 0.09872927973051646im, -0.12150771925043644 - 0.012812175864887374im,
                  -0.1299759980313551 - 0.07418447933244608im, -0.05894210892668888 + 0.24161474004043546im,
                  -0.2648591701697078 - 0.07362611929106554im, -0.000770797654486471 + 0.00014803398133387874im,
                  0.12194676645309753 - 0.25480936217584116im, 0.0718337239451382 + 0.20114630545134515im,
                  0.5172999110471845 - 0.09099393187858433im, -0.023839427516594167 - 0.056052737377955054im
                  )
            f = (-3.215770403317309 + 4.184450606321457im, 3.196349209039463 - 3.528571814265253im,
                 -3.0032703696390066 + 0.8968536198145782im, 1.7024961674754864 + 2.2913462636360844im, 
                 2.0860054490489937 - 0.919904256366073im, 0.019539554711925467 - 1.3245943976605632im, 
                 -0.5518276420937166 - 0.23608328648174784im, 0.7317073656843707 + 0.7151746200203796im, 
                 0.3926351501115586 + 0.00413652715802052im, 0.372728884764365 + 0.01864352217896087im, 
                 -3.733141424945208 - 2.8413601177219308im, 0.9786880011963323 - 0.618277150144583im, 
                 0.3971643533953302 + 0.007251821636383407im, 0.38646933950897416 + 0.008351088369392122im,
                 0.33879033458506147 + 0.08702715214401817im, 1.3930956061616984 + 0.6707847701397638im, 
                 0.38110598793307165 + 0.02730656109552859im, 0.39435250888241274 + 0.005949183211885836im, 
                 1.251416864948188 - 2.932439718874822im, 0.39076282374400617 + 0.005394185668824364im, 
                 0.3778378309823485 + 0.014609112728729376im, 0.39660674100748977 + 0.008337843914085801im, 
                 0.39140459884033496 + 0.004960218953458415im, 0.4210441290407154 + 0.10788322118417641im
                ).*w

            s1 = 0.0 + 0.0im
            s2 = 0.0 + 0.0im
            @fastmath for ind in eachindex(f)
                C = inv(z - zz[ind])
                s1 += C*f[ind]
                s2 += C*w[ind]
            end

        r = s1 / (s2 * sqrt(z) * exp(-z))
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

    if abs(z) > 10.0
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
    besseli(x::T) where T <: Union{Float32, Float64}

Modified Bessel function of the second kind of order nu, ``I_{nu}(x)``.
"""
# perhaps have two besseli(nu::Real, x::Real) and besseli(nu::AbstractRange, x::Real)
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

besseli!(out::DenseVector, nu::AbstractRange, x) = _besseli!(out, nu, float(x))

function _besseli!(out::DenseVector{T}, nu::AbstractRange, x::T) where T
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
    if k > 1
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
# which is not very flexible for arbitary orders

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
