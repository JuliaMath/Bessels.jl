#                           Airy functions
#
#                       airyai(z), airybi(nu, z)
#                   airyaiprime(z), airybiprime(nu, z)
#
#    A numerical routine to compute the airy functions and their derivatives in the entire complex plane.
#    These routines are based on the methods reported in [1] which use a combination of the power series
#    for small arguments and a large argument expansion for (x > ~10). The primary difference between [1]
#    and what is used here is that the regions where the power series and large argument expansions
#    do not provide good results they are filled by relation to other special functions (besselk and besseli)
#    using https://dlmf.nist.gov/9.6 (NIST 9.6.E1 - 9.6.E9). In this case the power series of besseli is used and then besselk 
#    is calculated using the continued fraction approach. This method is described in more detail in src/besselk.jl.
#    However, care must be taken when computing besseli because when the imaginary component is much larger than the real part
#    cancellation will occur. This can be overcome by shifting the order of besseli to be much larger and then using the power series
#    and downward recurrence to get besseli(1/3, x). Another difficult region is when -10<x<-5 and the imaginary part is close to zero.
#    In this region we use rotation (see connection formulas http://dlmf.nist.gov/9.2.v) to shift to different region of complex plane
#    where algorithms show good convergence. If imag(z) == zero then we use the reflection identities to compute in terms of bessel functions.
#    In general, the cutoff regions compared to [1] are different to provide full double precision accuracy and to prioritize using the power series
#    and asymptotic expansion compared to other approaches.
#
# [1] Jentschura, Ulrich David, and E. Lötstedt. "Numerical calculation of Bessel, Hankel and Airy functions." 
#     Computer Physics Communications 183.3 (2012): 506-519.

"""
    airyai(z)

Returns the Airy function of the first kind, ``\\operatorname{Ai}(z)``, which is the solution to the Airy differential equation ``f''(z) - z f(z) = 0``.

```math
\\operatorname{Ai}(z) = \\frac{\\sqrt{3}}{2 \\pi} \\int_{0}^^{\\infty} \\exp{-\\frac{t^3}{3} - \\frac{z^3}{3t^3}} dt
```

Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airyai(1.2)
0.10612576226331255

julia> airyai(1.2 + 1.4im)
-0.03254458873613304 - 0.14708163733976673im
```

External links: [DLMF](https://dlmf.nist.gov/9.2.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airyaiprime`](@ref), [`airybi`](@ref)
"""
airyai(z::Number) = _airyai(float(z))

_airyai(x::Float16) = Float16(_airyai(Float32(x)))
_airyai(z::ComplexF16) = ComplexF16(_airyai(ComplexF32(z)))

#######
####### Real arguments
#######

function airyai(x::Float64)
    if x >= 2.06
        return exp(-2 * (sqrt(x)^3 / 3)) * airyaix_large_pos_arg(x)[1]
    else
        if x <= -9.5
            return airyai_large_neg_arg(x)[1]
        else
            if x >= 0.0
                # taylor expansion at x = 1.5 
                return evalpoly(x - 1.5, (0.07174949700810541, -0.09738201284230132, 0.05381212275607906, -0.012387253709224428, -0.0013886523923485612, 0.0017615621096121207, -0.00048234107659157565, 2.9849780287371904e-5, 1.853661597722781e-5, -6.0773111966738585e-6, 6.406078250357068e-7, 8.5642265292882e-8, -3.876060196303256e-8, 4.929943737019422e-9, 1.5110638652930308e-10, -1.493604112262068e-10, 2.148584715338907e-11, -2.6814055261032026e-13, -3.827831388762196e-13, 6.164805942828536e-14, -2.2166191076964464e-15, -6.912167850804561e-16, 1.2624054278515299e-16, -6.42973178916429e-18, -9.091593675774033e-19, 1.9432657516901093e-19, -1.198995513927753e-20, -8.798710894927164e-22, 2.33256140820231e-22, -1.6391332233394835e-23, -6.091803198418045e-25))
            elseif x > -1.0
                return evalpoly(x + 0.5, (0.4757280916105396, -0.20408167033954738, -0.1189320229026349, 0.09629482113005221, -0.012051304907352494, -0.00835397167338305, 0.0034106824527909488, -0.00018748378739668977, -0.00017963058749604507, 4.867256036790685e-5, -1.0852054849851912e-6, -1.85424425163635e-6, 3.728421447757534e-7, -1.0133548664552323e-9, -1.1212446835297949e-8, 1.777851534328481e-9, 1.9136952296640593e-11, -4.449034045022864e-11, 5.778702804510329e-12, 1.2100035825074537e-13, -1.2468339961179948e-13, 1.3614768155678468e-14, 3.9684428150788985e-16, -2.598632088728038e-16, 2.430497466471834e-17, 8.779598099071528e-19, -4.184856864694815e-19))
            elseif x > -3.2
                return evalpoly(x + 2.338107410459767, (2.743319340666283e-17, 0.7012108227206914, -3.207087639834719e-17, -0.2732510368163064, 0.058434235226724286, 0.03194451370480103, -0.01366255184081532, -0.0003870349759524901, 0.0011408754894571796, -0.00017718920132546787, -3.3939160136269284e-5, 1.4137844310270033e-5, -7.41180299288454e-7, -4.2945486337203914e-7, 8.72022168160613e-8, 1.252053805808083e-9, -2.6389292196591304e-9, 3.0983375196473186e-10, 2.4255404477032373e-11, -9.8343678688258e-12, 6.661105552981143e-13, 1.1249812587695572e-13, -2.4657588515917296e-14, 7.965965484631788e-16, 3.0824314548929273e-16, -4.420019468170955e-17, 1.167553319557466e-19, 5.863076185446756e-19, -5.882695924413488e-20))
            elseif x > -4.75
                return evalpoly(x + 4.08794944413097, (-2.720348378642871e-16, -0.803111369654864, 5.560323321157856e-16, 0.5471797795259772, -0.06692594747123885, -0.11184216377764623, 0.027358988976298886, 0.009292361042237976, -0.003994362992058797, -0.00014760712751364074, 0.00028467905572535616, -3.082684106536036e-5, -9.934550872135151e-6, 2.6326770738641654e-6, 5.3764289286128445e-8, -9.855619834673579e-8, 1.0053714072345167e-8, 1.6788861968137084e-9, -4.5638978170010107e-10, 9.328982974624003e-12, 9.327854082162345e-12, -1.1774433153941012e-12, -6.234375094261117e-14, 2.7947001637990885e-14, -1.6713500242449534e-15, -2.9431613458960554e-16))
            elseif x > -6.1
                return evalpoly(x + 5.520559828095551, (2.313678943005095e-16, 0.8652040258941519, -6.386401513932251e-16, -0.7960684314096329, 0.07210033549117963, 0.21973717014275287, -0.0398034215704817, -0.027165996636626166, 0.007847756076526893, 0.0015301123354424383, -0.0007832222619265922, -5.448369227182677e-6, 4.434801281139783e-5, -4.82784752334856e-6, -1.3751331165365516e-6, 3.3809730430936423e-7, 1.1515237992029015e-8, -1.1917718796669944e-8, 8.97146222351664e-10, 2.2604595796334607e-10, -4.439596892555847e-11, -8.351286011527614e-13, 1.0197761050717805e-12, -7.862765122280811e-14, -1.1711709421130035e-14, 2.423074596316538e-15))
            elseif x > -7.2
                return evalpoly(x + 6.786708090071759, (-8.710477837103708e-17, -0.9108507370496018, 2.9557735202731247e-16, 1.0302796776637262, -0.07590422808746698, -0.3496103711718467, 0.05151398388318634, 0.05468569776946418, -0.012486084684708815, -0.004439192827500768, 0.0015491678856944288, 0.00016037655628253796, -0.00011327987159259477, 2.9534552161146124e-6, 5.10535152341918e-6, -6.348767142926879e-7, -1.3206281362722718e-7, 3.461056785480714e-8, 8.542321939505024e-10, -1.0729667675129367e-9, 7.582406135085136e-11, 1.9371772465258795e-11, -3.436282550010513e-12, -1.0997332719001945e-13, 7.734206349128181e-14, -4.483209467796499e-15))
            elseif x > -8.5
                return evalpoly(x + 7.944133587120853, (-3.222967925030853e-17, 0.9473357094415678, 1.28018438717254e-16, -1.25429357127562, 0.0789446424534639, 0.49821378438402086, -0.06271467856378098, -0.09235552894376721, 0.017793349442286454, 0.00931902751214878, -0.0025967585986179237, -0.0005112568183297996, 0.00022687897509904778, 9.389319637951164e-6, -1.2712163212229125e-5, 7.25185550490412e-7, 4.5990184323826115e-7, -6.791593419402475e-8, -9.56972591227924e-9, 2.922324845525447e-9, 2.1334860119088188e-11, -7.805967826213263e-11, 5.9585235209456126e-12, 1.2676904585002835e-12, -2.2716482806979397e-13, -6.8536313808318734e-15))
            else
                return evalpoly(x + 9.02265085334098, (2.1834671977219237e-16, -0.9779228085694986, -9.850331087383878e-16, 1.4705760105401993, -0.0814935673807908, -0.6634246948201652, 0.07352880052700973, 0.14057990051109173, -0.02369373910072015, -0.016595479983083087, 0.00393733595363381, 0.0011458316593658873, -0.0003948536938259645, -4.103271183031362e-5, 2.587065207093167e-5, -1.1728505435852704e-7, -1.1435607200608036e-6, 9.900321384824919e-8, 3.3335503439036955e-8, -5.955649567170201e-9, -5.309773544803389e-10, 2.0731250021063097e-10, -2.5212690187520218e-12, -4.7460190937036376e-12, 4.1677722875756535e-13, 6.716734034504283e-14))
            end
        end
    end
end

function airyaix(x::Float64)
    if x >= 2.06
        return airyaix_large_pos_arg(x)[1]
    elseif x >= 0.0
        # taylor expansion at x = 1.5 
        b = evalpoly(x - 1.5, (0.07174949700810541, -0.09738201284230132, 0.05381212275607906, -0.012387253709224428, -0.0013886523923485612, 0.0017615621096121207, -0.00048234107659157565, 2.9849780287371904e-5, 1.853661597722781e-5, -6.0773111966738585e-6, 6.406078250357068e-7, 8.5642265292882e-8, -3.876060196303256e-8, 4.929943737019422e-9, 1.5110638652930308e-10, -1.493604112262068e-10, 2.148584715338907e-11, -2.6814055261032026e-13, -3.827831388762196e-13, 6.164805942828536e-14, -2.2166191076964464e-15, -6.912167850804561e-16, 1.2624054278515299e-16, -6.42973178916429e-18, -9.091593675774033e-19, 1.9432657516901093e-19, -1.198995513927753e-20, -8.798710894927164e-22, 2.33256140820231e-22, -1.6391332233394835e-23, -6.091803198418045e-25))
        return exp(2 * (sqrt(x)^3 / 3)) * b
    else
        isnan(x) && return x
        # negative numbers return complex arguments
        throw(DomainError(x, "Complex result returned for real arguments. Use complex argument: airyaix(complex(x))"))
    end
end

@inline function airyaix_large_pos_arg(x::T) where T <: Float64
    if x > 3e10
        x3 = x * x * x
        p = evalpoly(1 / x3, (1.5707963267948966, -0.1636246173744684, 0.13124057851910487))
        p1 = evalpoly(1 / x3, (-1.5707963267948966, -0.22907446432425577, 0.15510250188621483))
        xsqr = sqrt(sqrt(x))
        ai = p / (PIPOW3O2(T) * xsqr)
        aip = p1 * xsqr / PIPOW3O2(T) 
        return ai, aip
    else
        inv_sqpi = 5.64189583547756286948e-1 # 1/sqrt(pi)
        xsqr = sqrt(x)
        zinv = 3 / (2 * x * xsqr)
        xsqr = sqrt(xsqr)
        p = evalpoly(zinv, (9.99999999999999995305e-1, 1.40264691163389668864e1, 7.05360906840444183113e1, 1.59756391350164413639e2, 1.68089224934630576269e2, 7.62796053615234516538e1, 1.20075952739645805542e1, 3.46538101525629032477e-1)) 
        q = evalpoly(zinv, (1.00000000000000000470e0, 1.40959135607834029598e1, 7.14778400825575695274e1, 1.64234692871529701831e2, 1.77318088145400459522e2, 8.45138970141474626562e1, 1.47562562584847203173e1, 5.67594532638770212846e-1))
        p1 = evalpoly(zinv, (1.00000000000000000550e0, 1.39470856980481566958e1, 6.99778599330103016170e1, 1.59317847137141783523e2, 1.71184781360976385540e2, 8.20584123476060982430e1, 1.47454670787755323881e1, 6.13759184814035759225e-1))
        q1 = evalpoly(zinv, (9.99999999999999994502e-1, 1.38498634758259442477e1, 6.86752304592780337944e1, 1.53206427475809220834e2, 1.58778084372838313640e2, 7.11727352147859965283e1, 1.11810297306158156705e1, 3.34203677749736953049e-1))
        ai = inv_sqpi * p / (q * 2 * xsqr)
        aip = -0.5 * inv_sqpi * xsqr * p1 / q1
        return ai, aip
    end
end
    
@inline function airyai_large_neg_arg(x::T) where T <: Float64
    x3 = x * x * x
    p = evalpoly(1 / x3, (1.5707963267948966, 0.13124057851910487, 0.4584353787485384, 5.217255928936184, 123.97197893818594, 5038.313653002081, 312467.7049060495, 2.746439545069411e7, 3.2482560591146026e9, 4.97462635569055e11, 9.57732308323407e13))
    q = evalpoly(1 / x3, (0.1636246173744684, 0.20141783231057064, 1.3848568733028765, 23.555289417250567, 745.2667344964557, 37835.063701047824, 2.8147130917899106e6, 2.8856687720069575e8, 3.8998976239149216e10, 6.718472897263214e12, 1.4370735281142392e15))
    p1 = evalpoly(1 / x3, (-1.5707963267948966, 0.15510250188621483, 0.4982993247266722, 5.515384839161109, 129.24738229725767, 5209.103946324185, 321269.61208650155, 2.812618811215662e7, 3.3166403972012258e9, 5.0676100258903735e11, 9.738286496397669e13))
    q1 = evalpoly(1 / x3, (0.22907446432425577, 0.22511404787652015, 1.4803642438754887, 24.70432792540913, 773.390007496322, 38999.21950723391, 2.8878225227454924e6, 2.950515261265541e8, 3.97712331943799e10, 6.837383921993536e12, 1.460066704564067e15))

    sq_x3 = im*sqrt(abs(x3))
    c1 = exp(2 * sq_x3 / 3) # loses some precision here
    a = muladd(1 / sq_x3, -q, p) / c1
    b = muladd(1 / sq_x3, q, p) * c1
    c = muladd(1 / sq_x3, -q1, p1) / c1
    d = muladd(1 / sq_x3, q1, p1) * c1

    xsqrt = sqrt(sqrt(complex(x)))
    ai = real((b + im*a) / (PIPOW3O2(T) * xsqrt))
    aip = imag((c + im*d) * xsqrt  / PIPOW3O2(T))
    return ai, aip
end

#######
####### Complex arguments
#######

function _airyai(z::Complex{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2*T(π)/3
            return exp(-z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airyai_large_argument(z)
    airyai_power_series_cutoff(x, y) && return airyai_power_series(z)

    if x > zero(T)
        # use relation to besselk (http://dlmf.nist.gov/9.6.E1)
        zz = 2 * z * sqrt(z) / 3
        return sqrt(z / 3) * besselk_continued_fraction_shift(one(T)/3, zz) / T(π)
    else
        # z is close to the negative real axis
        # for imag(z) == 0 use reflection to compute in terms of bessel functions of first kind (http://dlmf.nist.gov/9.6.E5)
        # use computation for real numbers then convert to input type for stability
        # for imag(z) != 0 use rotation identity (http://dlmf.nist.gov/9.2.E14)
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(one(T)/3, xx)
            Jmv = (Jv - sqrt(T(3)) * Yv) / 2
            return convert(eltype(z), sqrt(xabs) * (Jmv + Jv) / 3)
        else
            return cispi(one(T)/3) * _airyai(-z*cispi(one(T)/3))  + cispi(-one(T)/3) * _airyai(-z*cispi(-one(T)/3))
        end
    end
end

"""
    airyaiprime(z)

Returns the derivative of the Airy function of the first kind, ``\\operatorname{Ai}'(z)``.
Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airyaiprime(1.2)
-0.13278537855722622

julia> airyaiprime(1.2 + 1.4im)
-0.02884977394212135 + 0.21117856532576215im
```

External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airyai`](@ref), [`airybi`](@ref)
"""
airyaiprime(z::Number) = _airyaiprime(float(z))

_airyaiprime(x::Float16) = Float16(_airyaiprime(Float32(x)))
_airyaiprime(z::ComplexF16) = ComplexF16(_airyaiprime(ComplexF32(z)))

function _airyaiprime(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2*T(π)/3
            return -exp(-z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airyaiprime_large_argument(z)
    airyai_power_series_cutoff(x, y) && return airyaiprime_power_series(z)

    if x > zero(T)
        # use relation to besselk (http://dlmf.nist.gov/9.6.E2)
        zz = 2 * z * sqrt(z) / 3
        return -z * besselk_continued_fraction_shift(T(2)/3, zz) / (T(π) * sqrt(T(3)))
    else
        # z is close to the negative real axis
        # for imag(z) == 0 use reflection to compute in terms of bessel functions of first kind (http://dlmf.nist.gov/9.6.E5)
        # use computation for real numbers then convert to input type for stability
        # for imag(z) != 0 use rotation identity (http://dlmf.nist.gov/9.2.E14)
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(T(2)/3, xx)
            Jmv = -(Jv + sqrt(T(3))*Yv) / 2
            return convert(eltype(z), xabs * (Jv - Jmv) / 3)
        else
            return -(cispi(T(2)/3) * _airyaiprime(-z * cispi(one(T)/3)) + cispi(-T(2)/3) * _airyaiprime(-z * cispi(-one(T)/3)))
        end
    end
end

"""
    airybi(z)

Returns the Airy function of the second kind, ``\\operatorname{Bi}(z)``, which is the second solution to the Airy differential equation ``f''(z) - z f(z) = 0``.
Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airybi(1.2)
1.4211336756103483

julia> airybi(1.2 + 1.4im)
0.3150484065220768 + 0.7138432162853446im
```

External links: [DLMF](https://dlmf.nist.gov/9.2.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airybiprime`](@ref), [`airyai`](@ref)
"""
airybi(z::Number) = _airybi(float(z))

_airybi(x::Float16) = Float16(_airybi(Float32(x)))
_airybi(z::ComplexF16) = ComplexF16(_airybi(ComplexF32(z)))

function _airybi(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2π/3
            return exp(z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airybi_large_argument(z)
    airybi_power_series_cutoff(x, y) && return airybi_power_series(z)

    if x > zero(T)
        zz = 2 * z * sqrt(z) / 3
        shift = 20
        order = one(T)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return sqrt(z/3) * (inu + inu2)
    else
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(one(T)/3, xx)
            Jmv = (Jv - sqrt(T(3)) * Yv) / 2
            return convert(eltype(z), sqrt(xabs/3) * (Jmv - Jv))
        else
            return cispi(one(T)/3) * _airybi(-z * cispi(one(T)/3))  + cispi(-one(T)/3) * _airybi(-z*cispi(-one(T)/3))
        end
    end
end

"""
    airybiprime(z)

Returns the derivative of the Airy function of the second kind, ``\\operatorname{Bi}'(z)``.
Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airybiprime(1.2)
1.221231398704895

julia> airybiprime(1.2 + 1.4im)
-0.5250248310153249 + 0.9612736841097036im
```

External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airybi`](@ref), [`airyai`](@ref)
"""
airybiprime(z::Number) = _airybiprime(float(z))

_airybiprime(x::Float16) = Float16(_airybiprime(Float32(x)))
_airybiprime(z::ComplexF16) = ComplexF16(_airybiprime(ComplexF32(z)))

function _airybiprime(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2*T(π)/3
            return exp(z)
        else
            return -1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airybiprime_large_argument(z)
    airybi_power_series_cutoff(x, y) && return airybiprime_power_series(z)

    if x > zero(T)
        zz = 2 * z * sqrt(z) / 3
        shift = 20
        order = T(2)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return z / sqrt(3) * (inu + inu2)
    else
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(T(2)/3, xx)
            Jmv = -(Jv + sqrt(T(3))*Yv) / 2
            return convert(eltype(z), xabs * (Jv + Jmv) / sqrt(T(3)))
        else
            return -(cispi(T(2)/3) * _airybiprime(-z*cispi(one(T)/3)) + cispi(-T(2)/3) * _airybiprime(-z*cispi(-one(T)/3)))
        end
    end
end

#####
##### Power series for airyai(x)
#####

# cutoffs for power series valid for both airyai and airyaiprime
airyai_power_series_cutoff(x::T, y::T) where T <: Float64 = x < 2 && abs(y) > -1.4*(x + 5.5)
airyai_power_series_cutoff(x::T, y::T) where T <: Float32 = x < 4.5f0 && abs(y) > -1.4f0*(x + 9.5f0)

function airyai_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(0.3550280538878172)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_TWO_THIRDS(T)
    t2 = 3*x / GAMMA_ONE_THIRD(T)
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(2)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(4)/3))
    end
    return (ai1*3^(-T(2)/3) - ai2*3^(-T(4)/3))
end
airyai_power_series(x::Float32) = Float32(airyai_power_series(Float64(x), tol=eps(Float32)))
airyai_power_series(x::ComplexF32) = ComplexF32(airyai_power_series(ComplexF64(x), tol=eps(Float32)))

#####
##### Power series for airyaiprime(x)
#####

function airyaiprime_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(-0.2588194037928068)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_ONE_THIRD(T)
    t2 = 3*x2 / (2*GAMMA_TWO_THIRDS(T))
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(1)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(5)/3))
    end
    return -ai1*3^(-T(1)/3) + ai2*3^(-T(5)/3)
end
airyaiprime_power_series(x::Float32) = Float32(airyaiprime_power_series(Float64(x), tol=eps(Float32)))
airyaiprime_power_series(x::ComplexF32) = ComplexF32(airyaiprime_power_series(ComplexF64(x), tol=eps(Float32)))

#####
##### Power series for airybi(x)
#####

# cutoffs for power series valid for both airybi and airybiprime
# has a more complicated validity as it works well close to positive real line and for small negative arguments also works for angle(z) ~ 2pi/3
# the statements are somewhat complicated but we want to hit this branch when we can as the other algorithms are 10x slower
# the Float32 cutoff can be simplified because it overlaps with the large argument expansion so there isn't a need for more complicated statements
airybi_power_series_cutoff(x::T, y::T) where T <: Float64 = (abs(y) > -1.4*(x + 5.5) && abs(y) < -2.2*(x - 4)) || (x > zero(T) && abs(y) < 3)
airybi_power_series_cutoff(x::T, y::T) where T <: Float32 = abs(complex(x, y)) < 9

function airybi_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(0.6149266274460007)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_TWO_THIRDS(T)
    t2 = 3*x / GAMMA_ONE_THIRD(T)
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(2)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(4)/3))
    end
    return (ai1*3^(-T(1)/6) + ai2*3^(-T(5)/6))
end
airybi_power_series(x::Float32) = Float32(airybi_power_series(Float64(x), tol=eps(Float32)))
airybi_power_series(x::ComplexF32) = ComplexF32(airybi_power_series(ComplexF64(x), tol=eps(Float32)))

#####
##### Power series for airybiprime(x)
#####

function airybiprime_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(0.4482883573538264)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_ONE_THIRD(T)
    t2 = 3*x2 / (2*GAMMA_TWO_THIRDS(T))
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(1)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(5)/3))
    end
    return (ai1*3^(T(1)/6) + ai2*3^(-T(7)/6))
end
airybiprime_power_series(x::Float32) = Float32(airybiprime_power_series(Float64(x), tol=eps(Float32)))
airybiprime_power_series(x::ComplexF32) = ComplexF32(airybiprime_power_series(ComplexF64(x), tol=eps(Float32)))

#####
#####  Large argument expansion for airy functions
#####
airy_large_argument_cutoff(z::ComplexOrReal{Float64}) = abs(z) > 8
airy_large_argument_cutoff(z::ComplexOrReal{Float32}) = abs(z) > 4

function airyai_large_argument(x::Real)
    x < zero(x) && return real(airyai_large_argument(complex(x)))
    return airy_large_arg_a(abs(x))
end

function airyai_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    a = airy_large_arg_a(z)
    if x < zero(T) && abs(y) < 5.5
        b = airy_large_arg_b(z)
        y >= zero(T) ? (return a + im*b) : (return a - im*b)
    end
    return a
end

function airyaiprime_large_argument(x::Real)
    x < zero(x) && return real(airyaiprime_large_argument(complex(x)))
    return airy_large_arg_c(abs(x))
end

function airyaiprime_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    c = airy_large_arg_c(z)
    if x < zero(T) && abs(y) < 5.5
        d = airy_large_arg_d(z)
        y >= zero(T) ? (return c + im*d) : (return c - im*d)
    end
    return c
end

function airybi_large_argument(x::Real)
    if x < zero(x)
        return 2*real(airy_large_arg_b(complex(x)))
    else
        return 2*(airy_large_arg_b(x))
    end
end

function airybi_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    b = airy_large_arg_b(z)
    abs(y) <= 1.7*(x - 6) && return 2*b

    check_conj = false
    if y < zero(T)
        z = conj(z)
        b = conj(b)
        y = abs(y)
        check_conj = true
    end

    a = airy_large_arg_a(z)
    if x < zero(T) && y < 5
        out = b + im*a
        check_conj && (out = conj(out))
        return out
    else
        out = 2*b + im*a
        check_conj && (out = conj(out))
        return out
    end
end

function airybiprime_large_argument(x::Real)
    if x < zero(x)
        return 2*real(airy_large_arg_d(complex(x)))
    else
        return 2*(airy_large_arg_d(x))
    end
end

function airybiprime_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    d = airy_large_arg_d(z)
    abs(y) <= 1.7*(x - 6) && return 2*d

    check_conj = false
    if y < zero(T)
        z = conj(z)
        d = conj(d)
        y = abs(y)
        check_conj = true
    end

    c = airy_large_arg_c(z)
    if x < zero(T) && y < 5
        out = d + im*c
        check_conj && (out = conj(out))
        return out
    else
        out = 2*d + im*c
        check_conj && (out = conj(out))
        return out
    end
end

# see equations 24 and relations using eq 25 and 26 in [1]
function airy_large_arg_a(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = GAMMA_ONE_SIXTH(T) * GAMMA_FIVE_SIXTHS(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= -3*(i + one(T)/6) * (i + T(5)/6) / (a*(i + one(T)))
    end
    return out * exp(-a / 6) / (sqrt(T(π)^3) * sqrt(xsqr))
end

function airy_large_arg_b(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = GAMMA_ONE_SIXTH(T) * GAMMA_FIVE_SIXTHS(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= 3*(i + one(T)/6) * (i + T(5)/6) / (a*(i + one(T)))
    end
    return out * exp(a / 6) / (sqrt(T(π)^3) * sqrt(xsqr))
end

function airy_large_arg_c(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    # use identities of gamma to relate to defined constants
    # t = gamma(-one(T) / 6) * gamma(T(7) / 6) / 4
    t = -GAMMA_FIVE_SIXTHS(T) * GAMMA_ONE_SIXTH(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= -3*(i - one(T)/6) * (i + T(7)/6) / (a*(i + one(T)))
    end
    return out * exp(-a / 6) * sqrt(xsqr) / sqrt(T(π)^3)
end

function airy_large_arg_d(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    # use identities of gamma to relate to defined constants
    # t = gamma(-one(T) / 6) * gamma(T(7) / 6) / 4
    t = -GAMMA_FIVE_SIXTHS(T) * GAMMA_ONE_SIXTH(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= 3*(i - one(T)/6) * (i + T(7)/6) / (a*(i + one(T)))
    end
    return -out * exp(a / 6) * sqrt(xsqr) / sqrt(T(π)^3)
end

# negative arguments of airy functions oscillate around zero so as x -> -Inf it is more prone to cancellation
# to give best accuracy it is best to promote to Float64 numbers until the Float32 tolerance
airy_large_arg_a(x::Float32) = (airy_large_arg_a(Float64(x), tol=eps(Float32)))
airy_large_arg_a(x::ComplexF32) = (airy_large_arg_a(ComplexF64(x), tol=eps(Float32)))

airy_large_arg_b(x::Float32) = Float32(airy_large_arg_b(Float64(x), tol=eps(Float32)))
airy_large_arg_b(x::ComplexF32) = ComplexF32(airy_large_arg_b(ComplexF64(x), tol=eps(Float32)))

airy_large_arg_c(x::Float32) = Float32(airy_large_arg_c(Float64(x), tol=eps(Float32)))
airy_large_arg_c(x::ComplexF32) = ComplexF32(airy_large_arg_c(ComplexF64(x), tol=eps(Float32)))

airy_large_arg_d(x::Float32) = Float32(airy_large_arg_d(Float64(x), tol=eps(Float32)))
airy_large_arg_d(x::ComplexF32) = ComplexF32(airy_large_arg_d(ComplexF64(x), tol=eps(Float32)))
# calculates besselk from the power series of besseli using the continued fraction and wronskian
# this shift the order higher first to avoid cancellation in the power series of besseli along the imaginary axis
# for real arguments this is not needed because besseli can be computed stably over the entire real axis
function besselk_continued_fraction_shift(nu, x)
    shift = 20
    inu, inum1 = besseli_power_series_inu_inum1(nu + shift, x)
    inu, inum1 = besselk_down_recurrence(x, inum1, inu, nu + shift - 1, nu)
    H_knu = besselk_ratio_knu_knup1(nu-1, x)
    return 1 / (x * (inum1 + inu / H_knu))
end
