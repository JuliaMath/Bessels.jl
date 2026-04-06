for (T, max, rtol) in ((Float16, 13, 1.0), (Float32, 43, 1.0), (Float64, 170, 7))
    v = rand(T, 10000)*max
    for x in v
        @test isapprox(T(SpecialFunctions.gamma(widen(x))), Bessels.gamma(x), rtol=rtol*eps(T))
        if isinteger(x) && x != 0
            @test_throws DomainError Bessels.gamma(-x)
        else
            @test isapprox(T(SpecialFunctions.gamma(widen(-x))), Bessels.gamma(-x), atol=nextfloat(T(0.),2), rtol=rtol*eps(T))
        end
    end
    @test isnan(Bessels.gamma(T(NaN)))
    @test isinf(Bessels.gamma(T(Inf)))
end

x = [0, 1, 2, 3, 8, 15, 20, 30]
@test SpecialFunctions.gamma.(x) ≈ Bessels.gamma.(x)

# loggamma tests
using Bessels.GammaFunctions: gamma, loggamma, logabsgamma, logfactorial, _loggamma_oracle64_point

@testset "logfactorial" begin
    @test logfactorial(0) ≈ 0.0 atol=5eps(Float64)
    for n in (1, 2, 3, 10, 50)
        @test logfactorial(n) ≈ loggamma(Float64(n + 1))
        @test logfactorial(n) ≈ SpecialFunctions.loggamma(Float64(n + 1)) atol=10eps(Float64)
    end
    @test logfactorial(Int32(7)) ≈ loggamma(8.0)
    @test_throws DomainError logfactorial(-1)
end

# real loggamma for Float64, Float32, Float16 against SpecialFunctions.jl
# Note: Stirling-based approach has higher relative error near loggamma zeros (x ≈ 1, 2)
for (T, max, rtol) in ((Float16, 13, 1.0), (Float32, 43, 1.0), (Float64, 170, 7))    
    v = rand(T, 5000) * max
    for x in v
        ref = T(SpecialFunctions.loggamma(widen(x)))
        @test isapprox(ref, loggamma(x), atol=1e-10, rtol=rtol*eps(T))
    end
    @test isnan(loggamma(T(NaN)))
    @test loggamma(T(Inf)) == T(Inf)
end

@test gamma(0.29384) ≈ exp(loggamma(0.29384))
@test gamma(0.29384+0.12938im) ≈ exp(loggamma(0.29384+0.12938im))

# logabsgamma
for (T, rtol) in ((Float16, 1.0), (Float32, 1.0), (Float64, 7))
    for x in T[0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, -0.5, -1.5, -2.5, -3.5, -10.5]
        y1, s1 = logabsgamma(x)
        y2, s2 = SpecialFunctions.logabsgamma(Float64(x))
        @test isapprox(y1, T(y2), atol=T(1e-3), rtol=rtol*eps(T))
        @test s1 == s2
    end
end

# logabsgamma edge cases and SpecialFunctions behavior consistency
@test logabsgamma(0.0) == (Inf, 1)
@test logabsgamma(-0.0) == (Inf, -1)
@test logabsgamma(-1.0) == (Inf, 1)
@test logabsgamma(-2.0) == (Inf, 1)
@test isnan(logabsgamma(NaN)[1])
@test logabsgamma(NaN)[2] == 1
# real loggamma should throw for negative gamma
@test_throws DomainError loggamma(-0.5)
@test loggamma(-1.5) == logabsgamma(-1.5)[1]

# complex loggamma for Float64
for z in [1.0+1.0im, 2.0+0.5im, 0.5+3.0im, 5.0+2.0im, 0.1+0.1im,
          -1.5+0.5im, -0.5+2.0im, 3.0+0.01im, 10.0+5.0im, 0.5+0.01im]
    @test isapprox(loggamma(z), SpecialFunctions.loggamma(z), rtol=7*eps(Float64))
end

# complex loggamma for Float32 and Float16
for z in [1.0f0+1.0f0im, 5.0f0+2.0f0im, 0.5f0+3.0f0im]
    @test isapprox(loggamma(z), Complex{Float32}(SpecialFunctions.loggamma(Complex{Float64}(z))), rtol=eps(Float32))
end

# complex loggamma edge cases and SpecialFunctions consistency
@test loggamma(Complex(Inf, 0.0)) == Complex(Inf, 0.0)
@test all(isnan, reim(loggamma(Complex(NaN, NaN))))
@test loggamma(Complex(0.0, 0.0)) === Complex(Inf, -0.0)
@test loggamma(Complex(-0.0, 0.0)) == Complex(Inf, -Float64(π))

# BigFloat loggamma (real via complex with zero imaginary part)
for x in [big"0.5", big"1.5", big"5.0", big"10.0", big"50.0"]
    @test isapprox(loggamma(x), SpecialFunctions.loggamma(x), rtol=1e-30)
end
@test isapprox(loggamma(big"1.0"), big"0.0", atol=1e-60)
@test isapprox(loggamma(big"2.0"), big"0.0", atol=1e-60)

# Map Complex{Int64} to Complex{Float64} for loggamma tests
@test loggamma(Complex{Int64}(-300)) ≈ loggamma(Complex{Float64}(-300))

    # values taken from Wolfram Alpha
    @testset "loggamma & logabsgamma test cases" begin
        @test loggamma(-300im) ≈ -473.17185074259241355733179182866544204963885920016823743 - 1410.3490664555822107569308046418321236643870840962522425im
        @test loggamma(3.099) ≈ loggamma(3.099+0im) ≈ 0.786413746900558058720665860178923603134125854451168869796
        @test loggamma(1.15) ≈ loggamma(1.15+0im) ≈ -0.06930620867104688224241731415650307100375642207340564554
        @test logabsgamma(0.89)[1] ≈ loggamma(0.89+0im) ≈ 0.074022173958081423702265889979810658434235008344573396963
        @test loggamma(0.91) ≈ loggamma(0.91+0im) ≈ 0.058922567623832379298241751183907077883592982094770449167
        @test loggamma(0.01) ≈ loggamma(0.01+0im) ≈ 4.599479878042021722513945411008748087261001413385289652419
        @test loggamma(-3.4-0.1im) ≈ -1.1733353322064779481049088558918957440847715003659143454 + 12.331465501247826842875586104415980094316268974671819281im
        @test loggamma(-13.4-0.1im) ≈ -22.457344044212827625152500315875095825738672314550695161 + 43.620560075982291551250251193743725687019009911713182478im
        @test loggamma(-13.4+0.0im) ≈ conj(loggamma(-13.4-0.0im)) ≈ -22.404285036964892794140985332811433245813398559439824988 - 43.982297150257105338477007365913040378760371591251481493im
        @test loggamma(-13.4+8im) ≈ -44.705388949497032519400131077242200763386790107166126534 - 22.208139404160647265446701539526205774669649081807864194im
        @test logabsgamma(1+exp2(-20))[1] ≈ loggamma(1+exp2(-20)+0im) ≈ -5.504750066148866790922434423491111098144565651836914e-7
        @test loggamma(1+exp2(-20)+exp2(-19)*im) ≈ -5.5047799872835333673947171235997541985495018556426e-7 - 1.1009485171695646421931605642091915847546979851020e-6im
        @test loggamma(-300+2im) ≈ -1419.3444991797240659656205813341478289311980525970715668 - 932.63768120761873747896802932133229201676713644684614785im
        @test loggamma(300+2im) ≈ 1409.19538972991765122115558155209493891138852121159064304 + 11.4042446282102624499071633666567192538600478241492492652im
        @test loggamma(1-6im) ≈ -7.6099596929506794519956058191621517065972094186427056304 - 5.5220531255147242228831899544009162055434670861483084103im
        @test loggamma(1-8im) ≈ -10.607711310314582247944321662794330955531402815576140186 - 9.4105083803116077524365029286332222345505790217656796587im
        @test loggamma(1+6.5im) ≈ conj(loggamma(1-6.5im)) ≈ -8.3553365025113595689887497963634069303427790125048113307 + 6.4392816159759833948112929018407660263228036491479825744im
        @test loggamma(1+1im) ≈ conj(loggamma(1-1im)) ≈ -0.6509231993018563388852168315039476650655087571397225919 - 0.3016403204675331978875316577968965406598997739437652369im
        @test loggamma(-pi*1e7 + 6im) ≈ -5.10911758892505772903279926621085326635236850347591e8 - 9.86959420047365966439199219724905597399295814979993e7im
        @test loggamma(-pi*1e7 + 8im) ≈ -5.10911765175690634449032797392631749405282045412624e8 - 9.86959074790854911974415722927761900209557190058925e7im
        @test loggamma(-pi*1e14 + 6im) ≈ -1.0172766411995621854526383224252727000270225301426e16 - 9.8696044010873714715264929863618267642124589569347e14im
        @test loggamma(-pi*1e14 + 8im) ≈ -1.0172766411995628137711690403794640541491261237341e16 - 9.8696044010867038531027376655349878694397362250037e14im
        @test loggamma(2.05 + 0.03im) ≈ conj(loggamma(2.05 - 0.03im)) ≈ 0.02165570938532611215664861849215838847758074239924127515 + 0.01363779084533034509857648574107935425251657080676603919im
        @test loggamma(2+exp2(-20)+exp2(-19)*im) ≈ 4.03197681916768997727833554471414212058404726357753e-7 + 8.06398296652953575754782349984315518297283664869951e-7im
    end


@testset "Complex{BigFloat}" begin
        for p in (128, 256, 512)
            setprecision(p) do
                # loggamma test cases (taken from WolframAlpha)
                @test loggamma(Complex{BigFloat}(big"1.4", big"3.7")) ≈ big"-3.70940253309968418984314134854809331185736805646502631204023135069388352598997710462597918070702727437506783302173663095854186723146887577228573371509941123007270190112210546396844695302684314798287935259522555883298038366536077208173987529577169748995380430158372715939689026101797721568006330211802429958418317776826022950333285605816823534143341838578285140019591030846382087557833758489229193473180731236704742771447789943609553100247685639021852545170948778374393632642723407449801941928273507786" + big"2.45680905027686511843270815755752124912922749263425191557239423638341066818045733889313969102751920720477160888088575007996966892611180451906681270423482044538930858698226109815063261456588626497471947149577237077662427442208195627430175449469651634280548198970860137426524308382796138277607215437522597736608829802065700440231980717557305458696328442743516723459412895697201557510117612009902085541415052308628636399246616849495938682489375024940706179414665940019738206000502042295770027318071467942"*im
                @test loggamma(Complex{BigFloat}(big"-3.4", big"-0.1")) ≈ big"-1.17333533220647794810490885589189574408477150036591434542448532049403984398309615180598419193372394147554168405641323544591892928508095342822832004280366916047823075044462434956352385622175279286874520918371463520339183833036366580824529184504285582306128824924251346911575648934153502523354232665207345984362970323798726840265637880699099675010674509364494927004863383425318768708942281798027350435165386600514393768950581065682200893415680265730090421639141142873001336343605907188519518351947777089750716521425258995341843362124378354032424181927250141434187420346538318521452730170494364862200575165169281864878658796523132562054308721448489115739548477753359680696950183723903855692026149555142451921517583039596054377943432700206626447104309600454668378148843293998091304506665118850678678831194150927534302652888726369436145442541195405764173555455442990408710747398811929878918982539083715025044202675895474356318799592540065071048146945557430938208873062999453334736614448777192367740242361544911" + big"12.3314655012478268428755861044159800943162689746718192812954820421814925428937569884922482899922182108501683937660262759142172101775968893536424040892149708658863325903803507911939566037154489951252238647647402247597088855701451028588020604254079765523864277968914769300229513694667769438632836597735443816632344312013588575240574588529169102277436974026980920640009348503482380251876347214375511385088076700003044257500058872364330837264152707906065680970651314444536833041753145865573422902672476837825912427417183184497877656780666843913107467161665891469009396090946299606182430537112098946313562060208498550350090478607894628838869289194986670248766434102637364025088229859997978069923936222838027747044361676772501869744770389248222868789957322667506469525282107480146222663211761905584280946561123905319927661045637168771975340291467464330952607239039478342492953711672563497381915457648833579449139976570359273070513506747282021008512940692328868075470257049078667016119521218353123785031967384424" * im
                @test loggamma(Complex{BigFloat}(big"1.0", big"6.5")) ≈ big"-8.35533650251135956898874979636340693034277901250481133076775315558468225236054299379804391077446370723114013332576474380468988528404564621934767485982173074710822143101705650155694961710942206758861407956691226902034077206603599116440305209875779596048383199145404442671523667694954031176386914820878412589001648171807491290458567807444100704632567345208428201008150473023561239072738489117252232075537269146942018683897902370439498725767025305446910337402590297901357999640802408423888317102332950502" + big"6.43928161597598339481129290184076602632280364914798257448626986803948974762094722413114781643500457429277396907129309037120052638434079511888202779346673590756680534496688216048661275875197218642757656165236731041690196648409171135474570376929222374881526540613424000396812702483748804403461430606647372072917822435446045663422222644485591648971382453371140638905009688610340367386497741974035080606633217160834318470835793011218439321046372790514195519666101799949551367645485542165382267523023837261"*im
                # sanity check against Complex{Float64} values
                @test Complex{Float64}(loggamma(Complex{BigFloat}(big"1.4", big"3.7"))) ≈ loggamma(1.4+3.7im)
                @test Complex{Float64}(loggamma(Complex{BigFloat}(big"-3.4", big"-0.1"))) ≈ loggamma(-3.4-0.1im)
                @test Complex{Float64}(loggamma(Complex{BigFloat}(big"1.0", big"6.5"))) ≈ loggamma(1.0+6.5im)
                @test Complex{Float64}(gamma(Complex{BigFloat}(big"1.4", big"3.7"))) ≈ gamma(1.4+3.7im)
                @test Complex{Float64}(gamma(Complex{BigFloat}(big"-3.4", big"-0.1"))) ≈ gamma(-3.4-0.1im)
                @test Complex{Float64}(gamma(Complex{BigFloat}(big"1.0", big"6.5"))) ≈ gamma(1.0+6.5im)
                # consistency with exp(loggamma)
                @test gamma(Complex{BigFloat}(big"1.4", big"3.7")) ≈ exp(loggamma(Complex{BigFloat}(big"1.4", big"3.7")))
                @test gamma(Complex{BigFloat}(big"-3.4", big"-0.1")) ≈ exp(loggamma(Complex{BigFloat}(big"-3.4", big"-0.1")))
                @test gamma(Complex{BigFloat}(big"1.0", big"6.5")) ≈ exp(loggamma(Complex{BigFloat}(big"1.0", big"6.5")))
                # zero-imaginary inputs should match the real BigFloat results
                @test loggamma(Complex{BigFloat}(big"3.099", big"0.0")) ≈ loggamma(big"3.099")
                @test loggamma(Complex{BigFloat}(big"1.15", big"0.0")) ≈ loggamma(big"1.15")
                @test gamma(Complex{BigFloat}(big"3.099", big"0.0")) ≈ gamma(big"3.099")
                # branch mapping
                ε = BigFloat(2)^(-1400)
                xs = BigFloat.(["-0.1", "-0.7", "-1.3", "-2.6", "-3.4", "-5.2"])
                for x in xs
                    z_minus = Complex{BigFloat}(x, -ε)
                    z_plus  = Complex{BigFloat}(x,  ε)

                    Lm = loggamma(z_minus)
                    Lp = loggamma(z_plus)

                    zf_minus = _loggamma_oracle64_point(z_minus)
                    zf_plus  = _loggamma_oracle64_point(z_plus)
                    Lf_minus = loggamma(zf_minus)
                    Lf_plus  = loggamma(zf_plus)

                    @test isapprox(Float64(real(Lm)), real(Lf_minus); rtol=0, atol=1e-14)
                    @test isapprox(Float64(imag(Lm)), imag(Lf_minus); rtol=0, atol=1e-14)
                    @test isapprox(Float64(real(Lp)), real(Lf_plus);  rtol=0, atol=1e-14)
                    @test isapprox(Float64(imag(Lp)), imag(Lf_plus);  rtol=0, atol=1e-14)

                    kdiff = round(Int, (imag(Lm) - imag(Lp)) / (2*big(pi)))
                    @test kdiff != 0
                    @test isapprox(Lm - Lp, (2*big(pi))*BigFloat(kdiff)*im; rtol=0, atol=big"1e-70")
                end
                # Additional _loggamma_oracle64_point nudge tests at Float64 pole
                n    = -3
                x64  = Float64(n)
                δp64 = nextfloat(x64) - x64
                δm64 = x64 - prevfloat(x64)
                ε64  = eps(Float64)

                # right of pole: rez > n but Float64(rez) == n → nextfloat
                rezR = BigFloat(x64) + BigFloat(δp64)/4
                zR   = Complex{BigFloat}(rezR,  BigFloat(ε64)/4)
                zfR  = _loggamma_oracle64_point(zR)
                @test real(zfR) == nextfloat(x64)
                @test abs(imag(zfR)) ≤ 2ε64

                # left of pole: rez < n but Float64(rez) == n → prevfloat
                rezL = BigFloat(x64) - BigFloat(δm64)/4
                zL   = Complex{BigFloat}(rezL, -BigFloat(ε64)/4)
                zfL  = _loggamma_oracle64_point(zL)
                @test real(zfL) == prevfloat(x64)
                @test abs(imag(zfL)) ≤ 2ε64
            end
        end
    end

        # values taken from Wolfram Alpha
    @testset "loggamma & logabsgamma test cases" begin
        @test loggamma(-300im) ≈ -473.17185074259241355733179182866544204963885920016823743 - 1410.3490664555822107569308046418321236643870840962522425im
        @test loggamma(3.099) ≈ loggamma(3.099+0im) ≈ 0.786413746900558058720665860178923603134125854451168869796
        @test loggamma(1.15) ≈ loggamma(1.15+0im) ≈ -0.06930620867104688224241731415650307100375642207340564554
        @test logabsgamma(0.89)[1] ≈ loggamma(0.89+0im) ≈ 0.074022173958081423702265889979810658434235008344573396963
        @test loggamma(0.91) ≈ loggamma(0.91+0im) ≈ 0.058922567623832379298241751183907077883592982094770449167
        @test loggamma(0.01) ≈ loggamma(0.01+0im) ≈ 4.599479878042021722513945411008748087261001413385289652419
        @test loggamma(-3.4-0.1im) ≈ -1.1733353322064779481049088558918957440847715003659143454 + 12.331465501247826842875586104415980094316268974671819281im
        @test loggamma(-13.4-0.1im) ≈ -22.457344044212827625152500315875095825738672314550695161 + 43.620560075982291551250251193743725687019009911713182478im
        @test loggamma(-13.4+0.0im) ≈ conj(loggamma(-13.4-0.0im)) ≈ -22.404285036964892794140985332811433245813398559439824988 - 43.982297150257105338477007365913040378760371591251481493im
        @test loggamma(-13.4+8im) ≈ -44.705388949497032519400131077242200763386790107166126534 - 22.208139404160647265446701539526205774669649081807864194im
        @test logabsgamma(1+exp2(-20))[1] ≈ loggamma(1+exp2(-20)+0im) ≈ -5.504750066148866790922434423491111098144565651836914e-7
        @test loggamma(1+exp2(-20)+exp2(-19)*im) ≈ -5.5047799872835333673947171235997541985495018556426e-7 - 1.1009485171695646421931605642091915847546979851020e-6im
        @test loggamma(-300+2im) ≈ -1419.3444991797240659656205813341478289311980525970715668 - 932.63768120761873747896802932133229201676713644684614785im
        @test loggamma(300+2im) ≈ 1409.19538972991765122115558155209493891138852121159064304 + 11.4042446282102624499071633666567192538600478241492492652im
        @test loggamma(1-6im) ≈ -7.6099596929506794519956058191621517065972094186427056304 - 5.5220531255147242228831899544009162055434670861483084103im
        @test loggamma(1-8im) ≈ -10.607711310314582247944321662794330955531402815576140186 - 9.4105083803116077524365029286332222345505790217656796587im
        @test loggamma(1+6.5im) ≈ conj(loggamma(1-6.5im)) ≈ -8.3553365025113595689887497963634069303427790125048113307 + 6.4392816159759833948112929018407660263228036491479825744im
        @test loggamma(1+1im) ≈ conj(loggamma(1-1im)) ≈ -0.6509231993018563388852168315039476650655087571397225919 - 0.3016403204675331978875316577968965406598997739437652369im
        @test loggamma(-pi*1e7 + 6im) ≈ -5.10911758892505772903279926621085326635236850347591e8 - 9.86959420047365966439199219724905597399295814979993e7im
        @test loggamma(-pi*1e7 + 8im) ≈ -5.10911765175690634449032797392631749405282045412624e8 - 9.86959074790854911974415722927761900209557190058925e7im
        @test loggamma(-pi*1e14 + 6im) ≈ -1.0172766411995621854526383224252727000270225301426e16 - 9.8696044010873714715264929863618267642124589569347e14im
        @test loggamma(-pi*1e14 + 8im) ≈ -1.0172766411995628137711690403794640541491261237341e16 - 9.8696044010867038531027376655349878694397362250037e14im
        @test loggamma(2.05 + 0.03im) ≈ conj(loggamma(2.05 - 0.03im)) ≈ 0.02165570938532611215664861849215838847758074239924127515 + 0.01363779084533034509857648574107935425251657080676603919im
        @test loggamma(2+exp2(-20)+exp2(-19)*im) ≈ 4.03197681916768997727833554471414212058404726357753e-7 + 8.06398296652953575754782349984315518297283664869951e-7im
    end

    @testset "loggamma for non-finite arguments" begin
        @test loggamma(Inf + 0im) === Inf + 0im
        @test loggamma(Inf - 0.0im) === Inf - 0.0im
        @test loggamma(Inf + 1im) === Inf + Inf*im
        @test loggamma(Inf - 1im) === Inf - Inf*im
        @test loggamma(-Inf + 0.0im) === -Inf - Inf*im
        @test loggamma(-Inf - 0.0im) === -Inf + Inf*im
        @test loggamma(Inf*im) === -Inf + Inf*im
        @test loggamma(-Inf*im) === -Inf - Inf*im
        @test loggamma(Inf + Inf*im) === loggamma(NaN + 0im) === loggamma(NaN*im) === NaN + NaN*im
    end

    @testset "BigFloat" begin
        # test cases (taken from WolframAlpha, computed to 78 digits ≈ 256 bits)
        @test loggamma(big"3.099") ≈ big"0.78641374690055805872066586017892360313412585445116886979672329071050823224651" rtol=1e-67
        @test loggamma(big"1.15") ≈ big"-0.06930620867104688224241731415650307100375642207340564554412494594148673455871" rtol=1e-67
        @test logabsgamma(big"0.89")[1] ≈ big"0.0740221739580814237022658899798106584342350083445733969634566129726762260738245" rtol=1e-67
        @test loggamma(big"0.91") ≈ big"0.0589225676238323792982417511839070778835929820947704491677379048793029707373113" rtol=1e-67
        @test loggamma(big"0.01") ≈ big"4.59947987804202172251394541100874808726100141338528965241917138771477998920321" rtol=1e-67
        @test loggamma(1 + exp2(big"-20.0")) ≈ big"-5.50475006614886679092243442349111109814456565183691425527816079744208067935466e-7" rtol=1e-67

        # consistency
        @test loggamma(big(3124.0)) == log(gamma(big(3124.0)))
        @test loggamma(big(3124.0)) ≈ loggamma(3124.0)
        @test logabsgamma(big(3124.0)) == (loggamma(big(3124.0)), 1)
        @test logabsgamma(big(3124.0))[1] ≈ logabsgamma(3124.0)[1]

        # promotions
        @test loggamma(big(3124)) == loggamma(big(3124.0))
        @test loggamma(big(3//2)) == loggamma(big(1.5))
        @test logabsgamma(big(3124)) == logabsgamma(big(3124.0))
        @test logabsgamma(big(3//2)) == logabsgamma(big(1.5))
        @test loggamma(complex(3//2, 1//3)) ≈ loggamma(complex(1.5, 1 / 3))

        # negative values
        @test loggamma(big(-3.0)) == big(Inf)
        @test loggamma(big(-1.5)) == logabsgamma(big(-1.5))[1]
        @test_throws DomainError loggamma(big(-2.76))

        # non-finite values
        @test isnan(loggamma(big(NaN)))
        @test isnan(logabsgamma(big(NaN))[1])
        @test loggamma(big(Inf)) == big(Inf)
        @test isnan(loggamma(big(-Inf)))
        @test logabsgamma(big(Inf))[1] == big(Inf)
        @test isnan(logabsgamma(big(-Inf))[1])

        # BigFloat signed-zero edge cases for sign of Γ(x)
        @test logabsgamma(BigFloat(0.0)) == (big(Inf), 1)
        @test logabsgamma(BigFloat(-0.0)) == (big(Inf), -1)

        # Complex{BigFloat} non-finite branches
        @test loggamma(Complex{BigFloat}(big(Inf), BigFloat(0.0))) == Complex{BigFloat}(big(Inf), BigFloat(0.0))
        @test loggamma(Complex{BigFloat}(big(Inf), big(1.0))) == Complex{BigFloat}(big(Inf), big(Inf))
        @test loggamma(Complex{BigFloat}(big(Inf), big(-1.0))) == Complex{BigFloat}(big(Inf), big(-Inf))
        @test loggamma(Complex{BigFloat}(big(-Inf), big(1.0))) == Complex{BigFloat}(big(-Inf), big(-Inf))
        @test loggamma(Complex{BigFloat}(big(-Inf), big(-1.0))) == Complex{BigFloat}(big(-Inf), big(Inf))
        @test loggamma(Complex{BigFloat}(big(1.0), big(Inf))) == Complex{BigFloat}(big(-Inf), big(Inf))
        @test loggamma(Complex{BigFloat}(big(1.0), big(-Inf))) == Complex{BigFloat}(big(-Inf), big(-Inf))
        z_nanbf = loggamma(Complex{BigFloat}(big(Inf), big(Inf)))
        @test isnan(real(z_nanbf)) && isnan(imag(z_nanbf))
    end
