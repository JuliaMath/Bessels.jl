# most of the tests are contained in bessely_test and besselj_test which test the besseljy function
# which is called by hankelh1 and besselh
# here we will test just a few cases of the overall hankel function
# focusing on negative arguments and reflection

@testset "$T" for T in (Float32, Float64)
    rtol = T == Float64 ? 2e-13 : 2e-6
    v, x = T(1.5), T(1.3)
    @test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x); rtol)
    @test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x); rtol)
    @test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x); rtol)
    @test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x); rtol)
    @inferred besselh(v, 2, x)
    
    v, x = T(-2.6), T(9.2)
    @test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x); rtol)
    @test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x); rtol)
    @test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x); rtol)
    @test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x); rtol)
    @inferred besselh(v, 2, x)

    v, x = T(-4.0), T(11.4)
    @test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x); rtol)
    @test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x); rtol)
    @test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x); rtol)
    @test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x); rtol)
    @inferred besselh(v, 2, x)

    v, x = T(14.3), T(29.4)
    @test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x); rtol)
    @test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x); rtol)
    @test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x); rtol)
    @test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x); rtol)
    @inferred besselh(v, 2, x)

    @test isapprox(hankelh1(1:50, T(10)), SpecialFunctions.hankelh1.(1:50, 10.0); rtol)
    @test isapprox(hankelh1(T(0.5):T(25.5), T(15)), SpecialFunctions.hankelh1.(0.5:1:25.5, 15.0); rtol)
    @test isapprox(hankelh1(1:50, T(100)), SpecialFunctions.hankelh1.(1:50, 100.0); 2*rtol)
    @test isapprox(hankelh2(1:50, T(10)), SpecialFunctions.hankelh2.(1:50, 10.0); rtol)
    @inferred hankelh2(1:50, T(10))

    #test 2 arg version
    @test besselh(v, 1, x) == besselh(v, x)
    @test besselh(1:50, 1, T(10.0)) == besselh(1:50, T(10.0))
end
