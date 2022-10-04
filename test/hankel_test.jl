# most of the tests are contained in bessely_test and besselj_test which test the besseljy function
# which is called by hankelh1 and besselh
# here we will test just a few cases of the overal hankel function
# focusing on negative arguments and reflection

v, x = 1.5, 1.3
@test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x), rtol=2e-13)
@test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x), rtol=2e-13)
@test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x), rtol=2e-13)
@test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x), rtol=2e-13)

v, x = -2.6, 9.2
@test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x), rtol=2e-13)
@test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x), rtol=2e-13)
@test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x), rtol=2e-13)
@test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x), rtol=2e-13)

v, x = -4.0, 11.4
@test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x), rtol=2e-13)
@test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x), rtol=2e-13)
@test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x), rtol=2e-13)
@test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x), rtol=2e-13)

v, x = 14.3, 29.4
@test isapprox(hankelh1(v, x), SpecialFunctions.hankelh1(v, x), rtol=2e-13)
@test isapprox(hankelh2(v, x), SpecialFunctions.hankelh2(v, x), rtol=2e-13)
@test isapprox(besselh(v, 1, x), SpecialFunctions.besselh(v, 1, x), rtol=2e-13)
@test isapprox(besselh(v, 2, x), SpecialFunctions.besselh(v, 2, x), rtol=2e-13)

@test isapprox(hankelh1(1:50, 10.0), SpecialFunctions.hankelh1.(1:50, 10.0), rtol=2e-13)
@test isapprox(hankelh1(0.5:1:25.5, 15.0), SpecialFunctions.hankelh1.(0.5:1:25.5, 15.0), rtol=2e-13)
@test isapprox(hankelh1(1:50, 100.0), SpecialFunctions.hankelh1.(1:50, 100.0), rtol=2e-13)
@test isapprox(hankelh2(1:50, 10.0), SpecialFunctions.hankelh2.(1:50, 10.0), rtol=2e-13)
