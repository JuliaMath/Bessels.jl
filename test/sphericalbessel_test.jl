# test very small inputs
x = 1e-15
@test Bessels.sphericalbesselj(0, x) ≈ SpecialFunctions.sphericalbesselj(0, x)
@test Bessels.sphericalbesselj(1, x) ≈ SpecialFunctions.sphericalbesselj(1, x)
@test Bessels.sphericalbesselj(5.5, x) ≈ SpecialFunctions.sphericalbesselj(5.5, x)
@test Bessels.sphericalbesselj(10, x) ≈ SpecialFunctions.sphericalbesselj(10, x)
@test Bessels.sphericalbessely(0, x) ≈ SpecialFunctions.sphericalbessely(0, x)
@test Bessels.sphericalbessely(1, x) ≈ SpecialFunctions.sphericalbessely(1, x)
@test Bessels.sphericalbessely(5.5, x) ≈ SpecialFunctions.sphericalbessely(5.5, x)
@test Bessels.sphericalbessely(10, x) ≈ SpecialFunctions.sphericalbessely(10, x)

for x in 0.5:1.0:100.0, v in [0, 1, 5.5, 10]
    @test isapprox(Bessels.sphericalbesselj(v, x), SpecialFunctions.sphericalbesselj(v, x), rtol=1e-12)
    @test isapprox(Bessels.sphericalbessely(v, x), SpecialFunctions.sphericalbessely(v, x), rtol=1e-12)
end

for x in 5.5:4.0:160.0, v in [20, 25.0, 32.4, 40.0, 45.12, 50.0, 55.2, 60.124, 70.23, 75.0, 80.0, 92.3, 100.0, 120.0]
    @test isapprox(Bessels.sphericalbesselj(v, x), SpecialFunctions.sphericalbesselj(v, x), rtol=3e-12)
    @test isapprox(Bessels.sphericalbessely(v, x), SpecialFunctions.sphericalbessely(v, x), rtol=3e-12)
end
