# test very small inputs
x = 1e-15
@test @inferred(sphericalbesselj(0, x)) ≈ SpecialFunctions.sphericalbesselj(0, x)
@test @inferred(sphericalbesselj(1, x)) ≈ SpecialFunctions.sphericalbesselj(1, x)
@test @inferred(sphericalbesselj(5.5, x)) ≈ SpecialFunctions.sphericalbesselj(5.5, x)
@test @inferred(sphericalbesselj(10, x)) ≈ SpecialFunctions.sphericalbesselj(10, x)
@test @inferred(sphericalbessely(0, x)) ≈ SpecialFunctions.sphericalbessely(0, x)
@test @inferred(sphericalbessely(1, x)) ≈ SpecialFunctions.sphericalbessely(1, x)
@test @inferred(sphericalbessely(5.5, x)) ≈ SpecialFunctions.sphericalbessely(5.5, x)
@test @inferred(sphericalbessely(10, x)) ≈ SpecialFunctions.sphericalbessely(10, x)
@test @inferred(sphericalbesselk(5.5, x)) ≈ SpecialFunctions.besselk(5.5 + 1/2, x) * sqrt( 2 / (x*pi))
@test @inferred(sphericalbesselk(10, x)) ≈ SpecialFunctions.besselk(10 + 1/2, x) * sqrt( 2 / (x*pi))

x = 1e-20
@test @inferred(sphericalbesseli(0, x)) ≈ SpecialFunctions.besseli(0 + 1/2, x) * sqrt( pi / (x*2))
@test @inferred(sphericalbesseli(1, x)) ≈ SpecialFunctions.besseli(1 + 1/2, x) * sqrt( pi / (x*2))
@test @inferred(sphericalbesseli(2, x)) ≈ SpecialFunctions.besseli(2 + 1/2, x) * sqrt( pi / (x*2))
@test @inferred(sphericalbesseli(3, x)) ≈ SpecialFunctions.besseli(3 + 1/2, x) * sqrt( pi / (x*2))
@test @inferred(sphericalbesseli(4, x)) ≈ SpecialFunctions.besseli(4 + 1/2, x) * sqrt( pi / (x*2))
@test @inferred(sphericalbesseli(6.5, x)) ≈ SpecialFunctions.besseli(6.5 + 1/2, x) * sqrt( pi / (x*2))

# test zero
@test isone(sphericalbesselj(0, 0.0))
@test iszero(sphericalbesselj(3, 0.0))
@test iszero(sphericalbesselj(10.4, 0.0))
@test iszero(sphericalbesselj(100.6, 0.0))

@test sphericalbessely(0, 0.0) == -Inf
@test sphericalbessely(1.8, 0.0) == -Inf
@test sphericalbessely(10, 0.0) == -Inf
@test sphericalbessely(290, 0.0) == -Inf

@test isinf(sphericalbesselk(0, 0.0))
@test isinf(sphericalbesselk(4, 0.0))
@test isinf(sphericalbesselk(10.2, 0.0))

x = 0.0
@test isone(sphericalbesseli(0, x))
@test iszero(sphericalbesseli(1, x))
@test iszero(sphericalbesseli(2, x))
@test iszero(sphericalbesseli(3, x))
@test iszero(sphericalbesseli(4, x))
@test iszero(sphericalbesseli(6.4, x))

# test Inf
@test iszero(sphericalbesselj(1, Inf))
@test iszero(sphericalbesselj(10.2, Inf))
@test iszero(sphericalbessely(3, Inf))
@test iszero(sphericalbessely(4.5, Inf))

@test iszero(sphericalbesselk(0, Inf))
@test iszero(sphericalbesselk(4, Inf))
@test iszero(sphericalbesselk(10.2, Inf))

x = Inf
@test isinf(sphericalbesseli(0, x))
@test isinf(sphericalbesseli(1, x))
@test isinf(sphericalbesseli(2, x))
@test isinf(sphericalbesseli(3, x))
@test isinf(sphericalbesseli(4, x))
@test isinf(sphericalbesseli(6.4, x))

# test NaN
@test isnan(sphericalbesselj(1.4, NaN))
@test isnan(sphericalbesselj(4.0, NaN))
@test isnan(sphericalbessely(1.4, NaN))
@test isnan(sphericalbessely(4.0, NaN))

@test isnan(sphericalbesselk(1.4, NaN))
@test isnan(sphericalbesselk(4.0, NaN))

x = NaN
@test isnan(sphericalbesseli(0, x))
@test isnan(sphericalbesseli(1, x))
@test isnan(sphericalbesseli(2, x))
@test isnan(sphericalbesseli(3, x))
@test isnan(sphericalbesseli(4, x))
@test isnan(sphericalbesseli(6.4, x))

# test Float16, Float32 types
@test @inferred(sphericalbesselj(Float16(1.4), Float16(1.2))) isa Float16
@test @inferred(sphericalbessely(Float16(1.4), Float16(1.2))) isa Float16
@test @inferred(sphericalbesselj(1.4f0, 1.2f0)) isa Float32
@test @inferred(sphericalbessely(1.4f0, 1.2f0)) isa Float32

@test sphericalbesselk(Float16(1.4), Float16(1.2)) isa Float16
@test sphericalbesselk(1.0f0, 1.2f0) isa Float32

@test sphericalbesseli(Float16(1.4), Float16(1.2)) isa Float16
@test sphericalbesseli(1.0f0, 1.2f0) isa Float32

for x in 0.5:1.5:100.0, v in [0, 1, 2, 3, 4, 5.5, 8.2, 10]
    @test isapprox(sphericalbesselj(v, x), SpecialFunctions.sphericalbesselj(v, x), rtol=1e-12)
    @test isapprox(sphericalbessely(v, x), SpecialFunctions.sphericalbessely(v, x), rtol=1e-12)
    @test isapprox(sphericalbesselk(v, x), SpecialFunctions.besselk(v+1/2, x) * sqrt( 2 / (x*pi)), rtol=1e-12)
    @test isapprox(sphericalbesseli(v, x), SpecialFunctions.besseli(v+1/2, x) * sqrt( pi / (x*2)), rtol=1e-12)
end

for x in 5.5:4.0:160.0, v in [20, 25.0, 32.4, 40.0, 45.12, 50.0, 55.2, 60.124, 70.23, 75.0, 80.0, 92.3, 100.0, 120.0]
    @test isapprox(sphericalbesselj(v, x), SpecialFunctions.sphericalbesselj(v, x), rtol=3e-12)
    @test isapprox(sphericalbessely(v, x), SpecialFunctions.sphericalbessely(v, x), rtol=3e-12)
    @test isapprox(sphericalbesselk(v, x), SpecialFunctions.besselk(v+1/2, x) * sqrt( 2 / (x*pi)), rtol=1e-12)
    @test isapprox(sphericalbesseli(v, x), SpecialFunctions.besseli(v+1/2, x) * sqrt( pi / (x*2)), rtol=1e-12)
end

@test isapprox(sphericalbessely(270, 240.0), SpecialFunctions.sphericalbessely(270, 240.0), rtol=3e-12)

v, x = -4.0, 5.6
@test isapprox(sphericalbesselj(v, x), 0.07774965105230025584537, rtol=3e-12)
@test isapprox(sphericalbessely(v, x), -0.1833997131521190346258, rtol=3e-12)

v, x = -6.8, 15.6
@test isapprox(sphericalbesselj(v, x), 0.04386355397884301866595, rtol=3e-12)
@test isapprox(sphericalbessely(v, x), 0.05061013363904335437354, rtol=3e-12)

# test for negative order of spherical modified besselk in the special integer
# routine:
for v in 1:10
  @test sphericalbesselk(-v, 1.1) ≈ sphericalbesselk(v-1, 1.1)
end
