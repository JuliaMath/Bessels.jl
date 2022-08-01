# general array for testing input to SpecialFunctions.jl
x = 0.01:0.01:150.0

### Tests for bessely0
y0_SpecialFunctions = SpecialFunctions.bessely0.(big.(x))  # array to be tested against computed in BigFloats
@assert y0_SpecialFunctions[1] isa BigFloat                # just double check the higher precision

y0_64 = bessely0.(Float64.(x))
y0_32 = bessely0.(Float32.(x))
y0_big = bessely0.(big.(x))

# make sure output types match input types
@test y0_64[1] isa Float64
@test y0_32[1] isa Float32
@test y0_big[1] isa BigFloat

# test against SpecialFunctions.jl
@test y0_SpecialFunctions ≈ y0_64
@test y0_SpecialFunctions ≈ y0_32

# BigFloat precision only computed to 128 bits
@test isapprox(y0_big, y0_SpecialFunctions, atol=1.5e-34)

# negative numbers should result in a domain error
@test_throws DomainError bessely0(-1.0)

# NaN should return NaN
@test isnan(bessely0(NaN))

# test that zero inputs go to -Inf
@test bessely0(zero(Float32)) == -Inf32
@test bessely0(zero(Float64)) == -Inf64
@test bessely0(zero(BigFloat)) == -Inf

# test that Inf inputs go to zero
@test bessely0(Inf32) == zero(Float32)
@test bessely0(Inf64) == zero(Float64)

### tests for bessely1
y1_SpecialFunctions = SpecialFunctions.bessely1.(big.(x))
@assert y1_SpecialFunctions[1] isa BigFloat

y1_64 = bessely1.(Float64.(x))
y1_32 = bessely1.(Float32.(x))

# make sure output types match input types
@test y1_64[1] isa Float64
@test y1_32[1] isa Float32

# test against SpecialFunctions.jl
@test y1_64 ≈ y1_SpecialFunctions
@test y1_32 ≈ y1_SpecialFunctions

# negative numbers should result in a domain error
@test_throws DomainError bessely1(-1.0)

# NaN should return NaN
@test isnan(bessely1(NaN))

# test that zero inputs go to -Inf
@test bessely1(zero(Float32)) == -Inf32
@test bessely1(zero(Float64)) == -Inf64

# test that Inf inputs go to zero
@test bessely1(Inf32) == zero(Float32)
@test bessely1(Inf64) == zero(Float64)

## Tests for bessely

## test all numbers and orders for 0<nu<100
x = [0.05, 0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0, 1.001, 1.01, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0]
nu = [2, 4, 6, 10, 15, 20, 25, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 125, 150, 175, 200]
for v in nu, xx in x
    xx *= v
    sf = SpecialFunctions.bessely(BigFloat(v), BigFloat(xx))
    @test isapprox(Bessels._bessely(v, xx), Float64(sf), rtol=7e-14)
end

# test decimal orders
# SpecialFunctions.jl can give errors over 1e-12 so need to soften tolerance to match
# need to switch tests over to ArbNumerics.jl for better precision tests 
x = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5,0.55,  0.6,0.65,  0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0, 1.01, 1.05, 1.08, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 4.5, 4.99, 5.1]
nu = [0.1, 0.4567, 0.8123, 1.5, 2.5, 4.1234, 6.8, 12.3, 18.9, 28.2345, 38.1235, 51.23, 72.23435, 80.5, 98.5, 104.2]
for v in nu, xx in x
    xx *= v
    @test isapprox(Bessels._bessely(v, xx), SpecialFunctions.bessely(v, xx), rtol=5e-12)
end
