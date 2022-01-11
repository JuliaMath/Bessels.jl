# general array for testing input to SpecialFunctions.jl
x = 0.01:0.01:100.0

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
