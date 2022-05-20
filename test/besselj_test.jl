# general array for testing input to SpecialFunctions.jl
x = 1e-6:0.01:100.0

### Tests for besselj0
j0_SpecialFunctions = SpecialFunctions.besselj0.(big.(x)) # array to be tested against computed in BigFloats
@assert j0_SpecialFunctions[1] isa BigFloat               # just double check the higher precision

j0_64 = besselj0.(Float64.(x))
j0_32 = besselj0.(Float32.(x))
j0_big = besselj0.(big.(x))

# make sure output types match input types
@test j0_64[1] isa Float64
@test j0_32[1] isa Float32
@test j0_big[1] isa BigFloat

# test against SpecialFunctions.jl
@test j0_32 ≈ j0_SpecialFunctions

# BigFloat precision only computed to 128 bits
@test isapprox(j0_big, j0_SpecialFunctions, atol=1.5e-34)

# NaN should return NaN
@test isnan(besselj0(NaN))

# zero should return one
@test isone(besselj0(zero(Float32)))
@test isone(besselj0(zero(Float64)))
@test isone(besselj0(zero(BigFloat)))

# test that Inf inputs go to zero
@test besselj0(Inf32) == zero(Float32)
@test besselj0(Inf64) == zero(Float64)

# test negative inputs
@test besselj0(-2.0f0) ≈ SpecialFunctions.besselj0(-2.0f0)
@test besselj0(-2.0) ≈ SpecialFunctions.besselj0(-2.0)
@test besselj0(-80.0f0) ≈ SpecialFunctions.besselj0(-80.0f0)
@test besselj0(-80.0) ≈ SpecialFunctions.besselj0(-80.0)

### Tests for besselj1
j1_SpecialFunctions = SpecialFunctions.besselj1.(big.(x)) # array to be tested against computed in BigFloats
@assert j1_SpecialFunctions[1] isa BigFloat               # just double check the higher precision

j1_64 = besselj1.(Float64.(x))
j1_32 = besselj1.(Float32.(x))

# make sure output types match input types
@test j1_64[1] isa Float64
@test j1_32[1] isa Float32

# test against SpecialFunctions.jl
@test j1_64 ≈ j1_SpecialFunctions
@test j1_32 ≈ j1_SpecialFunctions

# NaN should return NaN
@test isnan(besselj1(NaN))

# zero should return zero
@test iszero(besselj1(zero(Float32)))
@test iszero(besselj1(zero(Float64)))

# test that Inf inputs go to zero
@test besselj1(Inf32) == zero(Float32)
@test besselj1(Inf64) == zero(Float64)

# test negative inputs
@test besselj1(-2.0f0) ≈ SpecialFunctions.besselj1(-2.0f0)
@test besselj1(-2.0) ≈ SpecialFunctions.besselj1(-2.0)
@test besselj1(-80.0f0) ≈ SpecialFunctions.besselj1(-80.0f0)
@test besselj1(-80.0) ≈ SpecialFunctions.besselj1(-80.0)

## Tests for besselj 
# note this is not complete just a simple test
# this needs work and removing for now

#@test besselj(3, 1.0) ≈ SpecialFunctions.besselj(3, 1.0)
#@test besselj(-5, 6.1) ≈ SpecialFunctions.besselj(-5, 6.1)

## test the small value approximation using power series 
nu = [0.5, 1.5, 3.0, 10.0, 22.2, 35.0, 52.1, 100.0, 250.2, 500.0, 1000.0]
x = [0.001, 1.0, 5.0, 15.0, 19.9]

for v in nu, x in x
    @test Bessels._besselj(v, x) ≈ SpecialFunctions.besselj(v, x)
end

## test the large argument asymptotic expansion
vnu = ((0.5, 21.0), (0.5, 45.0), (3.5, 21.0), (10.0, 21.0), (21.2, 45.0), (43.2, 90.0), (100.1, 205.2))

for (v, x) in vnu
    @test Bessels._besselj(v, x) ≈ SpecialFunctions.besselj(v, x)
end

## test the debye uniform asymptotic expansion for x < nu
vnu = ((60.5, 21.0), (100.5, 45.0), (150.5, 61.0))

for (v, x) in vnu
    @test Bessels._besselj(v, x) ≈ SpecialFunctions.besselj(v, x)
end

## test all numbers and orders for 0<nu<100
x = 0.1:0.5:100.0
nu = 2:100
for v in nu, xx in x
    @show v, xx
    @test isapprox(Bessels._besselj(BigFloat(v), BigFloat(xx)), SpecialFunctions.besselj(BigFloat(v), BigFloat(xx)), rtol=1e-12)
end

# test half orders (SpecialFunctions does not give big float precision)
# The SpecialFunctions implementation is actually very inaccurate
# julia> 1 - SpecialFunctions.besselj(10.5, 29.6) / -0.009263478934797420709865
#-1.63202784619898e-13
# with value from https://keisan.casio.com/exec/system/1180573474
x = 0.1:0.5:100.0
nu = 2.5:1.0:10.5
for v in nu, xx in x
    @test isapprox(Bessels._besselj(BigFloat(v), BigFloat(xx)), SpecialFunctions.besselj(v, xx), rtol=1e-12)
end

