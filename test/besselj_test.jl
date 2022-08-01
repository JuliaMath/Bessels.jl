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

#=
Notes
    - power series shows larger error when nu is large (146) and x is small (1.46)
    - asymptotic expansion shows larger error when nu is large or x is large
=#

## test all numbers and orders for 0<nu<100
x = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0, 1.01, 1.05]
nu = [2, 4, 6, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
for v in nu, xx in x
    xx *= v
    sf = SpecialFunctions.besselj(BigFloat(v), BigFloat(xx))
    @test isapprox(besselj(v, xx), Float64(sf), rtol=5e-14)
end

# test half orders (SpecialFunctions does not give big float precision)
# The SpecialFunctions implementation is only accurate to about 1e-11 - 1e-13

x = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0, 1.01, 1.05, 1.08, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]
nu = [0.1, 0.4567, 0.8123, 1.5, 2.5, 4.1234, 6.8, 12.3, 18.9, 28.2345, 38.1235, 51.23, 72.23435, 80.5, 98.5, 104.2]
for v in nu, xx in x
    xx *= v
    @test isapprox(besselj(v, xx), SpecialFunctions.besselj(v, xx), rtol=1e-12)
end

## test large orders
nu = [150, 165.2, 200.0, 300.0, 500.0, 1000.0, 5000.2, 10000.0, 50000.0]
x = [0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92,0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,0.995, 0.999, 1.0, 1.01, 1.05, 1.08, 1.1, 1.2]
for v in nu, xx in x
    xx *= v
    @test isapprox(besselj(v, xx), SpecialFunctions.besselj(v, xx), rtol=5e-11)
end

## test large arguments
@test isapprox(besselj(10.0, 150.0), SpecialFunctions.besselj(10.0, 150.0), rtol=1e-12)

# test BigFloat for single point
@test isapprox(besselj(big"2000", big"1500.0"), SpecialFunctions.besselj(big"2000", big"1500"), rtol=5e-20)
@test isapprox(besselj(big"20", big"1500.0"), SpecialFunctions.besselj(big"20", big"1500"), rtol=5e-20)


# need to test accuracy of negative orders and negative arguments and all combinations within
# SpecialFunctions.jl doesn't provide these so will hand check against hard values
# values taken from https://keisan.casio.com/exec/system/1180573474 which match mathematica
# need to also account for different branches when nu isa integer
nu = -9.102; x = -12.48
@test isapprox(besselj(nu, x), 0.09842356047575545808128 -0.03266486217437818486161im, rtol=8e-15)
nu = -5.0; x = -5.1
@test isapprox(besselj(nu, x), 0.2740038554704588327387, rtol=1e-14)
nu = -7.3; x = 19.1
@test isapprox(besselj(nu, x), 0.1848055978553359009813, rtol=8e-15)
nu = -14.0; x = 21.3
@test isapprox(besselj(nu, x), -0.1962844898264965120021, rtol=8e-15)
nu = 13.0; x = -8.5
@test isapprox(besselj(nu, x), -0.006128034621313167000171, rtol=8e-15)
nu = 17.45; x = -16.23
@test isapprox(besselj(nu, x), -0.01607335977752705869797 -0.1014831996412783806255im, rtol=8e-15)
