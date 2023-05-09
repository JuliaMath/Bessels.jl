using EnzymeCore, Enzyme
import Bessels.BesselFunctions: besselkx_levin

dbesselkx_dv(v, x) = autodiff(Forward, _v->besselkx_levin(_v, x, Val(30)), 
                              Duplicated, Duplicated(v, 1.0))[2]

dbesselkx_dx(v, x) = autodiff(Forward, _x->besselkx_levin(v, _x, Val(30)), 
                              Duplicated, Duplicated(x, 1.0))[2]

for line in eachline("data/besselk/enzyme/besselkx_levin_enzyme_tests.csv")
  (v, x, dv, dx) = parse.(Float64, split(line))
  test_dv = dbesselkx_dv(v, x)
  test_dx = dbesselkx_dx(v, x)
  @test isapprox(dv, test_dv, rtol=1e-4)
  @test isapprox(dx, test_dx, rtol=1e-4)
end
