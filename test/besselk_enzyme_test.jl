using EnzymeCore, Enzyme
import Bessels.BesselFunctions: besselkx_levin
import Bessels.BesselFunctions: besselk_power_series

dbesselkx_dv(v, x) = only(autodiff(Forward, _v->besselkx_levin(_v, x, Val(30)),
                              Duplicated, Duplicated(v, 1.0)))

dbesselkx_dx(v, x) = only(autodiff(Forward, _x->besselkx_levin(v, _x, Val(30)),
                              Duplicated, Duplicated(x, 1.0)))

dbesselk_ps_dv(v, x) = only(autodiff(Forward, _v->besselk(_v, x),
                                Duplicated, Duplicated(v, 1.0)))

dbesselk_ps_dx(v, x) = only(autodiff(Forward, _x->besselk(v, _x),
                                Duplicated, Duplicated(x, 1.0)))


@testset "Besselkx autodiff" begin
for line in eachline("data/besselk/enzyme/besselkx_levin_enzyme_tests.csv")
    local v, x
    (v, x, dv, dx) = parse.(Float64, split(line))
    test_dv = dbesselkx_dv(v, x)
    test_dx = dbesselkx_dx(v, x)
    @test isapprox(dv, test_dv, rtol=5e-14)
    @test isapprox(dx, test_dx, rtol=5e-14)
end
end

@testset "Besselk autodiff" begin
for line in eachline("data/besselk/enzyme/besselk_power_series_enzyme_tests.csv")
    local v, x
    (v, x, dv, dx) = parse.(Float64, split(line))
    test_dv   = dbesselk_ps_dv(v, x)
    test_dx   = dbesselk_ps_dx(v, x)
    if abs(v) <= 1e-8
        @test isapprox(dv, test_dv, rtol=1e-7)
        @test isapprox(dx, test_dx, rtol=1e-7)
    else
        @test isapprox(dv, test_dv, rtol=5e-14)
        @test isapprox(dx, test_dx, rtol=5e-14)
    end
end
end
