```julia
using ArbNumerics, DelimitedFiles, SpecialFunctions

# Because you can't just use FiniteDifferences due to the "numerical noise".
simplefd(f,x,h=ArbReal(1e-40)) = (f(x+h)-f(x))/h

ArbNumerics.setprecision(ArbReal, digits=100)
arb_besselk(v,x)  = ArbNumerics.besselk(ArbReal(v), ArbReal(x))
arb_besselkx(v,x) = arb_besselk(v,x)*ArbNumerics.exp(ArbReal(x))

#
# besselkx_levin test:
#
vgrid = sort(vcat(range(0.05, 15.0, length=20), [0.5, 1.5, 2.5, 3.5]))
xgrid = range(15.0, 30.0, length=20)
vx = vec(collect(Iterators.product(vgrid, xgrid)))

ref_values = map(vx) do vxj
  (v,x) = vxj
  dx = simplefd(_x->arb_besselkx(v, _x), x)
  dv = simplefd(_v->arb_besselkx(_v, x), v)
  Float64.((dv, dx))
end

out_matrix = hcat(getindex.(vx, 1),          # test v argument
                  getindex.(vx, 2),          # test x argument
                  getindex.(ref_values, 1),  # test d/dv value
                  getindex.(ref_values, 2))  # test d/dx value

writedlm("besselkx_levin_enzyme_tests.csv", out_matrix)

#
# besselk_power_series test:
#
vgrid = [-1e-8, 1e-8, 1-1e-8, 1.0, 1+1e-8, 2-1e-8, 2.0, 2+1e-8, 3-1e-8, 3.0, 3+1e-8] 
xgrid = range(1e-5, 1.5, length=20)
vx = vec(collect(Iterators.product(vgrid, xgrid)))

ref_values = map(vx) do vxj
  (v,x) = vxj
  dx = simplefd(_x->arb_besselk(v, _x), x) # NOT besselkx!
  dv = simplefd(_v->arb_besselk(_v, x), v) # NOT besselkx!
  Float64.((dv, dx))
end

out_matrix = hcat(getindex.(vx, 1),          # test v argument
                  getindex.(vx, 2),          # test x argument
                  getindex.(ref_values, 1),  # test d/dv value
                  getindex.(ref_values, 2))  # test d/dx value

writedlm("besselk_power_series_enzyme_tests.csv", out_matrix)
```
