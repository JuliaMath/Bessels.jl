```julia
using FiniteDifferences, ArbNumerics, DelimitedFiles

if !(@isdefined vx)
  const FD = central_fdm(10,1)
  vgrid = sort(vcat(range(0.05, 15.0, length=20), [0.5, 1.5, 2.5, 3.5]))
  xgrid = range(15.0, 30.0, length=20)
  const vx = vec(collect(Iterators.product(vgrid, xgrid)))
end

ArbNumerics.setprecision(ArbReal, digits=100)

function arb_besselkx(v,x) 
  (av, ax) = ArbReal.((v,x))
  ArbNumerics.besselk(av, ax)*ArbNumerics.exp(ax)
end

ref_values = map(vx) do vxj
  (v,x) = vxj
  dx = FD(_x->arb_besselkx(v, _x), x)
  dv = FD(_v->arb_besselkx(_v, x), v)
  Float64.((dv, dx))
end

out_matrix = hcat(getindex.(vx, 1),          # test v argument
                  getindex.(vx, 2),          # test x argument
                  getindex.(ref_values, 1),  # test d/dv value
                  getindex.(ref_values, 2))  # test d/dx value

writedlm("besselkx_levin_enzyme_tests.csv", out_matrix)
```
