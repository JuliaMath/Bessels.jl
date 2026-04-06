using BenchmarkTools
using Bessels
using Bessels.GammaFunctions: loggamma, logabsgamma
using ArbNumerics
import SpecialFunctions

setworkingprecision(ArbFloat, 500)
setextrabits(128)



# compute relative errors: abs(1 - Bessels.f(x)/ArbNumerics.f(ArbFloat(x)))
# 1000 random points in (0, 100), report mean and max

println("=== loggamma Float64 accuracy ===")
x_vals = rand(1000) .* 100
errs = [abs(1 - loggamma(x) / Float64(ArbNumerics.lgamma(ArbFloat(x)))) for x in x_vals]
println("  mean relative error: ", sum(errs)/length(errs))
println("  max  relative error: ", maximum(errs))

println("\n=== loggamma ComplexF64 accuracy ===")
z_vals = [rand()*100 + rand()*100*im for _ in 1:1000]
errs_c = Float64[]
for z in z_vals
    ref = Complex{Float64}(ArbNumerics.lgamma(ArbFloat(real(z)) + ArbFloat(imag(z))*im))
    push!(errs_c, abs(1 - loggamma(z) / ref))
end
println("  mean relative error: ", sum(errs_c)/length(errs_c))
println("  max  relative error: ", maximum(errs_c))

println("\n=== logabsgamma Float64 accuracy ===")
errs_la = Float64[]
for x in x_vals
    ref = Float64(ArbNumerics.lgamma(ArbFloat(x)))
    y, _ = logabsgamma(x)
    push!(errs_la, abs(1 - y / ref))
end
println("  mean relative error: ", sum(errs_la)/length(errs_la))
println("  max  relative error: ", maximum(errs_la))





# performance benchmarks

suite = BenchmarkGroup()

suite["loggamma"] = BenchmarkGroup()
suite["loggamma"]["Bessels_Float64"] = @benchmarkable loggamma(x) setup=(x = rand()*100)
suite["loggamma"]["SpecialFunctions_Float64"] = @benchmarkable SpecialFunctions.loggamma(x) setup=(x = rand()*100)
suite["loggamma"]["Bessels_ComplexF64"] = @benchmarkable loggamma(z) setup=(z = rand(ComplexF64)*100)
suite["loggamma"]["SpecialFunctions_ComplexF64"] = @benchmarkable SpecialFunctions.loggamma(z) setup=(z = rand(ComplexF64)*100)

suite["logabsgamma"] = BenchmarkGroup()
suite["logabsgamma"]["Bessels_Float64"] = @benchmarkable logabsgamma(x) setup=(x = rand()*100)
suite["logabsgamma"]["SpecialFunctions_Float64"] = @benchmarkable SpecialFunctions.logabsgamma(x) setup=(x = rand()*100)

results = run(suite, verbose=true)

println("\n=== loggamma Float64 timing ===")
println("Bessels:          ", results["loggamma"]["Bessels_Float64"])
println("SpecialFunctions: ", results["loggamma"]["SpecialFunctions_Float64"])
println("Speedup (Bessels): ", round(median(results["loggamma"]["SpecialFunctions_Float64"]).time / median(results["loggamma"]["Bessels_Float64"]).time, digits=2), "x")

println("\n=== loggamma ComplexF64 timing ===")
println("Bessels:          ", results["loggamma"]["Bessels_ComplexF64"])
println("SpecialFunctions: ", results["loggamma"]["SpecialFunctions_ComplexF64"])
println("Speedup (Bessels): ", round(median(results["loggamma"]["SpecialFunctions_ComplexF64"]).time / median(results["loggamma"]["Bessels_ComplexF64"]).time, digits=2), "x")

println("\n=== logabsgamma Float64 timing ===")
println("Bessels:          ", results["logabsgamma"]["Bessels_Float64"])
println("SpecialFunctions: ", results["logabsgamma"]["SpecialFunctions_Float64"])
println("Speedup (Bessels): ", round(median(results["logabsgamma"]["SpecialFunctions_Float64"]).time / median(results["logabsgamma"]["Bessels_Float64"]).time, digits=2), "x")
