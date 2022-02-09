using BenchmarkTools
using Bessels

suite = BenchmarkGroup()

suite["besseli"] = BenchmarkGroup()
suite["besseli"]["besseli0"] = @benchmarkable besseli0(x) setup=(x = rand()*100)
suite["besseli"]["besseli1"] = @benchmarkable besseli1(x) setup=(x = rand()*100)
suite["besseli"]["besseli20"] = @benchmarkable besseli(20, x) setup=(x = rand()*100)
suite["besseli"]["besseli120"] = @benchmarkable besseli(120, x) setup=(x = rand()*100 + 1.0)

suite["besselk"] = BenchmarkGroup()
suite["besselk"]["besselk0"] = @benchmarkable besselk0(x) setup=(x = rand()*100)
suite["besselk"]["besselk1"] = @benchmarkable besselk1(x) setup=(x = rand()*100)
suite["besselk"]["besselk20"] = @benchmarkable besselk(20, x) setup=(x = rand()*100)
suite["besselk"]["besselk120"] = @benchmarkable besselk(120, x) setup=(x = rand()*100 + 1.0)

suite["besselj"] = BenchmarkGroup()
suite["besselj"]["besselj0"] = @benchmarkable besselj0(x) setup=(x = rand()*100)
suite["besselj"]["besselj1"] = @benchmarkable besselj1(x) setup=(x = rand()*100)

suite["bessely"] = BenchmarkGroup()
suite["bessely"]["bessely0"] = @benchmarkable besselj0(x) setup=(x = rand()*100)
suite["bessely"]["bessely1"] = @benchmarkable besselj1(x) setup=(x = rand()*100)

#tune!(suite)
