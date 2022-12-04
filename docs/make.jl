push!(LOAD_PATH,"../src/")
using Documenter, Bessels

makedocs(
         sitename = "Bessels.jl",
         modules  = [Bessels],
         pages=[
                "Home" => "index.md",
                "Getting started" => "install.md",
                "Roadmap" => "roadmap.md",
                "Contributing" => "contribute.md",
                "API" => "API.md",
                "Function list" => "functions.md",
               ]
)
               
deploydocs(
    repo="github.com/JuliaMath/Bessels.jl.git",
)