
using Documenter
using ABCdeZ

makedocs(
    sitename = "ABCdeZ.jl",
    modules = [ABCdeZ],
    authors = "Maurice Langhinrichs <m.langhinrichs@icloud.com> and contributors",
    )

deploydocs(
    repo = "github.com/mauricelanghinrichs/ABCdeZ.jl.git",
    devbranch = "main",
    )