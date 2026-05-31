using LinearAlgebra, Plots

include("../src/optimization.jl")

function plot_lims(center, radius)
    xlims = (center[1]-radius[1], center[1]+radius[1])
    ylims = (center[2]-radius[2], center[2]+radius[2])
    return xlims, ylims
end

A = [
    7.5 2.0
    2.0 5.5
]

# Definição do problema a ser otimizado
L(x, d) = 0.5*norm(A*x - d)^2
x0 = [2.0, 2.0]
data = [1.0, 2.0]
problem = OptimizationProblem(L, x0, data)

f(x) = L(x, data)

values1 = gradient_descent(problem, save=true, epoch=10000, tol=1e-8)
values2 = newton(problem, save=true, epoch=100, tol=1e-8)
values3 = bfgs(problem, save=true, epoch=100, tol=1e-8)

minimal = A \ data

display(values1[end])
display(values2[end])
display(values3[end])

values1 = reduce(hcat, values1)
values2 = reduce(hcat, values2)
values3 = reduce(hcat, values3)

plot_center = [1.0, 1.0]
plot_radius = [0.5, 0.5]
plot_radius = 1.5 * ones(2)

xrange, yrange = plot_lims(plot_center, plot_radius)

# Plot
z(x, y) = f([x, y])
x1 = range(xrange[1], xrange[2], length=100)
x2 = range(yrange[1], yrange[2], length=100)
zf = @. z(x1', x2)

marker = :cross
ms = 2.5

plt = contour(x1, x2, zf, title="Gradiente descendente", xlims=xrange, ylims=yrange)
plot!(values1[1,:], values1[2,:], label="Gradiente Descendente", marker=marker, markersize=ms)
plot!(values2[1,:], values2[2,:], label="Newton-Raphson", marker=marker, markersize=ms)
plot!(values3[1,:], values3[2,:], label="BFGS", marker=marker, markersize=ms)
scatter!([minimal[1]], [minimal[2]], label="Mínimo local", marker=marker)
display(plt)
readline()

