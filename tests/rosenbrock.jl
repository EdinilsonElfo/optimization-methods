using LinearAlgebra, Plots

include("../src/optimization.jl")

function plot_lims(center, radius)
    xlims = (center[1]-radius[1], center[1]+radius[1])
    ylims = (center[2]-radius[2], center[2]+radius[2])
    return xlims, ylims
end

# Função de Rosenbrock
a = 1.
b = 20.
c = 2.
rosenbrock(x, a, b, c) = (a - x[1])^2 + b*(x[2] - x[1]^2)^2

f(x,y) = (1 - x)^2 + 20(y - x^2)^2

# Definição do problema a ser otimizado
L(x, d) = 0.5*norm(x - d)^2
x0 = [-1.0, 1.0]
data = [1.0, 20.0, 2.0]
L(x, d) = rosenbrock(x, d[1], d[2], d[3])/d[3]
problem = OptimizationProblem(L, x0, data)

f(x) = L(x, data)

values1 = gradient_descent(problem, save=true, epoch=10000, tol=1e-8)
values2 = newton(problem, save=true, epoch=100, tol=1e-8)
values3 = bfgs(problem, save=true, epoch=100, tol=1e-8)

minimal = [data[1], data[1]*data[1]]

display(values1[end])
display(values2[end])
display(values3[end])

values1 = reduce(hcat, values1)
values2 = reduce(hcat, values2)
values3 = reduce(hcat, values3)

plot_center = [0.0, 0.5]
plot_radius = [0.5, 0.5]
plot_radius = 1.2 * ones(2)

xrange, yrange = plot_lims(plot_center, plot_radius)

# Plot
z(x, y) = f([x, y])
x1 = range(xrange[1], xrange[2], length=100)
x2 = range(yrange[1], yrange[2], length=100)
zf = @. z(x1', x2)

marker = :cross
ms = 2.5
msw = 0.25

plt = contour(x1, x2, zf, title="Função de Rosenbrock f(x,y)=(1-x)²+20(y-x²)²", xlims=xrange, ylims=yrange)
plot!(values1[1,:], values1[2,:], label="Gradiente Descendente", marker=:square, markerstrokewidth=msw, markersize=ms)
plot!(values2[1,:], values2[2,:], label="Newton-Raphson", marker=:circle, markerstrokewidth=msw, markersize=ms)
plot!(values3[1,:], values3[2,:], label="BFGS", marker=:diamond, markerstrokewidth=msw, markersize=ms)
scatter!([minimal[1]], [minimal[2]], label="Mínimo local", marker=:xcross, markerstrokewidth=msw, markersize=ms)
display(plt)
readline()

