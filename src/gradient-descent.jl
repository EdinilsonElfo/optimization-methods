# ==================================================================================================
# Gradiente Descendente
# ==================================================================================================

# Gradient descent method for optimization
function gradient_descent_with_save(problem::OptimizationProblem; epoch::Integer, tol::Real, gradient::Function)
    step = 0.01
    value = copy(problem.initial_value)
    f(x) = problem.objective(x, problem.data)
    values = []
    push!(values, value)
    for k in 1:epoch
        grad = gradient(problem, value)
        step = line_search(f, value, -grad, grad)
        value -= step * grad
        push!(values, value)

        # Check if the method converged
        if norm(grad) < tol
            println("Gradient descent finished with $(k) iterations.")
            return values
        end
    end
    println("Gradient descent finished with $(epoch) iterations.")
    return values
end

# Gradient descent method for optimization
function gradient_descent_without_save(problem::OptimizationProblem; epoch::Integer, tol::Real, gradient::Function)
    step = 0.01
    value = copy(problem.initial_value)
    f(x) = problem.objective(x, problem.data)
    for k in 1:epoch
        grad = gradient(problem, value)
        step = line_search(f, value, -grad, grad)
        value -= step * grad

        # Check if the method converged
        if norm(grad) < tol
            println("Gradient descent finished with $(k) iterations.")
            return value
        end
    end
    println("Gradient descent finished with $(epoch) iterations.")
    return value
end

function gradient_descent(problem::OptimizationProblem; epoch::Integer = 1000, tol::Real = 1e-8, save::Bool = false, gradient::Function = gradient)
    if save == true
        return gradient_descent_with_save(problem, epoch=epoch, tol=tol, gradient=gradient)
    else
        return gradient_descent_without_save(problem, epoch=epoch, tol=tol, gradient=gradient)
    end
end

