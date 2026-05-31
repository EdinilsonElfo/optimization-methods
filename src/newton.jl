# ==================================================================================================
# Método de Newton
# ==================================================================================================

function newton_with_save(problem::OptimizationProblem; epoch::Integer, tol::Real)
    value = copy(problem.initial_value)
    f(x) = problem.objective(x, problem.data)
    values = []
    push!(values, value)
    for k in 1:epoch
        grad = gradient(problem, value)
        hess = hessian(problem, value)

        direction = - hess \ grad
        step = line_search(f, value, direction, grad)
        value += step * direction

        push!(values, value)

        # Check if the method converged
        if norm(grad) < tol
            println("Newton finished with $(k) iterations.")
            return values
        end
    end
    println("Newton finished with $(epoch) iterations.")
    return values
end

function newton_without_save(problem::OptimizationProblem; epoch::Integer, tol::Real)
    value = copy(problem.initial_value)
    f(x) = problem.objective(x, problem.data)
    for k in 1:epoch
        grad = gradient(problem, value)
        hess = hessian(problem, value)

        direction = - hess \ grad
        step = line_search(f, value, direction, grad)
        value += step * direction

        # Check if the method converged
        if norm(grad) < tol
            println("Newton finished with $(k) iterations.")
            return value
        end
    end
    println("Newton finished with $(epoch) iterations.")
    return value
end

function newton(problem::OptimizationProblem; epoch::Integer = 100, tol::Real = 1e-8, save::Bool = false)
    if save == true
        return newton_with_save(problem, epoch=epoch, tol=tol)
    else
        return newton_without_save(problem, epoch=epoch, tol=tol)
    end
end

