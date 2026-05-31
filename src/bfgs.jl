# ==================================================================================================
# Método BFGS
# ==================================================================================================

function update_hessian(B::Matrix, y::Vector, s::Vector)
    return B + (y * y') / (y' * s) - (B * s * s' * B) / (s' * B * s)
end

function bfgs_with_save(problem::OptimizationProblem; epoch::Integer, tol::Real, gradient::Function)
    value = copy(problem.initial_value)
    f(x) = problem.objective(x, problem.data)

    grad = gradient(problem, value)
    
    dim = length(grad)
    # hess = hessian(problem, value)
    B_inv = Matrix{Float64}(I, dim, dim)

    values = []
    push!(values, value)

    for k in 1:epoch
        # Guarda o valor antigo do gradiente
        grad_old = copy(grad)

        # Encontra a direção
        # direction = - hess \ grad
        direction = - B_inv * grad

        # Estima o passo
        step = line_search(f, value, direction, grad)

        # Aplica o incremento
        s = step * direction
        value += s

        # Atualiza o gradiente
        grad = gradient(problem, value)

        # Cálcula o y
        y = grad - grad_old

        # Calcula o termo s^T y
        sty = s' * y

        # Verifica a curvatura local e atualiza a B^-1
        if (sty > tol)
            B_inv = B_inv + (sty + y' * B_inv * y) / sty^2 * (s * s') - (B_inv * y * s' + s * y' * B_inv) / sty
        else
            # B_inv = Matrix{Float64}(I, dim, dim)
        end

        push!(values, value)

        # Check if the method converged
        if norm(grad) < tol
            println("Method converged")
            println("BFGS finished with $(k) iterations.")
            return values
        end
    end
    println("BFGS finished with $(epoch) iterations.")
    return values
end

function bfgs_without_save(problem::OptimizationProblem; epoch::Integer, tol::Real, gradient::Function)
    value = copy(problem.initial_value)
    f(x) = problem.objective(x, problem.data)

    grad = gradient(problem, value)

    dim = length(grad)
    # hess = hessian(problem, value)
    B_inv = Matrix{Float64}(I, dim, dim)

    for k in 1:epoch
        # Guarda o valor antigo do gradiente
        grad_old = copy(grad)

        # Encontra a direção
        direction = - B_inv * grad

        # Estima o passo
        step = line_search(f, value, direction, grad)

        # Aplica o incremento
        s = step * direction
        value += s

        # Atualiza o gradiente
        grad = gradient(problem, value)

        # Cálcula o y
        y = grad - grad_old

        # Calcula o termo s^T y
        sty = s' * y

        # Verifica a curvatura local e atualiza a B^-1
        if (sty > tol)
            B_inv = B_inv + (sty + y' * B_inv * y) / sty^2 * (s * s') - (B_inv * y * s' + s * y' * B_inv) / sty
        else
            # B_inv = Matrix{Float64}(I, dim, dim)
        end

        # Check if the method converged
        if norm(grad) < tol
            println("BFGS finished with $(k) iterations.")
            return value
        end
    end
    println("BFGS finished with $(epoch) iterations.")
    return value
end

function bfgs(problem::OptimizationProblem; epoch::Integer = 100, tol::Real = 1e-8, save::Bool = false, gradient::Function = gradient)
    if save == true
        return bfgs_with_save(problem, epoch=epoch, tol=tol, gradient=gradient)
    else
        return bfgs_without_save(problem, epoch=epoch, tol=tol, gradient=gradient)
    end
end

