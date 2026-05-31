# ==================================================================================================
# Optimization Problem
# ==================================================================================================

# Struct for Optimization Problem
struct OptimizationProblem
    objective :: Function
    initial_value :: Vector
    data :: Union{Vector, Nothing}
    constraint :: Union{Function, Nothing}
end

OptimizationProblem(objective, initial_value, data) = OptimizationProblem(objective, initial_value, data, nothing)

# Constructor with no data
function OptimizationProblem(objective::Function, initial_value::Vector)
    return OptimizationProblem(objective, initial_value, nothing)
end

# Get the gradient of the objective function
function gradient(problem::OptimizationProblem, x::Vector)
    f(x) = problem.objective(x, problem.data)
    return ForwardDiff.gradient(f, x)
end

# Get the gradient of the objective function
function hessian(problem::OptimizationProblem, x::Vector)
    f(x) = problem.objective(x, problem.data)
    return ForwardDiff.hessian(f, x)
end

# Do the Backtracing Line Search to find the best step
function line_search(f::Function, x::Vector, p::Vector, grad::Vector; step_init::Real=2.0, c::Real = 1e-4)
    step = step_init
    for k in 1:100
        m = grad' * p

        # Verify the Armijo-Goldstein condition
        if ( f(x + step*p) <= f(x) + step*c*m )
            # println(step)
            return step
        end
        step = step * 0.5
    end
    return step
end

