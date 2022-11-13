# Import required packages
using DifferentialEquations
using BenchmarkTools
using Base.Threads


# Define the simple ODE system
function simple_system!(du, u, p, t)
    du[1] = -u[1] + u[2]*t
    du[2] = -u[2] - u[1]*t
end

u0 = [0.5, 0.5]
tspan = (0.0, 10.0)
prob = ODEProblem(simple_system!, u0, tspan)

# Create a function to generate initial conditions for the ensemble
function prob_func(prob, i, repeat)
    prob.u0 .= 0.5 .+ 0.1 .* rand(2)  # Adjust initial conditions as needed
    prob # = ODEProblem(simple_system!, u0, tspan)
end

# Set up the parallel ensemble simulation
num_sims = 1000  # Specify the number of initial conditions you want to simulate

# Define the function to create the EnsembleProblem
function create_ensemble_problem()
    return EnsembleProblem(prob, prob_func=prob_func)
end

# Define the function to solve the EnsembleProblem
function solve_ensemble_problem(ensemble_prob)
    return solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=num_sims)
end

# Define the function to solve problems without EnsembleProblem
function solve_without_ensemble()
    solutions = Vector{ODESolution}(undef, num_sims)
    @threads for i in 1:num_sims
        prob_i = prob_func(prob, i, false)
        solutions[i] = solve(prob_i, Tsit5())
    end
    return solutions
end

# Create the EnsembleProblem
ensemble_prob = create_ensemble_problem()

# Benchmark the EnsembleProblem creation and solving
println("Benchmarking EnsembleProblem creation:")
@btime create_ensemble_problem()

println("Benchmarking EnsembleProblem solving:")
ensemble_benchmark = @btime solve_ensemble_problem($ensemble_prob)

# Benchmark solving without EnsembleProblem
println("Benchmarking solving without EnsembleProblem:")
no_ensemble_benchmark = @btime solve_without_ensemble()

println("Done")

# Compare benchmark results
# println("EnsembleProblem solving time: ", median(ensemble_benchmark.time), " ns")
# println("Solving without EnsembleProblem time: ", median(no_ensemble_benchmark.time), " ns")
