using DifferentialEquations
using LinearAlgebra

# showing an example of a DiscreteCallback.
# This could be used for checking GP intersections with CPs.

function custom_callback_function(integrator, data)
    # Access the data passed to the callback
    # Perform some operation using the data (e.g., print it)
    println("Data in the callback: ", data, integrator.u)
end

function create_custom_callback(data)
    function affect!(integrator)
        custom_callback_function(integrator, data)
    end
    return affect!
end

function simple_ode!(du, u, p, t)
    du[1] = -u[1]
end

u0 = [1.0]
tspan = (0.0, 1.0)
ode_prob = ODEProblem(simple_ode!, u0, tspan)

my_data = "Hello from the custom callback!"
affect! = create_custom_callback(my_data)

# Create a ContinuousCallback using the custom affect! function
# Here, the condition function is set to trigger the callback when the solution is close to 0.5
condition(u, t, integrator) = abs(u[1] - 0.5) < 0.1
custom_callback = DiscreteCallback(condition, affect!)

# Set up the solver with tstops
# tstops = range(tspan[1], tspan[2], length=101)[2:end]
# sol = solve(ode_prob, Tsit5(), callback=custom_callback, tstops=tstops)

sol = solve(ode_prob, Tsit5(), callback=custom_callback, dtmax=0.01)
