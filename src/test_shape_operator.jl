using LinearAlgebra
using ForwardDiff
using SpecialFunctions

# Define the functions
f_list = [
    (x -> exp(- sum(x.^2)), x -> exp(- sum(x.^2) / 5)),
    (x -> besselj0(norm(x)), x -> bessely0(norm(x)))
]
# Define function names for printing
f_names = ["exp(- sum(x.^2))", "exp(- sum(x.^2) / 5)", "besselj0(norm(x))", "bessely0(norm(x))"]


# Define the gradient and Hessian using automatic differentiation
grad(f) = x -> ForwardDiff.gradient(f, x)
hess(f) = x -> ForwardDiff.hessian(f, x)

function principal_curvatures_and_directions(r, sys)
    # Compute gradient and hessian at r
    grad = sys["grad"]
    hess = sys["hess"]
    grad_rho = grad(r)
    hessian_rho = hess(r)

    # Compute the shape operator
    S = - hessian_rho / norm(grad_rho)

    # Compute the eigenvalues and eigenvectors of the shape operator
    e = eigen(S)

    # Sort indices based on the absolute inner products of eigenvectors with gradient (in ascending order)
    sorted_indices = sortperm(1:3, by = i -> abs(dot(grad_rho, e.vectors[:, i])))

    # Select the two smallest absolute value inner product eigenvalues and their corresponding eigenvectors
    principal_curvatures = e.values[sorted_indices[1:2]]
    principal_directions = e.vectors[:, sorted_indices[1:2]]

    return principal_curvatures, principal_directions
end

# Loop over the list of function pairs
for (i, (f1, f2)) in enumerate(f_list)
    println("Testing functions: ", f_names[2i-1], " and ", f_names[2i], " where x is xyz point")

    # Define the gradients and Hessians for each function
    grad_f1 = grad(f1)
    grad_f2 = grad(f2)

    hess_f1 = hess(f1)
    hess_f2 = hess(f2)

    # Define the systems
    sys_f1 = Dict("grad" => grad_f1, "hess" => hess_f1)
    sys_f2 = Dict("grad" => grad_f2, "hess" => hess_f2)

    # Print out the curvatures at a few sample points
    samples = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
    for s in samples
        # Get function values, gradients and curvatures
        curvatures1, directions1 = principal_curvatures_and_directions(s, sys_f1)
        curvatures2, directions2 = principal_curvatures_and_directions(s, sys_f2)

        println("At point ", s)
        println("In first field, function value: ", f1(s), ", principal curvatures: ", curvatures1)
        println("In second field, function value: ", f2(s), ", principal curvatures: ", curvatures2)
    end
    println()
end