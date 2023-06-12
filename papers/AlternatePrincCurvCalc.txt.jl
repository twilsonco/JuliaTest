Alternate ways to calculate the isosurface principal curvatures

# function principal_curvatures_and_directions_1(point, sys)
#     # rho = sys["rho"]
#     grad = sys["grad"]
#     hess = sys["hess"]

#     # rho_value = rho(point) # [e/bohr^3]
#     grad_value = grad(point) # [e/bohr^4]
#     n = normalize(grad_value) 
#     H = hess(point) / norm(grad_value) # [e/bohr^5] / [e/bohr^4] = [1/bohr]

#     # Creating the projection matrix.
#     P = I - n * n'

#     # Projecting the Hessian onto the tangent plane and negating it (to indicate that we're interested in curvature inward towards regions of lower electron density)
#     Hp = - P * H * P

#     # Finding the eigenvalues and eigenvectors of the Weingarten matrix.
#     eigen_decomposition = eigen(Hp)

#     # Getting the indices that would sort the eigenvalues by their absolute values in descending order.
#     sorted_indices = sortperm(abs.(eigen_decomposition.values), rev=true)

#     # Selecting the two largest absolute value eigenvalues and their corresponding eigenvectors.
#     principal_curvatures = eigen_decomposition.values[sorted_indices][1:2]
#     principal_directions = eigen_decomposition.vectors[:, sorted_indices[1:2]]

#     return principal_curvatures, principal_directions
# end

# function principal_curvatures_and_directions_2(point, sys)
#     # Get gradient and Hessian
#     grad_value = sys["grad"](point)
#     hess_value = sys["hess"](point)

#     # Normalize the gradient
#     grad_norm = norm(grad_value)

#     if grad_norm == 0
#         return (0, 0), ([0, 0, 0], [0, 0, 0]) # Return zero curvatures and zero vectors at the origin
#     end

#     n = grad_value / grad_norm

#     # Compute the shape operator (3x3 matrix)
#     S = -hess_value / grad_norm

#     # Compute two tangent vectors
#     tangent1 = cross(n, [1, 0, 0])
#     if norm(tangent1) < 1e-6
#         tangent1 = cross(n, [0, 1, 0])
#     end
#     tangent1 /= norm(tangent1)
#     tangent2 = cross(n, tangent1)

#     # Project the shape operator onto the tangent plane
#     S_projected = [dot(tangent1, S*tangent1) dot(tangent1, S*tangent2);
#                    dot(tangent2, S*tangent1) dot(tangent2, S*tangent2)]

#     # Compute the eigenvalues and eigenvectors of the projected shape operator
#     principal_curvatures, principal_directions_projected = eigen(S_projected)

#     # Translate the principal directions back into 3D space
#     principal_directions = [tangent1 * vec[1] + tangent2 * vec[2] for vec in eachcol(principal_directions_projected)]

#     return principal_curvatures, principal_directions
# end