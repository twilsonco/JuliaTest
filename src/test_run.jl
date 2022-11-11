using Plots
x, y = -π:π/100:π, -π:π/100:π
print(x)
sin_data = [ sin(xi) + sin(yi) for xi=x, yi=y ]
contourf(x, y, sin_data) 