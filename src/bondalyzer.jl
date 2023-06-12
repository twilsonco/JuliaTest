# import Pkg
# Pkg.add("ModularIndices")
# return

using Parsers, PeriodicTable, PlotlyJS, SplitApplyCombine, LoggingFormats, LoggingExtras, ModularIndices, Random, Distributions
using Interpolations, NLsolve, LinearAlgebra, AngleBetweenVectors, DifferentialEquations, Rotations, Optim, Meshes, QuadGK, ProgressMeter, StatsBase
using PyFormattedStrings

atom_colors = ("#FFFFFF","#D9FFFF","#CC80FF","#C2FF00","#FFB5B5","#909090","#3050F8",
               "#FF0D0D","#90E050","#B3E3F5","#AB5CF2","#8AFF00","#BFA6A6","#F0C8A0",
               "#FF8000","#FFFF30","#1FF01F","#80D1E3","#8F40D4","#3DFF00","#E6E6E6",
               "#BFC2C7","#A6A6AB","#8A99C7","#9C7AC7","#E06633","#F090A0","#50D050",
               "#C88033","#7D80B0","#C28F8F","#668F8F","#BD80E3","#FFA100","#A62929",
               "#5CB8D1","#702EB0","#00FF00","#94FFFF","#94E0E0","#73C2C9","#54B5B5",
               "#3B9E9E","#248F8F","#0A7D8C","#006985","#C0C0C0","#FFD98F","#A67573",
               "#668080","#9E63B5","#D47A00","#940094","#429EB0","#57178F","#00C900",
               "#70D4FF","#FFFFC7","#D9FFC7","#C7FFC7","#A3FFC7","#8FFFC7","#61FFC7",
               "#45FFC7","#30FFC7","#1FFFC7","#00FF9C","#00E675","#00D452","#00BF38",
               "#00AB24","#4DC2FF","#4DA6FF","#2194D6","#267DAB","#266696","#175487",
               "#D0D0E0","#FFD123","#B8B8D0","#A6544D","#575961","#9E4FB5","#AB5C00",
               "#754F45","#428296","#420066","#007D00","#70ABFA","#00BAFF","#00A1FF",
               "#008FFF","#0080FF","#006BFF","#545CF2","#785CE3","#8A4FE3","#A136D4",
               "#B31FD4","#B31FBA","#B30DA6","#BD0D87","#C70066","#CC0059","#D1004F",
               "#D90045","#E00038","#E6002E","#EB0026","#FF00A0","#FF00A0","#FF00A0",
               "#FF00A0","#FF00A0","#FF00A0","#FF00A0","#FF00A0","#FF00A0")
function import_cub(fname)
    @info "Importing data from $fname"
    f = open(fname)
    title, var, natoms_origin = [readline(f) for i=1:3]
    io = IOBuffer(natoms_origin)
    opts = Parsers.Options(delim=" ", ignorerepeated=true)

    num_atoms = Parsers.parse(Int, io, opts)
    origin = [Parsers.parse(Float32, io, opts) for i=1:3]

    text_data = join(split(read(f, String), "\n"), " ")
    close(f)
    io = IOBuffer(text_data)

    npts_lattice = [Parsers.parse(Float32, io, opts) for i=1:4, ln=1:3]
    nIJK = Int.(Tuple(npts_lattice[1,:]))
    fIJK = npts_lattice[1,:]
    lattice = transpose(npts_lattice[2:end,:])

    atoms_raw = [Parsers.parse(Float32, io, opts) for i=1:5, ln=1:num_atoms]
    atoms = [Dict("data" => PeriodicTable.elements[Int(atoms_raw[1,i])], 
                  "color" => atom_colors[Int(atoms_raw[1,i])],
                  "r" => atoms_raw[3:5,i])
                for i=1:num_atoms]

    data = Array{Float32}(undef, nIJK)
    for i=1:nIJK[1], j=1:nIJK[2], k=1:nIJK[3]
        data[i,j,k] = Parsers.parse(Float32, io, opts)
    end
    spacing = [LinearAlgebra.norm(lattice[i,:]) for i=1:3]
    extent = [spacing[i] .* fIJK[i] for i=1:3]
    g = [(origin[i] : spacing[i] : origin[i] + extent[i])[1:end-1] for i=1:3]
    out = Dict("name" => fname, 
        "title" => "$title ($var)",
        "o" => origin,
        "lv" => lattice,
        "extent" => extent,
        "spacing" => spacing,
        "grid" => g,
        "IJK" => nIJK,
        "fIJK" => fIJK,
        "atoms" => atoms,
        "rho_data" => data)
    out["rho(x,y,z)"] = Interpolations.scale(extrapolate(interpolate(out["rho_data"], BSpline(Quadratic(Interpolations.Flat(OnCell())))), Interpolations.Flat()), g[1], g[2], g[3])
    out["rho"] = (r) -> out["rho(x,y,z)"](r[1], r[2], r[3])
    out["rho!"] = (F, r) -> (F = out["rho"](r))
    out["grad!"] = (F, r) -> Interpolations.gradient!(F, out["rho(x,y,z)"], r[1], r[2], r[3])
    out["grad"] = (r) -> (F = zeros(3); out["grad!"](F, r); F)
    out["hess!"] = (J, r) -> Interpolations.hessian!(J, out["rho(x,y,z)"], r[1], r[2], r[3])
    out["hess"] = (r) -> (J = zeros(3,3); out["hess!"](J, r); J)

    @info "System: $(length(out["atoms"])) atoms, $(join(out["IJK"], " × ")) = $(prod(out["IJK"])) points"
    return out
end

function find_cps(sys, spacing)
    # setup bounding box condition for each root search
    # (so they stop if they leave their cell)
    LOW = sys["o"]
    HIGH = sys["o"] + sys["lv"] * sys["fIJK"]
    # define critical point search grid
    X, Y, Z = [LOW[i] + 2spacing : spacing : HIGH[i] - 2spacing for i=1:3]
    cl = Float32(spacing * √2/2)
    @info "CP search using $(prod(length.([X,Y,Z]))) cells with spacing of $cl"
    results = [ [] for i=1:Threads.nthreads() ] 
    # perform grid-based CP search in shared memory parallel
    Threads.@threads for xi=X
        for yi=Y, zi=Z
            x = [xi,yi,zi]
            low = max.(x .- cl, LOW)
            high = min.(x .+ cl, HIGH)
            fb!(F, x) = (1 in (x .<= low) ? 
                         sys["grad!"](F, low) : 
                         (1 in (x .>= high) ? 
                            sys["grad!"](F, high) : 
                            sys["grad!"](F, x)))
            jb!(F, x) = (1 in (x .<= low) ? 
                         sys["hess!"](F, low) : 
                         (1 in (x .>= high) 
                            ? sys["hess!"](F, high) 
                            : sys["hess!"](F, x)))
            # run root finder in cell
            r = nlsolve(fb!, jb!, x)
            if r.x_converged || r.f_converged
                # save if converged
                push!(results[Threads.threadid()],r)
            end
        end
    end


    # create critical point data structure
    cp_info = []
    pushfirst!(results, sys["atoms"])
    for (ri,ro) in enumerate(results)
        for r in ro
            x1 = (ri > 1 ? r.zero : r["r"])
            if minimum([LinearAlgebra.norm(x1-a["r"]) for a in sys["atoms"]]) > 4
                continue # skip points too far from any known atoms
            end
            ρ = sys["rho"](x1)
            A = sys["hess"](x1)
            λ = eigvals(A)
            ε = eigvecs(A)
            rank = sum(sign.(λ))
            if length(cp_info) > 0 
                check_cps = [LinearAlgebra.norm(x1-cp["r"]) for 
                             cp in cp_info if 
                                cp["rank"] >= rank && 
                                sys["rho"](cp["r"]) > ρ]
                if ! isempty(check_cps)
                    min_dist = minimum(check_cps)
                    if min_dist < 0.05
                        continue # skip points too close to another CP of the same rank
                    end
                end
            end
            push!(cp_info, Dict("rank" => rank, 
                                "ε" => ε, 
                                "λ" => λ, 
                                "r" => x1,
                                "root" => (ri > 1 ? r : "")))
        end
    end
    max_rho = max([sys["rho"](cp["r"]) for cp in cp_info]...) + 1
    sort!(cp_info; by= cp->cp["rank"] + sys["rho"](cp["r"]) / max_rho)
    for i in 1:lastindex(cp_info)
        cp_info[i]["index"] = i
    end
    @info "$(length(cp_info)) CPs found"
    for rnk in [-3,-1,1,3]
        cps = [cp for cp in cp_info if cp["rank"] == rnk]
        @info "$(length(cps)) cps of rank $(rnk)"
    end
    # with_logger(FormatLogger(LoggingFormats.Truncated(200))) do
    #     for cp in cp_info
    #         r = cp["r"]
    #         ρ = sys["rho"](r)
    #         λ = cp["λ"]
    #         @debug "Rank $(cp["rank"]): CP #$(cp["index"])" r ρ λ
    #     end
    # end
    return cp_info
end

# bond paths and ring lines
path_length(path) = sum((LinearAlgebra.norm(i-j) for (i,j) in zip(path[1:end-1], path[2:end])))
minimumby(f, itr) = itr[argmin(map(f, itr))]
function create_gradient_path(sys, start_pt, direction, cutoff)
    padding = sys["lv"] * (ones(3) .* sys["fIJK"] .* 0.05)
    LOW = sys["o"] + padding
    HIGH = sys["o"] + sys["lv"] * sys["fIJK"] - padding
    tspan = (0.0, 1e5)
    u0 = start_pt
    f!(F, r) = sys["grad!"](F, r)
    function fb!(F, x, p, t)
        (1 in (x .< LOW) ? f!(F, LOW) : (1 in (x .> HIGH) ? f!(F, HIGH) : f!(F, x)))
        F .*= direction
    end
    prob = ODEProblem(fb!,u0,tspan)
    sol = solve(prob, AutoVern9(KenCarp5()), reltol=1e-8, abstol=1e-8, dtmax=1)
    bp = sol[:]
    end_cp = -1
    for i=3:lastindex(bp[:,1])
        cps = [cp for cp in sys["critical_points"] if abs(cp["rank"]) == 3]
        min_dist_cp = minimumby(a->a[3], [[j, cps[j]["r"], LinearAlgebra.norm(cps[j]["r"] - bp[i])] for j in 1:lastindex(cps)])
        if min_dist_cp[3] < 0.05
            bp = vcat(bp[1:i-1,:], [min_dist_cp[2]])
            end_cp = min_dist_cp[1]
            @debug "Snapping to terminal cp @ $(min_dist_cp[2])" bp
            break
        end
        last_step = (i < length(bp) - 10 ? LinearAlgebra.norm(bp[i]-bp[i+5]) : 1)
        ρ = sys["rho"](bp[i])
        if sum(LOW .<= bp[i] .<= HIGH) ≠ 3 ||
                (last_step < 0.00001) || ρ < cutoff
            bp = bp[1:i,:]
            @debug "Truncating" i last_step ρ bp
            break
        end
    end
    out = Dict("r" => mapreduce(permutedims, vcat, bp),
               "start_cp" => -1,
               "end_cp" => end_cp)
    return out
end
function find_saddle_paths(sys, rank, cutoff=0.001)
    saddles = [ cp for cp in sys["critical_points"] if cp["rank"] == rank ]
    padding = sys["lv"] * (ones(3) .* sys["fIJK"] .* 0.05)
    @info "Finding $(length(saddles)) paths from rank $rank CPs with padding $padding"
    offset = 0.1 * ones(3)
    bps_threads = [ [] for i=1:Threads.nthreads() ]
    Threads.@threads for cp in sys["critical_points"]
    # for cp in sys["critical_points"]
        if cp["rank"] ≠ rank
            continue
        end
        f_sign = -rank
        for i in [-1,1]
            u0 = copy(cp["r"])
            u0 .+= cp["ε"][:,(cp["rank"] > 0 ? 1 : 3)] .* offset * i
            gp = create_gradient_path(sys, u0, f_sign, cutoff)
            gp["r"] = vcat(transpose(cp["r"]), gp["r"])
            gp["start_cp"] = cp["index"]
            push!(bps_threads[Threads.threadid()], gp)
        end
    end
    bps = []
    [push!(bps, bp) for th in bps_threads for bp in th]
    return bps
end
find_bond_paths(sys) = find_saddle_paths(sys, -1)
find_ring_lines(sys) = find_saddle_paths(sys, 1)


function find_saddle_surface(sys, start_cp, num_start_gps, seed_offset, cutoff)
    if start_cp > length(sys["critical_points"]) ||
        abs(sys["critical_points"][start_cp]["rank"]) != 1
        return false
    end
    cp = sys["critical_points"][start_cp]
    pd = cp["ε"][(cp["rank"] > 0 ? 1 : 3),:] # principal direction
    dα = 2π / num_start_gps # step angle
    # vector to be rotated about saddle point to get seed points
    p = LinearAlgebra.normalize(cp["ε"][2,:]) .* seed_offset
    f_sign = cp["rank"]
    gps_threads = [ [] for i in 1:Threads.nthreads() ]
    Threads.@threads for gpi in 0:num_start_gps-1
        α = dα * gpi
        r = AngleAxis(α, pd[1], pd[2], pd[3]) # rot matrix
        q = QuatRotation(r)
        s = q * p # rotated vector
        s += cp["r"]
        s = Array(s)
        @debug "creating gp" cp["r"] α s 
        gp = create_gradient_path(sys, s, f_sign, cutoff)
        gp["r"] = vcat(transpose(cp["r"]), gp["r"])
        gp["start_cp"] = cp["index"]
        push!(gps_threads[Threads.threadid()], gp)
    end
    gps = []
    [push!(gps, gp) for th=gps_threads for gp=th]
    return gps
end
function find_saddle_surfaces(sys, rank, cutoff = 0.001)
    num_start_gps = 15
    seed_offset = 0.01
    cps = [ (i,cp) for (i,cp) in enumerate(sys["critical_points"]) if cp["rank"] == rank ]
    @info "Finding $(length(cps)) surfaces from rank $rank CPs"
    surfs_threads = [ [] for i=1:Threads.nthreads() ]
    Threads.@threads for (i,cp) in cps
        surf = find_saddle_surface(sys, i, num_start_gps, seed_offset, cutoff)
        push!(surfs_threads[Threads.threadid()], surf)
    end
    surfs = []
    [push!(surfs, surf) for th in surfs_threads for surf in th]
    return surfs
end

function plot_results(sys; extra_paths=[], extra_points=[], extra_gps=[])
    @info "Plotting results"
    # prepare data for isosurface and slice plot
    data = [range(sys["o"][i] + 1, stop=sys["o"][i] + (sys["lv"] * sys["fIJK"])[i] - 1, length=150) for i=1:3]
    X, Y, Z = mgrid(data[1], data[2], data[3])
    values = sys["rho(x,y,z)"].(X,Y,Z)
    # println(join(data),"\n")
    traces = [ 
        PlotlyJS.isosurface(
        x=X[:],
        y=Y[:],
        z=Z[:],
        value=values[:],
        # colorInterpolations.scale=colors.RdBu_3,
        opacity=0.5,
        isomin=-1,
        isomax=-1,
        surface_count=0,
        caps=attr(x_show=false, y_show=false),
        name=split(sys["name"], "/")[end],
        legend=false,
        slices_z=attr(show=true, locations=[0]),
        # slices_y=attr(show=true, locations=[0]),
        # slices_x=attr(show=true, locations=[0]),
        )
    ]

    # bond paths
    for (i,bp) in enumerate(vcat(sys["bond_paths"], sys["ring_lines"]))
        name = (i <= length(sys["bond_paths"]) ? "BP" : "RP") * " $(bp["start_cp"])—$(bp["end_cp"])"
        w = (bp["end_cp"] > 0 && sys["critical_points"][bp["end_cp"]]["rank"] == -3 ? 10 : 0.8)
        push!(traces, PlotlyJS.scatter(x=bp["r"][:,1], y=bp["r"][:,2], z=bp["r"][:,3],
                    line=attr(color=log.(sys["rho"].(eachrow(bp["r"]))), width=w),
                    type = "scatter3d", legend=false,
                    mode = "lines",
                    name=name))
    end

    # extra gps
    for (i,bp) in enumerate(extra_gps)
        name = "Extra gp $(i)"
        w = 5
        push!(traces, PlotlyJS.scatter(x=bp["r"][:,1], y=bp["r"][:,2], z=bp["r"][:,3],
                    line=attr(color=:red, width=w),
                    type = "scatter3d", legend=false,
                    mode = "lines",
                    name=name))
    end

    # interatomic surfaces
    # for (i,rs) in enumerate(sys["interatomic_surfaces"])
    #     for gp in rs
        #     push!(traces, PlotlyJS.scatter(x=gp["r"][:,1], y=gp["r"][:,2], z=gp["r"][:,3],
        #                 line=attr(color="red", width=2),
        #                 type = "scatter3d", legend=false,
        #                 mode = "lines"))
    #     end
    # end

    # ring surfaces
    C = ["black","red","blue","green","yellow","orange"]
    for (i,rs) in enumerate(sys["ring_surfaces"])
        for (j,gp) in enumerate(rs)
            push!(traces, PlotlyJS.scatter(x=gp["r"][:,1], y=gp["r"][:,2], z=gp["r"][:,3],
                        # line=attr(color=C[Mod(j)], width=2),
                        line=attr(color="green", width=2),
                        type = "scatter3d", legend=false,
                        mode = "lines"))
        end
        break
    end

    # nuclear coordinates
    r = invert([a["r"] for a in sys["atoms"]])
    push!(traces, PlotlyJS.scatter(x=r[1], y=r[2], z=r[3], 
                mode="markers", type="scatter3d", legend=false,
                marker=attr(color=[a["color"] for a in sys["atoms"]],
                    size=[20 + min(a["data"].number * 3, 100) for a in sys["atoms"]]), 
                line=attr(color="black", width=3),
                name="Atom coord.")
                )

    saddles = [ cp for cp in sys["critical_points"] if abs(cp["rank"]) == 1 ]
    if ! isempty(saddles)
        r = invert([ cp["r"] for cp in saddles ])
        # ε = invert([ 
        #         LinearAlgebra.normalize(cp["ε"][:,(cp["rank"] > 0 ? 1 : 3)]) .* 
        #         abs(cp["λ"][(cp["rank"] > 0 ? 1 : 3)]) .*
        #         (cp["rank"] > 0 ? 20 : 0.8) for cp in saddles 
        #     ])
        # push!(traces, PlotlyJS.cone(x=vcat(r[1], r[1]), y=vcat(r[2], r[2]), z=vcat(r[3], r[3]),
        #                     u=vcat(ε[1], -ε[1]), v=vcat(ε[2], -ε[2]), w=vcat(ε[3], -ε[3]),
        #                     anchor="tip", sizemode="absolute", sizeref=0.075, legend=false,
        #                     name="Saddle CP"))
        push!(traces, PlotlyJS.scatter(x=r[1], y=r[2], z=r[3], mode="markers", 
                                type="scatter3d", legend=false,
                                marker=attr(color=[a["rank"] for a in saddles], line=attr(color="black", width=3)),
                                name="Saddle CP")) 
    end

    # cps = [a for a in sys["critical_points"] if a["rank"] == 3]
    # if ! isempty(cps)
    #     r = invert([a["r"] for a in cps]) # only cage cps
    #     push!(traces, PlotlyJS.scatter(x=r[1], y=r[2], z=r[3], mode="markers", 
    #     type="scatter3d", legend=false,
    #     marker=attr(color=[a["rank"] for a in cps], line=attr(color="black", width=3)),
    #     name="Cage CP")) 
    # end

    # extra paths

    for (i,bp) in enumerate(extra_paths)
        name = "Extra path $i"
        w = 5
        push!(traces, PlotlyJS.scatter(x=bp[:,1], y=bp[:,2], z=bp[:,3],
            line=attr(color=log.(sys["rho"].(eachrow(bp))), width=w),
            type = "scatter3d", legend=false,
            mode = "lines",
            name=name))
    end

    # extra points
    if size(extra_points,1) > 0
        println("size of extra points: $(size(extra_points))")
        for r in extra_points
            println(r)
            push!(traces, PlotlyJS.scatter(x=r[1], y=r[2], z=r[3], 
                        mode="markers", type="scatter3d", legend=false,
                        marker=attr(size=20, color="black", line=attr(color="black", width=3)),
                        name="Extra point")
                        )
        end
    end

    layout = PlotlyJS.Layout(autosize=false, width=1000, height=900,
                        margin=attr(l=0, r=0, b=0, t=65))

    PlotlyJS.plot(traces, layout)
end

function main()
    fname = "/Users/haiiro/SynologyDrive/Projects/Julia/JuliaTest/DrWatson Example/data/sims/adamantane-fine.cub" 
    # fname = "/Users/haiiro/SynologyDrive/Projects/Julia/JuliaTest/DrWatson Example/data/sims/buckyball-water.cub" 

    sys = import_cub(fname)
    sys["critical_points"] = find_cps(sys, 0.8)
    sys["bond_paths"] = find_bond_paths(sys)
    sys["ring_lines"] = find_ring_lines(sys)
    # sys["interatomic_surfaces"] = find_saddle_surfaces(sys, -1)
    sys["ring_surfaces"] = find_saddle_surfaces(sys, 1)

    plot_results(sys)

    return sys
end

# Define a custom interpolation type
# struct ParametrizedGradientPath{T, IT<:AbstractInterpolation{T, 1}, GT<:AbstractArray{T, 1}}
#     x_interpolation::IT
#     y_interpolation::IT
#     z_interpolation::IT
#     grid::GT
# end
struct ParametrizedGradientPath{T, IT<:AbstractInterpolation{T, 1}}
    x_interpolation::IT
    y_interpolation::IT
    z_interpolation::IT
    fmin::T
    fmax::T
end

Base.getindex(interp::ParametrizedGradientPath, y::Number) = [
    interp.x_interpolation(y),
    interp.y_interpolation(y),
    interp.z_interpolation(y)
]

function create_y_to_xyz_interpolation(xyz_positions, y_values; interp_type=Interpolations.Quadratic)
    sorted_indices = sortperm(y_values)
    sorted_y = (y_values[sorted_indices],)
    sorted_xyz = xyz_positions[sorted_indices, :]
    minmax = extrema(y_values)

    if interp_type == Interpolations.Linear
        x_interpolation = extrapolate(interpolate(sorted_y, sorted_xyz[:, 1], Gridded(Linear())), Interpolations.Flat())
        y_interpolation = extrapolate(interpolate(sorted_y, sorted_xyz[:, 2], Gridded(Linear())), Interpolations.Flat())
        z_interpolation = extrapolate(interpolate(sorted_y, sorted_xyz[:, 3], Gridded(Linear())), Interpolations.Flat())
        return ParametrizedGradientPath(x_interpolation, y_interpolation, z_interpolation, minmax[1], minmax[2])
    else
        y_step = (minmax[2] - minmax[1]) / length(y_values)
        y_range = minmax[1]:y_step:minmax[2]
        regular_y_points = collect(y_range)

        gridded_interpolant = interpolate(sorted_y, sorted_xyz[:, 1], Gridded(Linear()))
        temp_vals = gridded_interpolant.(regular_y_points)
        x_interpolation = extrapolate(Interpolations.scale(interpolate(temp_vals, BSpline(interp_type(Interpolations.Line(OnGrid())))), y_range), Interpolations.Flat())

        gridded_interpolant = interpolate(sorted_y, sorted_xyz[:, 2], Gridded(Linear()))
        temp_vals = gridded_interpolant.(regular_y_points)
        y_interpolation = extrapolate(Interpolations.scale(interpolate(temp_vals, BSpline(interp_type(Interpolations.Line(OnGrid())))), y_range), Interpolations.Flat())

        gridded_interpolant = interpolate(sorted_y, sorted_xyz[:, 3], Gridded(Linear()))
        temp_vals = gridded_interpolant.(regular_y_points)
        z_interpolation = extrapolate(Interpolations.scale(interpolate(temp_vals, BSpline(interp_type(Interpolations.Line(OnGrid())))), y_range), Interpolations.Flat())
        return ParametrizedGradientPath(x_interpolation, y_interpolation, z_interpolation, minmax[1], minmax[2])
    end

end

function gp_parametrize(path; f = "length", interp_type=Interpolations.Quadratic)
    x = path["r"]
    if f == "length"
        g(i) = i > 1 ? sum([LinearAlgebra.norm(path["r"][j, :] .- path["r"][j-1, :]) for j in 2:i]) : 0.0
        y = [g(i) for i in 1:size(path["r"], 1)]
    else
        y = [f(r) for r in eachrow(path["r"])]
    end
    return create_y_to_xyz_interpolation(x, y, interp_type=interp_type)
    # return interpolate(rho_vals, path["r"], Gridded(Linear()))
end

function gp_get_deviation(path1::ParametrizedGradientPath, path2::ParametrizedGradientPath, tol::Float64)
    f(x) = norm(path1[x] - path2[x])
    if f(maximum([path1.fmax, path2.fmax])) < 2tol
        return (false, 0.0, 0.0, zeros(3))
    end

    min, max = extrema([path1.fmin, path1.fmax, path2.fmin, path2.fmax])
    # println([path1.fmin, path1.fmax, path2.fmin, path2.fmax])
    step = (max - min) / 100
    for x in min:step:max
        # println("x: $x, f(x): $(f(x)), tol: $tol, path1[x] $(path1[x]), path2[x] $(path2[x])")
        if f(x) > tol
            return (true, x, f(x), (path1[x] .+ path2[x]) ./ 2)
        end
    end

    return (false, 0.0, 0.0, zeros(3))
end

# function gradient_path_saddle_surface_constrained(seed_point, surface_normal_vector, rho, grad, direction=1)
#     # This type of gradient path will, at each step, project the solution onto a saddle surface.
#     # The surface is identified by taking the current point, calculating a vector normal to the gradient,
#     # then rotating that vector abount the gradient vector in order to identify the surface normal direction.
#     # This type of path is for finding paths back to the critical point at the "center" of a saddle surface.

#     function surface_constraint_callback(integrator, surface_normal_vector, rho, grad)
#         # When first seeded, the surface normal direction is provided.
#         # Then it is updated each iteration

#     end

#     function create_custom_callback(data)
#         function affect!(integrator)
#             surface_constraint_callback(integrator, data)
#         end
#         return affect!
#     end
# end

function gradient_path_dimer(seed_points, grad_in; direction=1.0)
    grad(x) = grad_in(x) .* direction
    function ode_func(du, u, p, t)
        du .= grad(u)
    end

    dimer_points = copy(seed_points)
    println("Starting dimer repositioning between points $(dimer_points[1]) and $(dimer_points[2])")
    function get_gradient_midpoint()
        # Given two points at which the gradient is deviating (pointing away from each other),
        # then find the point between them where the gradient points in the average direction of
        # that of the two points.
        
        # First define the angle difference between the the gradient at each point and at the 
        # intermediate point and returns the absolute difference of these differences.
        # When this function is zero, then x corresponds to the actual "midpoint" between the
        # two points according to the gradient.
        # This point then lies along a deviation path in the gradient.
        xvec = dimer_points[2] .- dimer_points[1]
        # print("$(LinearAlgebra.norm(xvec)), ")
        function f(x)
            pt = dimer_points[1] .+ x[1] .* xvec
            a1 = angle(grad(pt), grad(dimer_points[1]))
            a2 = angle(grad(pt), grad(dimer_points[2]))
            out = abs(a1 - a2)
            # print("$out at $(x[1]), ")
            out
        end
        res = optimize(f, [0.5])
        x = Optim.minimizer(res)[1]
        xpt = dimer_points[1] .+ xvec * x
        
        # println("grad midpoint at $x which is $(xpt)")

        # reposition the dimer points so the dimer orientation is aligned with the gradient
        rot_axis = cross(xvec, grad(xpt))
        grad_normal = norm(cross(grad(xpt), rot_axis))
        xvec_half_norm = norm(xvec) / 2
        new_points = [xpt .+ grad_normal .* xvec_half_norm, xpt .- grad_normal .* xvec_half_norm]
        if norm(dimer_points[1] - new_points[2]) < norm(dimer_points[1] - new_points[2])
            dimer_points = new_points
        else
            dimer_points = [new_points[2], new_points[1]]
        end

        return xpt
    end

    function dimer_reposition_affect!(integrator)
        # Get the current position of the dimer midpoint
        integrator.u = get_gradient_midpoint()
    end

    function deviation_termination_condition(u, t, integrator)
        # Terminate integrator if gradient at dimer points  are pointing in
        # opposite directions
        return abs(angle(grad(dimer_points[1]), grad(dimer_points[2]))) > 0.95π
    end

    function detect_bifurcation(point1, point2)
        gradient1 = grad(point1)
        gradient2 = grad(point2)
    
        # Calculate the angle between the gradients
        angle = acos(dot(gradient1, gradient2) / (norm(gradient1) * norm(gradient2)))
    
        # Check if the gradients are pointing towards each other (angle between 90 and 270 degrees)
        bifurcation_detected = (π/2 <= angle) && (angle <= 3π/2)
    
        return bifurcation_detected
    end

    u0 = (seed_points[1] + seed_points[2]) / 2
    dimer_callback = DiscreteCallback((u, t, integrator) -> true, dimer_reposition_affect!)
    termination_callback = ContinuousCallback(deviation_termination_condition, (integrator) -> terminate!(integrator))
    callbacks = CallbackSet(dimer_callback, termination_callback)

    tspan = (0.0, 1e5)
    prob = ODEProblem(ode_func, u0, tspan)
    sol = solve(prob, Tsit5(), callback=callbacks)
    return hcat(sol[:]...)
end

function test()
    f(x) = (x[1]+1.3)^2+0.02
    return optimize(f, [0.0])
end

function test_gp_dimer(sys)
    paths = []
    points = []
    for i in 1:size(sys["ring_surfaces"][1], 1)
        ip1 = i+1
        if ip1 > size(sys["ring_surfaces"][1], 1)
            ip1 = 1
        end
        path1=gp_parametrize(sys["ring_surfaces"][1][i],x->log(sys["rho"](x)))
        path2=gp_parametrize(sys["ring_surfaces"][1][ip1],x->log(sys["rho"](x)))
        # print("fmin points ", path1[path1.fmin], " and ", path2[path2.fmin])
        # println(":  fmax points ", path1[path1.fmax], " and ", path2[path2.fmax])
        # continue
        dev=gp_get_deviation(path1,path2,0.4)
        # println("deviation = ",dev, " between paths ", i, " and ", ip1)
        if dev[1]
            testgp = create_gradient_path(sys, dev[4], 1, 1e-3)
            seed_points = [path1[dev[2]], path2[dev[2]]]
            push!(paths, gradient_path_dimer(seed_points, sys["grad"]))
            push!(points, dev[4])
            # return [testgp["r"], gp', dev]
        end
    end
    return plot_results(sys, extra_paths=paths, extra_points=points)
end

function plot_parameterized_gps(sys)
    plot_traces = GenericTrace[]
    for i in 1:size(sys["ring_surfaces"][1], 1)
        path=gp_parametrize(sys["ring_surfaces"][1][i],x->log(sys["rho"](x)))
        x = path.fmin:0.1:path.fmax
        y = hcat([path[x] for x in x]...)
        # println(y)
        # now make a 3d line plot of the xyz points y as a function of x, with plotlyjs
        push!(plot_traces, PlotlyJS.scatter(x=y[1,:], y=y[2,:], z=y[3,:], type = "scatter3d", legend=false,
            mode = "lines"))
    end
    # now plot the list of traces
    layout = PlotlyJS.Layout(autosize=false, width=1000, height=900,
                        margin=attr(l=0, r=0, b=0, t=65))
    PlotlyJS.plot(plot_traces, layout)
end


function principal_curvatures_and_directions(r, sys)
    # Compute gradient and hessian at r
    grad_rho = sys["grad"](r)
    hessian_rho = sys["hess"](r)

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


function isosurface_mean_curvature_at_point(point, sys)
    iso_k = principal_curvatures_and_directions(point, sys)
    mean_k = (iso_k[1][1] + iso_k[1][2]) / 2
    return mean_k
end


function make_sphere(point, radius, resolution=30)
    # sphere geometry
    sphere = Meshes.Sphere(point, radius)

    # polygonal mesh with quadrangles + triangles
    mesh = Meshes.discretize(sphere, RegularDiscretization(resolution, resolution))
end

function dA(gp, sys, s; a = gp.fmin)
    f(x) = 2 * isosurface_mean_curvature_at_point(gp[x], sys)
    return quadgk(f, a, s)[1]
end

# for precomputing the dA values
function gp_parametrize_dA(parameterized_gp, sys; interp_type=Interpolations.Quadratic)
    path_step = parameterized_gp.fmax / 100
    x = 0:path_step:parameterized_gp.fmax
    y = [dA(parameterized_gp, sys, s) for s in x]
    if interp_type == Interpolations.Linear
        return extrapolate(interpolate(x, y, Gridded(Linear())), Interpolations.Flat())
    else
        return extrapolate(Interpolations.scale(interpolate(y, BSpline(interp_type(Interpolations.Line(OnGrid())))), x), Interpolations.Flat())
    end
    # return interpolate(rho_vals, path["r"], Gridded(Linear()))
end

function gp_parametrize_dA1(parameterized_gp, sys; interp_type=Interpolations.Quadratic)
    path_step = parameterized_gp.fmax / 20
    x = 0:path_step:parameterized_gp.fmax
    y = zeros(length(x))
    for i in 1:length(x)
        prev_sum = y[max(i-1,1)]
        lower_limit = x[max(i-1,1)]
        new_sum_component = dA(parameterized_gp, sys, x[i], a = lower_limit)
        y[i] = prev_sum + new_sum_component
        # check_val = dA(parameterized_gp, sys, x[i])
        # @info "step $i" prev_sum lower_limit new_sum_component y[i] check_val
    end
    if interp_type == Interpolations.Linear
        return extrapolate(interpolate(x, y, Gridded(Linear())), Interpolations.Flat())
    else
        return extrapolate(Interpolations.scale(interpolate(y, BSpline(interp_type(Interpolations.Line(OnGrid())))), x), Interpolations.Flat())
    end
    # return interpolate(rho_vals, path["r"], Gridded(Linear()))
end

function integrate_dgb(gp, sys, a_0, f_list)
    out = zeros(length(f_list))
    gp_dA = gp_parametrize_dA1(gp, sys)
    for i in 1:length(f_list)
        f(x) = gp_dA[x] * f_list[i](gp[x])
        out[i] = quadgk(f, gp.fmin+0.05, gp.fmax)[1] * a_0
    end
    return out
end

function test_isosurface_curvature_for_gp(sys)
    # gp = create_gradient_path(sys, [1; 1; 1], 1, 1e-3)
    # gp1 = create_gradient_path(sys, [1; 1; 1], -1, 1e-3)
    gp = create_gradient_path(sys, [0; 0; 0.5], 1, 1e-3)
    gp1 = create_gradient_path(sys, [0; 0; 0.5], -1, 1e-3)
    println(size(gp["r"]))
    gp["r"] = vcat(reverse(gp["r"], dims=1), gp1["r"][2:end, :])
    gp["start_cp"] = gp1["end_cp"]
    println(size(gp["r"]))
    gp_parametarized = gp_parametrize(gp)
    gp_dA1 = @timed gp_parametrize_dA1(gp_parametarized, sys)
    # gp_dA = @timed gp_parametrize_dA(gp_parametarized, sys)
    # @info "timing " gp_dA[2] gp_dA1[2]
    # gp_dA = gp_dA[1]
    gp_dA1 = gp_dA1[1]
    path_len = gp_parametarized.fmax-gp_parametarized.fmin
    gp_step = path_len/20
    println("Isosurface mean curvatures along a gradient path from atom to cage")
    for i in gp_parametarized.fmin+gp_step/100:gp_step/3:gp_parametarized.fmax/4-gp_step
        iso_k = principal_curvatures_and_directions(gp_parametarized[i], sys)
        mean_k = isosurface_mean_curvature_at_point(gp_parametarized[i], sys)
        rho = sys["rho"](gp_parametarized[i])
        diff = mean_k - 1/i
        # println("rho = $(sys["rho"](gp_parametarized[i])) mean_k = ", mean_k, " at $i (of $path_len)")
        println(f"rho(s) = {rho:0.03F}, H = {mean_k:0.03F} (), 1/s = {1/i:0.03F}, diff = {diff:0.03F},  at s = {i:0.03F} from nuclear CP (of {path_len:0.03F})")
        # println("gp dA = ", gp_dA[i])
        # println("gp dA1 = ", gp_dA1[i])
    end
    println()
    for i in gp_parametarized.fmin+15gp_step:gp_step/3:gp_parametarized.fmax
        iso_k = principal_curvatures_and_directions(gp_parametarized[i], sys)
        mean_k = isosurface_mean_curvature_at_point(gp_parametarized[i], sys)
        rho = sys["rho"](gp_parametarized[i])
        len=gp_parametarized.fmax-i
        diff = -mean_k - 1/len
        # println("rho = $(sys["rho"](gp_parametarized[i])) mean_k = ", mean_k, " at $i (of $path_len)")
        println(f"rho(s) = {rho:0.03F}, H = {mean_k:0.03F} (), 1/s = {1/len:0.03F}, diff = {diff:0.03F},  at s = {len:0.03F} from cage CP (of {path_len:0.03F})")
        # println("gp dA = ", gp_dA[i])
        # println("gp dA1 = ", gp_dA1[i])
    end
    # println(integrate_dgb(gp_parametarized, sys, 1.0, [x->1, sys["rho"]]))
    # plot_results(sys, extra_gps=[gp])
end


function element_midpoints_and_areas(mesh)
    midpoints = []
    areas = []

    for element in 1:nelements(mesh.topology)
        vertices = [mesh.vertices[i] for i in element]

        # Calculate midpoint
        midpoint = sum(vertices) / length(vertices)
        push!(midpoints, midpoint)

        # Calculate area
        if length(vertices) == 3  # Triangle
            v1, v2, v3 = vertices
            area = 0.5 * norm(cross(v2 - v1, v3 - v1))
        elseif length(vertices) == 4  # Quadrangle
            v1, v2, v3, v4 = vertices
            area = 0.5 * (norm(cross(v2 - v1, v3 - v1)) + norm(cross(v3 - v1, v4 - v1)))
        else
            error("Unsupported element type")
        end
        push!(areas, area)
    end

    return midpoints, areas
end

function test_gp_only_gba(sys)
    nuclear_cps = [a for a in sys["atoms"]]
    f_list = [x->1, sys["rho"]]
    f_names = ["V", "ρ"]
    for (ncpi,ncp) in enumerate(nuclear_cps)
        closest_bond_distance = minimum([ norm(cpi["r"] - ncp["r"]) for cpi in sys["critical_points"] if cpi["rank"] == -1 ])
        radius = 0.2closest_bond_distance
        gps_threads = [ [] for i=1:Threads.nthreads() ]
        num_points_list = [24, 128]
        @info "Checking resolutions $num_points_list for atom $(ncp["data"].name) $ncpi at $(ncp["r"])..."
        for num_points in num_points_list
            sphere = points_on_sphere_regular(num_points, ncp["r"], radius)
            num_points = length(sphere)
            elem_area = 4 * pi * radius^2 / num_points
            # loop over sphere points
            int_vals = zeros(length(f_list))
            gp_int_vals = []
            pm = Progress(num_points, "Atom $(ncp["data"].name) $ncpi: Seeding $num_points gradient paths...")
            Threads.@threads for p in sphere
                # find gradient path
                gp = create_gradient_path(sys, p, 1, 1e-3)
                gp1 = create_gradient_path(sys, p, -1, 1e-3)
                gp["r"] = vcat(reverse(gp["r"], dims=1), gp1["r"][2:end, :])
                gp["start_cp"] = gp1["end_cp"]
                # push!(gps_threads[Threads.threadid()], gp)
                # parametrize gradient path
                gp_parametarized = gp_parametrize(gp)
                # integrate
                int_val = integrate_dgb(gp_parametarized, sys, elem_area, f_list)
                push!(gp_int_vals, int_val)
                int_vals .+= int_val
                next!(pm)
            end
            finish!(pm)
            # @info "Atom $(ncp["data"].name) $ncpi with $num_points GPs: int_vals = $int_vals"
            # print stats of each column of gp_int_vals with n, min, max, mean, std
            gp_int_vals = hcat(gp_int_vals...)
            gp_int_vals_stats = [sum(gp_int_vals, dims=2), minimum(gp_int_vals, dims=2), maximum(gp_int_vals, dims=2), mean(gp_int_vals, dims=2), std(gp_int_vals, dims=2)]

            formatted_stats = "Sum: $(gp_int_vals_stats[1]), Min: $(gp_int_vals_stats[2]), Max: $(gp_int_vals_stats[3]), Mean: $(gp_int_vals_stats[4]), Std: $(gp_int_vals_stats[5])"

            @info "Atom $(ncp["data"].name) $ncpi with $num_points GPs: gp_int_vals_stats for functions $(join(f_names, ", ")) = $formatted_stats"


        end
        # gps = []
        # [push!(gps, gp) for th in gps_threads for gp in th]
        # return plot_results(sys, extra_gps=gps)
        # if ncpi >=8
            break
        # end
    end
end

# [ Info: Checking resolutions [24, 128, 1024, 4096] for atom Carbon 1 at Float32[-1.687072, -1.687072, -1.687072]...
# Atom Carbon 1: Seeding 20 gradient paths... 100% Time: 0:02:30
# [ Info: Atom Carbon 1 with 20 GPs: gp_int_vals_stats for functions V, ρ = Sum: [25.90793607722703; 32.81978969400329;;], Min: [-3.452307908272171; 0.943696456824771;;], Max: [3.4602734411514215; 3.413345289860068;;], Mean: [1.2953968038613515; 1.6409894847001645;;], Std: [1.7674442924731328; 0.6265669114411503;;]
# Atom Carbon 1: Seeding 128 gradient paths... 100% Time: 0:16:32
# [ Info: Atom Carbon 1 with 128 GPs: gp_int_vals_stats for functions V, ρ = Sum: [35.13495011305954; 33.773062426097695;;], Min: [-0.32869214073534364; 0.1561714018918065;;], Max: [0.660330759264329; 0.6441946436549573;;], Mean: [0.27449179775827764; 0.26385205020388824;;], Std: [0.1312774516655372; 0.09363891767141092;;]
# Atom Carbon 1: Seeding 998 gradient paths... 100% Time: 2:37:56
# [ Info: Atom Carbon 1 with 998 GPs: gp_int_vals_stats for functions V, ρ = Sum: [34.699092280424075; 33.82115900948734;;], Min: [-0.15722844216165358; 0.01713763184392208;;], Max: [0.12401887860408875; 0.11462884555294456;;], Mean: [0.03476862953950308; 0.033888936883253847;;], Std: [0.019705063640001453; 0.01221671012780067;;]
# Atom Carbon 1: Seeding 4136 gradient paths... 100% Time: 8:57:55
# [ Info: Atom Carbon 1 with 4136 GPs: gp_int_vals_stats for functions V, ρ = Sum: [34.79064604109993; 33.836057358517884;;], Min: [-0.037306127545212876; 0.004261891394502226;;], Max: [0.03333179857562209; 0.029686384174541364;;], Mean: [0.008411664903554142; 0.008180864931943395;;], Std: [0.004618431977415803; 0.002951058361225472;;]

function points_on_sphere_regular(N::Int, center::Vector{Float32}, radius::Float64)
    points = []
    Ncount = 0

    a = 4 * pi / N
    d = sqrt(a)
    M_theta = round(pi / d)
    d_theta = pi / M_theta

    for m in 0:(M_theta - 1)
        theta = pi * (m + 0.5) / M_theta
        M_phi = round(2 * pi * sin(theta) / d_theta)

        for n in 0:(M_phi - 1)
            phi = 2 * n * pi / M_phi
            x = radius * sin(theta) * cos(phi) + center[1]
            y = radius * sin(theta) * sin(phi) + center[2]
            z = radius * cos(theta) + center[3]
            push!(points, [x; y; z])
            Ncount += 1
        end
    end

    return points
end


function plot_regular_points_on_sphere(N::Int, center::Tuple{Float64, Float64, Float64}, radius::Float64)
    points = points_on_sphere_regular(N, center, radius)

    x_coords = [point[1] for point in points]
    y_coords = [point[2] for point in points]
    z_coords = [point[3] for point in points]

    trace = scatter3d(x=x_coords, y=y_coords, z=z_coords,
                      mode="markers",
                      marker=attr(size=5, line=attr(color="rgba(0, 0, 0, 0.5)", width=0.5)))
    
    layout = Layout(scene=attr(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                    title="Regular Points on Sphere")

    plot(trace, layout)
end



function points_on_sphere_random(center::Tuple{Float64, Float64, Float64}, radius::Float64, n_points::Int)
    # Initialize an empty array to store the points
    points = []

    # Generate points on the sphere
    for i in 1:n_points
        theta = 2 * pi * rand()
        phi = acos(1 - 2 * rand())
        x = center[1] + radius * sin(phi) * cos(theta)
        y = center[2] + radius * sin(phi) * sin(theta)
        z = center[3] + radius * cos(phi)
        push!(points, (x, y, z))
        @info "point" theta  phi  x  y  z
    end

    return points
end

# main()
# @profview main()