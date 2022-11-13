

using Parsers, PeriodicTable, PlotlyJS, SplitApplyCombine, LoggingFormats, LoggingExtras
using Interpolations, NLsolve, LinearAlgebra, DifferentialEquations


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
    atoms = [Dict("data" => elements[Int(atoms_raw[1,i])], 
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
    # out["rho(x,y,z)"] = scale(interpolate(out["rho_data"], BSpline(Quadratic(Flat(OnCell())))), g[1], g[2], g[3])
    out["rho(x,y,z)"] = scale(interpolate(out["rho_data"], BSpline(Cubic(Flat(OnCell())))), g[1], g[2], g[3])
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
    cl = Float32(spacing)
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
            if min([LinearAlgebra.norm(x1-a["r"]) for a in sys["atoms"]]...) > 4
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
                    min_dist = min(check_cps...)
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
path_length(path) = sum([LinearAlgebra.norm(i-j) for (i,j) in zip(path[1:end-1], path[2:end])])
minimumby(f, itr) = itr[argmin(map(f, itr))]
function create_gradient_path(sys, start_pt, direction, cutoff)
    padding = sys["lv"] * (ones(3) .* sys["fIJK"] .* 0.05)
    LOW = sys["o"] + padding
    HIGH = sys["o"] + sys["lv"] * sys["fIJK"] - padding
    tspan = (0.0, 1e10)
    u0 = start_pt
    f!(F, r) = sys["grad!"](F, r)
    function fb!(F, x, p, t)
        (1 in (x .< LOW) ? f!(F, LOW) : (1 in (x .> HIGH) ? f!(F, HIGH) : f!(F, x)))
        F .*= direction
    end
    prob = ODEProblem(fb!,u0,tspan)
    sol = solve(prob, AutoVern9(KenCarp5()))
    bp = sol[:]
    end_cp = -1
    for i=3:lastindex(bp[:,1])
        # min_dist_atom = minimumby(a->a[3], [[j, sys["atoms"][j]["r"], LinearAlgebra.norm(sys["atoms"][j]["r"] - bp[i])] for j in 1:lastindex(sys["atoms"])])
        # if min_dist_atom[3] < 0.05
        #     bp = vcat(bp[1:i-1,:], [min_dist_atom[2]])
        #     end_cp = min_dist_atom[1]
        #     @debug "Snapping to atom @ $(min_dist_atom[2])" bp
        #     break
        # else
        cps = [cp for cp in sys["critical_points"] if abs(cp["rank"]) == 3]
        min_dist_cp = minimumby(a->a[3], [[j, cps[j]["r"], LinearAlgebra.norm(cps[j]["r"] - bp[i])] for j in 1:lastindex(cps)])
        if min_dist_cp[3] < 0.5
            bp = vcat(bp[1:i-1,:], [min_dist_cp[2]])
            end_cp = min_dist_cp[1]
            @debug "Snapping to terminal cp @ $(min_dist_cp[2])" bp
            break
        end
        # end
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
        f_sign = (cp["rank"] > 0 ? -1 : 1)
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
    [push!(bps, bp) for th=bps_threads for bp=th]
    return bps
end
find_bond_paths(sys) = find_saddle_paths(sys, -1)
find_ring_lines(sys) = find_saddle_paths(sys, 1)

function plot_results(sys)
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
        # colorscale=colors.RdBu_3,
        opacity=0.5,
        isomin=0.25,
        isomax=0.25,
        surface_count=0,
        caps=attr(x_show=false, y_show=false),
        name=split(sys["name"], "/")[end],
        # slices_z=attr(show=true, locations=[0]),
        # slices_y=attr(show=true, locations=[0]),
        # slices_x=attr(show=true, locations=[0]),
        )
    ]

    # bond paths
    for (i,bp) in enumerate(vcat(sys["bond_paths"], sys["ring_lines"]))
        # r = invert(bp)
        # r = transpose(bp)
        # c = [ sys["rho"](i) for i in bp ]
        # with_logger(FormatLogger(LoggingFormats.Truncated(500))) do
        #     @info "bp plot" [length(i) for i in [r[1], r[2], r[3], c]]
        # end
        name = (i <= length(sys["bond_paths"]) ? "BP" : "RP") * " $(bp["start_cp"])—$(bp["end_cp"])"
        w = (bp["end_cp"] > 0 && sys["critical_points"][bp["end_cp"]]["rank"] == -3 ? 10 : 0.8)
        push!(traces, PlotlyJS.scatter(x=bp["r"][:,1], y=bp["r"][:,2], z=bp["r"][:,3],
                    line=attr(color=log.(sys["rho"].(eachrow(bp["r"]))), width=w),
                    type = "scatter3d",
                    mode = "lines",
                    name=name))
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

    cps = [a for a in sys["critical_points"] if a["rank"] == 3]
    if ! isempty(cps)
        r = invert([a["r"] for a in cps]) # only bond/ring/cage cps
        push!(traces, PlotlyJS.scatter(x=r[1], y=r[2], z=r[3], mode="markers", 
        type="scatter3d", legend=false,
        marker=attr(color=[a["rank"] for a in cps], line=attr(color="black", width=3)),
        name="Cage CP")) 
    end

    layout = PlotlyJS.Layout(autosize=false, width=1000, height=900,
                        margin=attr(l=0, r=0, b=0, t=65))

    PlotlyJS.plot(traces, layout)
end

# fname = "/Users/haiiro/SynologyDrive/Projects/Julia/JuliaTest/DrWatson Example/data/sims/adamantane-fine.cub" 
fname = "/Users/haiiro/SynologyDrive/Projects/Julia/JuliaTest/DrWatson Example/data/sims/buckyball-water.cub" 

sys = import_cub(fname)
# r = [a["r"] for a in sys["atoms"] if a["data"].name == "Carbon"][1]
# x = -1:0.01:1
# rho = [sys["rho(x,y,z)"](r[1] + dx, r[2], r[3]) for dx in x]
# PlotlyJS.plot(PlotlyJS.scatter(x=x,y=rho))
sys["critical_points"] = find_cps(sys, 0.8)
sys["bond_paths"] = find_bond_paths(sys)
sys["ring_lines"] = find_ring_lines(sys)

plot_results(sys)