cd("/home/andris/MDrops/")

using Pkg

pkg"activate ."
pkg"resolve"

using JLD2
using StatsBase
using LinearAlgebra
using FastGaussQuadrature
using Optim
using Distributed
#using Makie

include("./SurfaceGeometry/dt20L/src/Iterators.jl")
include("./mesh_functions.jl")
include("./physics_functions.jl")
include("./mathematics_functions.jl")

#
# using LinearAlgebra
# #using CSV
# using JLD2
# #using Makie
# using StatsBase
# using Optim
# using FastGaussQuadrature
#
#
# include("./SurfaceGeometry/dt20L/src/Iterators.jl")
# #include("./stabilization.jl")
# include("./mathematics_functions.jl")
# include("./mesh_functions.jl")
# include("./physics_functions.jl")

#points_csv= CSV.read("./meshes/points_critical_hyst_2_21.csv", header=0)
#faces_csv = CSV.read("./meshes/faces_critical_hyst_2_21.csv", header=0)
#fields = CSV.read("/home/laigars/sim_data/field.csv", header=0)[1] * 10 # mT -> Oe
#times = CSV.read("/home/laigars/sim_data/time.csv", header=0)[1]
#points_csv= CSV.read("./meshes/points_ellipse_fewN.csv", header=0)
#faces_csv = CSV.read("./meshes/faces_ellipse_fewN.csv", header=0)

# points = convert(Array, points_csv)
# faces = convert(Array, faces_csv)
points, faces = expand_icosamesh(R=1, depth=2)

#@load "./meshes/faces_critical_hyst_2_21.jld2" faces
points = Array{Float64}(points)
faces = Array{Int64}(faces)
#points = points

println("Loaded mesh; nodes = $(size(points,2))")

continue_sim = true

dataname = "elongation_Bm5_lamdba10_mu30_adaptiveN_adaptive_dt_uncoupled_parabs_contd5_paral"
datadir = "/home/andris/sim_data/$dataname"

H0 = [0., 0., 1.]
mu = 30.

# Bm_crit = 3.68423 pie mu=30
Bm = 5. ################################################ zemāk iespējams loado citu
#R0 = 21.5 * 100/480 * 1e-4 # um to cm for cgs
R0 = 1.
lambda = 10.
gamma = H0[3]^2 * R0 / Bm
#gamma = 8.2 * 1e-4
#gamma = 7.7 * 1e-4 # from fitted exp data with mu=34
w = 0

reset_vmax = true
last_step = 0
t = 0
dt = 0.05
steps = 12000
epsilon = 0.05
normals = Normals(points, faces)

max_vs = zeros(3, steps)
mean_vs = zeros(3, steps)
max_abs_v = zeros(1, steps)

if continue_sim
    reset_vmax = false

    last_file = readdir(datadir)[end-1]
    global data
    println("continuing simulation from: $datadir/$last_file")
    @load "$datadir/$last_file" data

    global points, faces, t, H0, Bm, v0max = data[1], data[2], data[3], data[4], data[5], data[6]
    global last_step = parse(Int32, last_file[5:9])
    println("last step: $last_step")
    normals = Normals(points, faces)
    cp("main_lan_stikcopy.jl", "$datadir/aa_source_code.jl"; force=true)
end

if !isdir("$datadir")
    mkdir("$datadir")
    println("Created new dir: $datadir")
    open("$datadir/aa_params.txt", "w") do file
        write(file, "H0=$H0\nmu=$mu\nBm=$Bm\nlambda=$lambda\nsteps=$steps\ndt=$dt\nw=$w\n")
    end
    cp("main_lan_stikcopy.jl", "$datadir/aa_source_code.jl")
end



previous_i_when_flip = -1000
previous_i_when_split = -1000
println("Running on $(Threads.nthreads()) threads")
for i in 1:steps
    println("----------------------------------------------------------------------")
    println("----- Number of points: $(size(points,2)) ---------- Step ($i)$(i+last_step)----------")
    println("----------------------------------------------------------------------")
    global points, faces, connectivity, normals, all_vs, velocities, neighbor_faces, edges, CDE
    global t, H0, epsilon
    global max_abs_v, max_v_avg
    global previous_i_when_flip, previous_i_when_split
    edges = make_edges(faces)
    neighbor_faces = make_neighbor_faces(faces)
    connectivity = make_connectivity(edges)
    #normals, CDE = make_normals_spline(points, connectivity, edges, normals)
    normals, CDE, AB = make_normals_parab(points, connectivity, normals; eps = 10^-8)
    psi = PotentialSimple_par(points, faces, normals, mu, H0)
    Ht = HtField_par(points, faces, psi, normals)
    Hn_norms = NormalFieldCurrent_par(points, faces, normals, Ht, mu, H0)
    Hn = normals .* Hn_norms'
    #println("H = $(H0)")
    #mup = mu
    # magnitudes squared of the normal force
    Hn_2 = sum(Hn.^2, dims=1)
    # magnitudes squared of the tangential force
    Ht_2 = sum(Ht.^2, dims=1)

    #tensorn = mup*(mup-1)/8/pi * Hn_2 + (mup-1)/8/pi * Ht_2
    #tensorn = tensorn * Bm
    #zc = SG.Zinchenko2013(points, faces, normals)
    #println("velocity:")
    #@time velocitiesn_norms = InterfaceSpeedZinchenko(points, faces, tensorn, eta, gamma, normals)
    #velocities = normals .* velocitiesn_norms' # need to project v_norms on surface

    println("Bm = $Bm")

    @time velocities = make_magvelocities_par(points, normals, lambda, Bm, mu, Hn_2, Ht_2)
    @time velocities = make_Vvecs_conjgrad_par(normals,faces, points, velocities, 1e-6, 500)

    #@time velocities = make_enright_velocities(points, t)
    #passive stabilization
    #velocities = SG.stabilise!(velocities, points, faces, normals, zc)

    dt = 0.05*minimum(make_min_edges(points,connectivity)./sum(sqrt.(velocities.^2),dims=1))
     #if dt < 0.2
    #     dt = 0.2
    # end
    println("max v: $(maximum(abs.(velocities))),   min v: $(minimum(abs.(velocities)))")
    t += dt
    println("---- t = $t, dt = $dt ------")

    points = points + velocities * dt
    normals, CDE, AB = make_normals_parab(points, connectivity, normals; eps = 10^-8)
    #normals, CDE = make_normals_spline(points, connectivity, edges, normals)

    cutoff_crit = 0.5 # 0.5 for sqrt(dS) measure, 0.55 for maxd measure
    minN_triangles_to_split = 5
    #if i == 1
    	# hot fix to stop the LoadError: SingularException(5) on 1788th step
    	
    #	minN_triangles_to_split = 6
    #end
    
    # min_fraction_triangles_to_split = 13/162
    # minN_triangles_to_split = size(points,2) * min_fraction_triangles_to_split
    #minN_triangles_to_split = 30

    # changed this function to measure charecteritic length as sqrt(dS)
    marked_faces  = mark_faces_for_splitting(points, faces, edges, CDE, neighbor_faces; cutoff_crit = cutoff_crit)
    println("Number of too large triangles: ",sum(marked_faces))
    if (sum(marked_faces) >= minN_triangles_to_split) && (i - previous_i_when_split >= 5)
        # split no more often than every 26 iterations to allow flipping to remedy the mesh
        previous_i_when_split = i
        # if i == 9
        #     break
        # end
        println("-----------------------------------")
        println("----------Adding mesh points-------")
        println("    V-E+F = ", size(points,2)-size(edges,2)+size(faces,2))
        println("    number of points: ", size(points,2))
        println("    number of faces: ", size(faces,2))
        println("    number of edges: ", size(edges,2))
        println("-----------------------------------")

        points_new, faces_new = add_points(points, faces,normals, edges, CDE; cutoff_crit = cutoff_crit)
        edges_new = make_edges(faces_new)
        connectivity_new = make_connectivity(edges_new)
	
	if i == 1
		data_fresh = [points_new, faces_new]
		#println("Finished step $(last_step + i)")
		@save "/home/andris/sim_data/just_added_points$(lpad(i + last_step,5,"0")).jld2" data_fresh
        end
        
        println("-----------------------------------")
        println("New V-E+F = ", size(points_new,2)-size(edges_new,2)+size(faces_new,2))
        println("New number of points: ", size(points_new,2))
        println("New number of faces: ", size(faces_new,2))
        println("New number of edges: ", size(edges_new,2))
        println("-----------------------------------")
        println("active stabbing after adding points")
        println("------flipping edges first---------")
        faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
        edges_new = make_edges(faces_new)
        println("-- flipped?: $do_active")
        println("---- active stabbing first --------")
        points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)
        println("------flipping edges second---------")
        faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
        edges_new = make_edges(faces_new)
        println("-- flipped?: $do_active")
        println("---- active stabbing second --------")
        points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)
	
	
	if i >= 1 # ==1 
	# hot fix to stop the not stopping of paraboloid fit on 1947th step
	# hot fix to stop the not stopping of paraboloid fit on 2001st step # from now on do 6 stabbings after adding points
		println("-----------------------------------")
		println("New V-E+F = ", size(points_new,2)-size(edges_new,2)+size(faces_new,2))
		println("New number of points: ", size(points_new,2))
		println("New number of faces: ", size(faces_new,2))
		println("New number of edges: ", size(edges_new,2))
		println("-----------------------------------")
		println("active stabbing after adding points")
		println("------flipping edges 3---------")
		faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
		edges_new = make_edges(faces_new)
		println("-- flipped?: $do_active")
		println("---- active stabbing 3 --------")
		points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)
		println("------flipping edges 4---------")
		faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
		edges_new = make_edges(faces_new)
		println("-- flipped?: $do_active")
		println("---- active stabbing 4 --------")
		points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)
		println("-----------------------------------")
		println("New V-E+F = ", size(points_new,2)-size(edges_new,2)+size(faces_new,2))
		println("New number of points: ", size(points_new,2))
		println("New number of faces: ", size(faces_new,2))
		println("New number of edges: ", size(edges_new,2))
		println("-----------------------------------")
		println("active stabbing after adding points")
		println("------flipping edges 5---------")
		faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
		edges_new = make_edges(faces_new)
		println("-- flipped?: $do_active")
		println("---- active stabbing 5 --------")
		points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)
		println("------flipping edges 6---------")
		faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
		edges_new = make_edges(faces_new)
		println("-- flipped?: $do_active")
		println("---- active stabbing 6 --------")
		points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)

	end
	
        points, faces, edges, connectivity = points_new, faces_new, edges_new, connectivity_new
        normals = Normals(points, faces)
        println("New first approx normals pointing out? ", all(sum(normals .* points,dims=1).>0))
        normals, CDE, AB = make_normals_parab(points, connectivity, normals; eps = 10^-8)
        #normals, CDE = make_normals_spline(points, connectivity, edges, normals)
        println("New normals pointing out? ", all(sum(normals .* points,dims=1).>0))
        println("-----------------------------------")
        println("---------- Points added -----------")
        println("-----------------------------------")

    else # stabilize regularly if havent added new faces
        #H0 = [sin(w*t), 0., cos(w*t)]
        do_active = false

        faces, connectivity, do_active = flip_edges(faces, connectivity, points)

        if do_active
            if i - previous_i_when_flip > 5
                println("doing active because flipped")
                edges = make_edges(faces)
                points = active_stabilize(points, faces, CDE, connectivity, edges, normals)
            else
                println("flipped; not doing active")
            end
            previous_i_when_flip = i
        end

        if i % 100 == 0 && i > 2

            println("doing active every 100th time step")
            points = active_stabilize(points, faces, CDE, connectivity, edges, normals)

        end
    end


    #dt = 0.1 * scale / max(sqrt(sum(Vvecs.*Vvecs,2)))
    # ElTopo magic
    #actualdt,points2,faces2 = improvemeshcol(points,faces,points2,par)
    if reset_vmax
        println("Resetting v0max")
        global v0max = maximum(abs.(velocities))
        reset_vmax = false
    end
    vi = maximum(abs.(velocities))
    max_abs_v[i] = vi

    max_vs[:,i] = [velocities[1, argmax(abs.(velocities[1,:]))],
                    velocities[2, argmax(abs.(velocities[2,:]))],
                    velocities[3, argmax(abs.(velocities[3,:]))]]
    mean_vs[:,i] = [StatsBase.mean(abs.(velocities[1,:])),
                    StatsBase.mean(abs.(velocities[2,:])),
                    StatsBase.mean(abs.(velocities[3,:]))]
    try
        max_v_avg = mean(max_abs_v[i-4:i])
    catch
        max_v_avg = max_abs_v[i]
    end

    if max_v_avg > v0max
        println("updated v0max")
        v0max = max_v_avg
    end

    println("Bm = $Bm")
    println("vi = $vi, max_v_avg = $max_v_avg, v0max = $v0max, max_v_avg/v0max = $(max_v_avg/v0max)")
    println("mean vs: $(mean_vs[:,i])")

    a,b,c = maximum(points[1,:]), maximum(points[2,:]), maximum(points[3,:])
    println(" --- c/a = $(c/a) , c/b = $(c/b)")

    if i % 1 == 0
        data = [points, faces, t, H0, Bm, v0max]
        #println("Finished step $(last_step + i)")
        @save "$datadir/data$(lpad(i + last_step,5,"0")).jld2" data
        data2 = [max_vs[:, 1:i], max_abs_v[1:i]]
        @save "$datadir/speeds.jld2" data2
    end

    if max_v_avg/v0max < epsilon
        println("-------------------------------------------------------------------- Increasing Bm at step $i")
        global reset_vmax
        reset_vmax = true
        global Bm
        #Bm -= 0.2
        println("----- new Bm = $Bm")
        #break
   end
end # end simulation iterations

data = [max_vs, mean_vs, points, faces]
@save "/home/andris/sim_data/$(dataname)_v.jld2" data

println("Sim done :)")

#scene = Makie.mesh(points', faces', color = :gray, shading = false, visible = true)
#Makie.wireframe!(scene[end][1], color = :black, linewidth = 2)


# scene = Makie.mesh(points2_small', faces1',color = :gray, shading = false, visible = true)
# Makie.wireframe!(scene[end][1], color = :blue, linewidth = 2,visible = true)
#
# scene = Makie.mesh(points1_large', faces1',color = :gray, shading = false, visible = true)
# Makie.wireframe!(scene[end][1], color = :red, linewidth = 2,visible = true)
#
# using Plots
# using PyPlot
# pygui()
#
# fig = figure(figsize=(7,7))
# ax = fig[:gca](projection="3d")
#
# (x, y, z) = [points[i,:] for i in 1:3]
# (vx, vy, vz) = [velocities[i,:] for i in 1:3]
#
# ax[:scatter](x,y,z, s=2,color="k")
# ax[:quiver](x,y,z,vx,vy,vz, length=30, arrow_length_ratio=0.5)
#
# ax[:set_xlim](-2,2)
# ax[:set_ylim](-2,2)
# ax[:set_zlim](-2,2)
# ax[:set_xlabel]("x axis")
# ax[:set_ylabel]("y axis")
# ax[:set_zlabel]("z axis")
# fig[:show]()

#
# using PyPlot
#
# pygui(true)
#
# fig = figure()
# ax = fig[:gca](projection="3d")
#
# N = 10
# x,y,z,u,v,w = [randn(N) for _ in 1:6]
# ax[:quiver](x,y,z, u,v,w)




# elparameters(scale) = Elparameters( # comments are some values that work fine
#  m_use_fraction = false,                     # false
#  m_min_edge_length = 0.7*scale,             # 0.7 * scale
#  m_max_edge_length = 1.5*scale,               # 1.5 * scale
#  m_max_volume_change = 0.1*scale^3,         # 0.1 * scale^3
#  m_min_curvature_multiplier = 1,             # 1
#  m_max_curvature_multiplier = 1,            # 1
#  m_merge_proximity_epsilon = 0.5*scale,     # 0.5 * scale
#  m_proximity_epsilon = 0.00001,             # 0.00001
#  m_perform_improvement = true,              # true
#  m_collision_safety = false,                 # false
#  m_min_triangle_angle = 15,                 # 15
#  m_max_triangle_angle = 120,                # 120
#  m_allow_vertex_movement = false,           # false   ### This is where is a bug
#  m_use_curvature_when_collapsing = false,    # false
#  m_use_curvature_when_splitting = false,    # false
#  m_dt = 1                                   # 1
# )
#par = elparameters(scale

# for i in 1:size(edges,2)
#     for j in 1:size(edges,2)
#         if i!= j
#             if edges[:,i] == edges[:,j]
#                 println("double")
#             end
#         end
#     end
# end
