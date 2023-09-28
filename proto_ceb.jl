#cd("/home/stikuts/MDropsAdvanced/")

using Pkg

pkg"activate ."
pkg"resolve"

using JLD2
using StatsBase
using LinearAlgebra
using FastGaussQuadrature
using Optim
using Distributed
using Dates
using Roots

include("./SurfaceGeometry/dt20L/src/Iterators.jl")
include("./mesh_functions.jl")
include("./physics_functions.jl")
include("./mathematics_functions.jl")

points, faces = expand_icosamesh(R=1, depth=3)

#@load "./meshes/faces_critical_hyst_2_21.jld2" faces
points = Array{Float64}(points)
faces = Array{Int64}(faces)

println("Loaded mesh; nodes = $(size(points,2))")

t = 0.
dt = 0.05
steps = 3500000
last_step = 0
cutoff_crit = 0.2 # for triangle size


previous_i_when_flip = -1000
previous_i_when_split = -1000

println("Running on $(Threads.nthreads()) threads")


steps = steps - last_step
for i in 1:steps
    if t > 500.
    	break
    end
    println("----------------------------------------------------------------------")
    println("----- Number of points: $(size(points,2)) ---------- Step ($i)$(i+last_step)--- t = $(t)-------")
	println("----------$(Dates.format(now(), "yyyy-mm-dd;  HH:MM:SS"))------------")
    println("----------------------------------------------------------------------")

    global points, faces, connectivity, normals, velocities, velocities_n, neighbor_faces, edges, CDE
    global dt, t, V0
    global previous_i_when_flip, previous_i_when_split, cutoff_crit

    edges = make_edges(faces)
    neighbor_faces = make_neighbor_faces(faces)
    connectivity = make_connectivity(edges)
	if i == 1
		normals = Normals(points, faces)
	end
    normals, CDE, AB = make_normals_parab(points, connectivity, normals; eps = 10^-8)
	if i == 1
		V0 = make_volume(points,faces, normals)
		println("start volume = ",V0)
	end



    ##### calculate velocity here #####
    velocities_phys = zeros(size(points))

    velocities = make_Vvecs_conjgrad(normals,faces, points, velocities_phys, 1e-6, 500) # passive stabilization
    ###################################



    dt = min(make_zinchencko_dt(points, connectivity, CDE, 7.4), 0.07)
    println("---- dt = ", dt, " ----")
    t += dt
    points = points + velocities * dt


    normals, CDE, AB = make_normals_parab(points, connectivity, normals; eps = 10^-8)


    # rescaling so that volume = V0
	rc = make_center_of_mass(points,faces, normals)
	V = make_volume(points,faces, normals)
	println("volume before rescale ", V)
	points = (points .- rc) * (V0/V)^(1/3) .+ rc
	println("rescaled to volume ", make_volume(points,faces, normals), "; start volume ", V0)
	println("center of mass ", rc)



    # Mesh stabilization procedure
    minN_triangles_to_split = 1

    marked_faces  = mark_faces_for_splitting(points, faces, edges, CDE, neighbor_faces; cutoff_crit = cutoff_crit)
    println("Number of too large triangles: ",sum(marked_faces))
    if (sum(marked_faces) >= minN_triangles_to_split) && (i - previous_i_when_split >= 5)
        # split no more often than every 5 iterations to allow flipping to remedy the mesh
        previous_i_when_split = i

        # println("-----------------------------------")
        println("----------Adding mesh points-------")
        # println("    V-E+F = ", size(points,2)-size(edges,2)+size(faces,2))
        # println("    number of points: ", size(points,2))
        # println("    number of faces: ", size(faces,2))
        # println("    number of edges: ", size(edges,2))
        # println("-----------------------------------")

        points_new, faces_new = add_points(points, faces,normals, edges, CDE; cutoff_crit = cutoff_crit)
        edges_new = make_edges(faces_new)
        connectivity_new = make_connectivity(edges_new)

        println("-----------------------------------")
        println("New V-E+F = ", size(points_new,2)-size(edges_new,2)+size(faces_new,2))
        println("New number of points: ", size(points_new,2))
        println("New number of faces: ", size(faces_new,2))
        println("New number of edges: ", size(edges_new,2))
        println("-----------------------------------")
        # println("active stabbing after adding points")
        # println("------flipping edges first---------")


        # faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
        # edges_new = make_edges(faces_new)
        # println("-- flipped?: $do_active")
        # println("---- active stabbing first --------")
        # points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)
        # println("------flipping edges second---------")
        # faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
        # edges_new = make_edges(faces_new)
        # println("-- flipped?: $do_active")
        # println("---- active stabbing second --------")
        # points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)

		faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
		stabiter = 1
		while do_active && stabiter <= 3
			println("-- flipped?: $do_active")
			println("---- active stabbing $stabiter --------")
			stabiter += 1

			points_new = active_stabilize_old_surface(points,CDE,normals,points_new, faces_new, connectivity_new, edges_new)
			faces_new, connectivity_new, do_active = flip_edges(faces_new, connectivity_new, points_new)
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
    end # end mesh stabilization


    # check ~ellipsoidal aprameters
    a,b,c = maximum(points[1,:]), maximum(points[2,:]), maximum(points[3,:])
    println(" --- c/a = $(c/a) , c/b = $(c/b)")


end # end simulation iterations
println("hooray, simulation finished :)")
