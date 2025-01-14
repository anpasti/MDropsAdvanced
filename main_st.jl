using LinearAlgebra
using CSV
using Makie
using StatsBase
using Optim
using JLD2
using Dates
#using SurfaceGeometry
#using ElTopo
#using PyPlot
#include("./SurfaceGeometry/dt20L/src/Iterators.jl")
#include("./SurfaceGeometry/dt20L/src/ComplexDS.jl")
include("./SurfaceGeometry/dt20L/src/SurfaceGeometry.jl")
SG = SurfaceGeometry
#include("./SurfaceGeometry/dt20L/src/Iterators.jl")
include("./stabilization.jl")
include("./functions.jl")
include("./mesh_functions.jl")
include("./sandbox_lang.jl")
include("./physics_functions.jl")

points_csv= CSV.read("./meshes/points_ellipse_manyN.csv", header=0)
faces_csv = CSV.read("./meshes/faces_ellipse_manyN.csv", header=0)
println("Loaded mesh")

points = convert(Array, points_csv)
faces = convert(Array, faces_csv)
points = Array{Float64}(points')
faces = Array{Int64}(faces')

# datadir="/home/andris/sim_data/2019-03-15/1"
# last_file = readdir(datadir)[end]
# println("loaded: ",last_file)
# @load "$datadir/$last_file" data
# points = data[1]
# faces = data[2]
# faces = Array{Int64,2}(faces)
# t = data[3]

#points, faces = expand_icosamesh(;R=1,depth=3)
#points = Array{Float64}(points)
#faces = Array{Int64}(faces)

# dir = "pushing_to_limit_langfix_extended"
# sourcedir = "/home/andris/sim_data/$dir"
# sourcefile = "data00013.jld2"
# @load "$sourcedir/$sourcefile" data
# points = data[1]
# faces = data[2]
# points = Array{Float64}(points)
# faces = Array{Int64}(faces)
# t = data[3]

edges = make_edges(faces)
connectivity = make_connectivity(edges)

H0 = [0., 0., 1.]
#omega = 1.
mu = 30.
lambda = 10.
Bm = 0.#2.5
dtsize = 0.3
#H0 = [cos(omega*t), sin(omega*t), 0.]

steps = 200
#
date = string(Dates.today())
daydir = "/home/andris/sim_data/$date"
if !isdir("$daydir")
    mkdir("$daydir")
    println("Good morning!")
    println("Created new dir: $daydir")
end
datadir = "$daydir"
for i = 1:100
    global datadir
    datadir = "$daydir/$i"
    if !isdir("$datadir")
        mkdir("$datadir")
        println("Created new dir: $datadir")
        break
    end
end

open("$datadir/_params.txt", "w") do file
    write(file, "H0=$H0\nmu=$mu\nlambda=$lambda\nBm=$Bm\nsteps=$steps\ndtsize=$dtsize")#\nomega=$omega")
end

t = 0.
normals = Normals(points, faces)
for iter in 1:steps
    println("--------------time step $(iter)---------------------")

    global points, faces,normals, edges, connectivity, t, velocities, Hn_2, Ht_2, H0
    #global points2
    #H0 = [cos(omega*t), sin(omega*t), 0.]

    (normals, CDE) = make_normals_spline(points, connectivity, edges, normals)

    psi = PotentialSimple(points, faces, mu, H0; normals = normals)
    #psi2 = PotentialSimple(points2, faces, mu, H0; normals = normals2)
    Ht = HtField(points, faces, psi, normals)
    #Ht2 = HtField(points2, faces, psi2, normals2)
    Hn_norms = NormalFieldCurrent(points, faces, Ht, mu, H0; normals = normals)
    #Hn_norms2 = NormalFieldCurrent(points2, faces, Ht2, mu, H0; normals = normals2)
    Hn = normals .* Hn_norms'
    #Hn2 = normals2 .* Hn_norms2'

    mup = mu
    # magnitudes squared of the normal force
    Hn_2 = sum(Hn.^2, dims=1)
    #Hn2_2 = sum(Hn2.^2, dims=1)
    # magnitudes squared of the tangential force
    Ht_2 = sum(Ht.^2, dims=1)
    #Ht2_2 = sum(Ht2.^2, dims=1)

    #tensorn = mup*(mup-1)/8/pi * Hn_2 + (mup-1)/8/pi * Ht_2

    # make the force normal to surface
    #tensorn = normals .* tensorn

    # eta = 1, gamma = 1.
    #velocitiesn_norms = InterfaceSpeedZinchenko(points, faces, tensorn, 1., 1., normals)
    #velocitiesn = normals .* velocitiesn_norms'
    #velocities = velocitiesn

    velocities = make_magvelocities(points, normals, lambda, Bm, mu, Hn_2, Ht_2)
    velocities = make_Vvecs_conjgrad(normals,faces, points, velocities, 1e-6, 500)

    #velocities = sum(velocities .* normals,dims=1) .* normals

    #zc = SG.Zinchenko2013(points, faces, normals)
    #SG.stabilise!(velocities,points, faces, normals, zc)

    #dt = dtsize*minimum(make_min_edges(points,connectivity)./sum(sqrt.(velocities.^2),dims=1))
    dt = 0.3
    t += dt
    #dt2 = 0.4*minl2/maxv2
    println("dt = $(dt)")
    #println("dt2 = $(dt2)")
    #
    points += velocities * dt
    normals, CDE = make_normals_spline(points, connectivity, edges, normals)
    #points2 += velocities2 * dt
    do_active = false
    faces, connectivity, do_active = flip_edges(faces, connectivity, points)
    edges = make_edges(faces)
    connectivity = make_connectivity(edges)

    if iter % 20 == 0 || do_active

        println("OUTSIDE re-did sum: ", sum(edges))
        println("doing active / step $iter / flipped?: $do_active")
        points = active_stabilize(points, faces, CDE, connectivity,edges, normals,deltakoef=0.05)

    end

    if iter % 1 == 0
       data = [points, faces, t, H0]
       @save "$datadir/data$(lpad(iter,5,"0")).jld2" data
    end
end
println("hooray calculation Finnished!!!! :-)")
# (normals, CDE) = make_normals_spline(points, connectivity, edges, normals)
# points1 = active_stabilize(points,faces,CDE,connectivity,normals;maxiters=100)
# for i in 1:size(points,2)
#     for j in 1:size(points,2)
#         edges = make_edges(faces2)
#         connectivity = make_connectivity(edges)
#         println(i," ",j)
#         flip_connectivity!(faces2,i,j,points,connectivity)
#     end
# end
# edges = make_edges(faces2)
# connectivity = make_connectivity(edges)
# (normals2, CDE2) = make_normals_spline(points1, connectivity, edges, normals)
# points2 = active_stabilize(points1,faces2,CDE2,connectivity,normals2;maxiters=100)

#scene = Makie.mesh(points', faces',color = :white, shading = false,visible = true)
#Makie.wireframe!(scene[end][1], color = :black, linewidth = 1)
# scene = Makie.mesh!(points2', faces',color = :gray, shading = false,visible = true)
# Makie.wireframe!(scene[end][1], color = :blue, linewidth = 1,visible = true)

#
# using PyPlot
# pygui()
#
# fig = figure(figsize=(7,7))
# ax = fig[:gca](projection="3d")
#
# (x, y, z) = [points[i,:] for i in 1:3]
# (vx, vy, vz) = [velocities[i,:] for i in 1:3]
# ax[:scatter](x,y,z, s=2,color="k")
#
# ax[:quiver](x,y,z,vx,vy,vz, length=0.0001, arrow_length_ratio=0.5)
# ax[:set_title]("Normal velocities")
#
# ax[:set_xlim](-2,2)
# ax[:set_ylim](-2,2)
# ax[:set_zlim](-2,2)
# ax[:set_xlabel]("x axis")
# ax[:set_ylabel]("y axis")
# ax[:set_zlabel]("z axis")
# fig[:show]()
