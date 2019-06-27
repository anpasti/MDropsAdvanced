using Pkg
pkg"activate ."
pkg"resolve"


#using Plots
#using Makie
using JLD2
using FileIO
using Optim

#p = Plots

include("./SurfaceGeometry/dt20L/src/SurfaceGeometry.jl")
SG = SurfaceGeometry

function eq431(e, mu)
    # 4pi ?
    N = 4pi*(1-e^2) / (2*e^3) * (log((1+e)/(1-e)) - 2e)
    temp1 = (3-2e^2)/e^2 - (3-4e^2)*asin(e)/(e^3 * sqrt(1-e^2))
    temp2 = (1-e^2)^(2/3) * ((3-e^2) * log((1+e)/(1-e))/e^5 - 6/e^4)

    # 4pi ?
    return (4pi/(mu-1) + N)^2 * 1/2pi * temp1/temp2
end

function shank(seq)
    return seq[3] - (seq[3] - seq[2])^2 / ((seq[3] - seq[2]) - (seq[2] - seq[1]))
end

mu=3

dir = "elong_sphere_5"
sourcedir = "/home/laigars/sim_data/$dir"
len = size(readdir(sourcedir),1) - 1

es = []
num_es = []
Bms = []
num_Bms = []
cs = []
all_Bms = []
all_es = []
last_Bm = 0

volumes = []

for i in 5:5:1035
    # data = [points, faces, t, H0, Bm, v0max]
    global last_Bm, es
    @load "$sourcedir/data$(lpad(i,5,"0")).jld2" data
    #println("step $i")
    points = data[1]
    faces = data[2]
    Bm = data[end-1]

    #println("$Bm -- $i")
    function f(ab::Array{Float64,1})
        return sum((points[1,:].^2/ab[1]^2 .+ points[2,:].^2/ab[1]^2 .+
                points[3,:].^2/ab[2]^2 .- 1).^2)
    end

    #println(f([1.,1.]))
    x0 = [0.99, 1.01]
    res = Optim.optimize(f,x0)
    a = Optim.minimizer(res)[1]
    b = Optim.minimizer(res)[2]

    e = sqrt(1-(a/b)^2)
    push!(num_es, e)
    push!(num_Bms, Bm)
    push!(volumes, -SG.volume(points, faces))

    if Bm > last_Bm
        es = []
        push!(all_Bms, Bm)
        last_Bm = Bm
        push!(all_es, es)
    end
    push!(es, e)

end

finals = []
finals2 = []

for arr in all_es
    final = shank(arr[end-2:end])
    push!(finals, final)
    if length(arr) < 4
	final2 = -1
    else
        final2 = shank(arr[end-3:end-1])
    end

    push!(finals2, final2)

    println("1st: $final, 2nd: $final2, delta: $((final-final2)/final)")

end

println("finals new: $finals")



#p.scatter(num_es, num_Bms, label="all")#, color="lightblue")
#p.scatter(finals, all_Bms, label="shanked")#, color="red")#, title="mu = $mu")
#title("mu = $mu")
#p.xlabel!("e")
#p.ylabel!("Bm")

#e = 0:0.002:finals[end]+0.01
#p.plot!(e, eq431.(e, mu), legend=:right, title=:"mu=3", label="teor", color="green")

#p.plot!(e, (3/4/pi*4.141)^(1/3) * eq431.(e, mu), legend=:right, title=:"mu=3", label="teor_scaled")
#p.plot!(es, eq431.(es, 30))
