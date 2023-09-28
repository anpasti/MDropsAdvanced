#include("./SurfaceGeometry/dt20L/src/SurfaceGeometry.jl")
#include("./SurfaceGeometry/dt20L/src/Iterators.jl")

function make_magvelocities_old(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2)
    # lambda = int viscosity / ext viscosity
    # Returns vertex velocities from the Stokes flow integral equations
    # using Wielandt's deflation as describeed in Zinchenko 1997, Das 2017
    # w + T*w + 4*pi*C*w + B*w = F
    println("calculating velocities")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2

    S = sum(deltaS)
    xc = [0.,0.,0.]
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        xc = xc + rx * deltaS[x] / S
    end

    F = zeros(3*size(vertices,2),1)
    T = zeros(3*size(vertices,2),3*size(vertices,2))
    #C = C1+C2
    C1 = zeros(3*size(vertices,2),3*size(vertices,2))
    C2 = zeros(3*size(vertices,2),3*size(vertices,2))
    B = zeros(3*size(vertices,2),3*size(vertices,2))

    M = zeros(3,3)
    # fill M
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        # Matrix(1.0I, 3, 3) <- 3x3 Identity matrix
        M += (Matrix(1.0I, 3, 3) * dot(rx - xc, rx - xc) -
            (rx - xc) * (rx - xc)') * deltaS[x]
            #   ^ changed the transpose
    end
    Minv = inv(M) # calculate inverse of M

    #create Levi-Civita symbol
    epsilon = cat(dims=3, [0 0  0;  0 0 1; 0 -1 0],
                              [0 0 -1;  0 0 0; 1  0 0],
                              [0 1  0; -1 0 0; 0  0 0])


    for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3

            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                #else
                #polar integration scheme
                end

                for i = 1:3
                    if x != y
                       T[3*y-3+j,3*x-3+i] = -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]

                       T[3*y-3+j,3*y-3+i] -= -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]
                    end

                    B[3*y-3+j,3*x-3+i] = -4*pi/S * normals[j,y] * normals[i,x] * deltaS[x]

                    C1[3*y-3+j,3*x-3+i] =  deltaS[x]/S

                    for k = 1:3
                    for m = 1:3
                    for n = 1:3
                    for p = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            epsilon[j,k,m]*Minv[k,n]*epsilon[n,p,i]*
                            (rx[p] - xc[p])*deltaS[x]*(ry[m] - xc[m])
                    end
                    end
                    end
                    end

                end
            end

        end
    end
    C = C1 + C2

    # w + T*w + 4*pi*C*w + B*w = F
    #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
    w = (Matrix(1.0I, size(T)) + T + (lambda-1)/2*C + (lambda-1)/(8*pi)*B) \ F

    # w' = C*w
    # v = w + (lambda-1)/2 * w'
    varr = w + (lambda-1)/2 * C*w


    velocities = reshape(varr ,3,size(vertices,2))
    return velocities
end

function make_magvelocities(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2)
    #without the use of levi civita tensor
    # lambda = int viscosity / ext viscosity
    # Returns vertex velocities from the Stokes flow integral equations
    # using Wielandt's deflation as describeed in Zinchenko 1997, Das 2017
    # w + T*w + 4*pi*C*w + B*w = F
    println("calculating velocities")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2


    S = sum(deltaS)
    xc = [0.,0.,0.]
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        xc = xc + rx * deltaS[x] / S
    end

    F = zeros(3*size(vertices,2),1)
    T = zeros(3*size(vertices,2),3*size(vertices,2))
    #C = C1+C2
    C1 = zeros(3*size(vertices,2),3*size(vertices,2))
    C2 = zeros(3*size(vertices,2),3*size(vertices,2))
    B = zeros(3*size(vertices,2),3*size(vertices,2))

    M = zeros(3,3)
    # fill M
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        # Matrix(1.0I, 3, 3) <- 3x3 Identity matrix
        M += (Matrix(1.0I, 3, 3) * dot(rx - xc, rx - xc) -
            (rx - xc) * (rx - xc)') * deltaS[x]
            #   ^ changed the transpose
    end
    Minv = inv(M) # calculate inverse of M

    for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3

            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                #else
                #polar integration scheme
                end

                for i = 1:3
                    if x != y
                       T[3*y-3+j,3*x-3+i] = -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]

                       T[3*y-3+j,3*y-3+i] -= -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]
                    end

                    B[3*y-3+j,3*x-3+i] = -4*pi/S * normals[j,y] * normals[i,x] * deltaS[x]

                    C1[3*y-3+j,3*x-3+i] =  deltaS[x]/S

                    for k = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,j]*(rx[k] - xc[k])*deltaS[x]*(ry[i] - xc[i]) -
                            Minv[i,j]*(rx[k] - xc[k])*deltaS[x]*(ry[k] - xc[k]) -
                            Minv[k,k]*(rx[j] - xc[j])*deltaS[x]*(ry[i] - xc[i]) +
                            Minv[i,k]*(rx[j] - xc[j])*deltaS[x]*(ry[k] - xc[k])
                    end
                    if i==j
                        for k = 1:3
                        for m = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,k]*(rx[m] - xc[m])*deltaS[x]*(ry[m] - xc[m]) -
                            Minv[k,m]*(rx[k] - xc[k])*deltaS[x]*(ry[m] - xc[m])
                        end
                        end
                    end

                end
            end

        end
    end
    C = C1 + C2

    # w + T*w + 4*pi*C*w + B*w = F
    #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
    w = (Matrix(1.0I, size(T)) + T + (lambda-1)/2*C + (lambda-1)/(8*pi)*B) \ F

    # w' = C*w
    # v = w + (lambda-1)/2 * w'
    varr = w + (lambda-1)/2 * C*w


    velocities = reshape(varr ,3,size(vertices,2))
    return velocities
end


function make_magvelocities_no_Wie(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2)
    # Returns vertex velocities from the Stokes flow integral equations without Wielandt's deflation
    # v = Av + F
    println("calculation velocities, no Wieldandt")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2

    F = zeros(3*size(vertices,2),1)
    A = zeros(3*size(vertices,2), 3*size(vertices,2))

    for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3

            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                end

                # build the A matrix
                for i = 1:3
                    if x != y
                        A[3y-3+j, 3x-3+i] = (1-lambda)/8/pi * (-6 * (rx[i]-ry[i])*(rx[j]-ry[j])) / norm(rx-ry)^5 *
                            dot(rx-ry, normals[:,x]) * deltaS[x]
                        A[3y-3+j, 3y-3+i] += (1-lambda)/8/pi * (-6 * (rx[i]-ry[i])*(rx[j]-ry[j])) / norm(rx-ry)^5 *
                            dot(rx-ry, normals[:,x]) * deltaS[x]


                    end # if x!=y
                end # i for
            end # x for
        end # j for
    end # y for

    varr = (A-I) \ F;  # solve integral eq. for velocity

    velocities = reshape(varr ,3,size(vertices,2))
    return velocities

end # function

function make_magvelocities_2(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2)
    # early exit for lambda=1
    #without the use of levi civita tensor
    # lambda = int viscosity / ext viscosity
    # Returns vertex velocities from the Stokes flow integral equations
    # using Wielandt's deflation as describeed in Zinchenko 1997, Das 2017
    # w + T*w + 4*pi*C*w + B*w = F
    println("calculating velocities")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2


    S = sum(deltaS)
    xc = [0.,0.,0.]
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        xc = xc + rx * deltaS[x] / S
    end

    F = zeros(3*size(vertices,2),1)
    T = zeros(3*size(vertices,2),3*size(vertices,2))
    #C = C1+C2
    C1 = zeros(3*size(vertices,2),3*size(vertices,2))
    C2 = zeros(3*size(vertices,2),3*size(vertices,2))
    B = zeros(3*size(vertices,2),3*size(vertices,2))



    if lambda == 1.
        println("lambda = 1; calculating v directly")
        for y = 1:size(vertices,2)
            ry = vertices[:,y]

            for j = 1:3

                for x = 1:size(vertices,2)
                    rx = vertices[:,x]

                    Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                    (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                    if x != y
                       F[3*y-3+j] +=
                           1/(8*pi) *(
                           dot(rx-ry,normals[:,x]) * normals[j,y]+
                           dot(rx-ry,normals[:,y]) * normals[j,x]+
                           (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                           3*(rx[j]-ry[j])/norm(rx-ry)^2*
                           dot(normals[:,x]+normals[:,y],rx-ry)*
                           dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                    # magnetic part
                       F[3*y-3+j] += Bm/(8*pi) *
                       (fbar[x] - fbar[y]) *
                       dot(Gji[j,:], normals[:,x]) * deltaS[x]
                    #else
                    #polar integration scheme
                    end
                end

            end
        end
        # w + T*w + 4*pi*C*w + B*w = F
        #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
        w = (Matrix(1.0I, size(T)) + T) \ F

        # w' = C*w
        # v = w + (lambda-1)/2 * w'
        varr = w


        velocities = reshape(varr ,3,size(vertices,2))
        return velocities
    end



    M = zeros(3,3)
    # fill M
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        # Matrix(1.0I, 3, 3) <- 3x3 Identity matrix
        M += (Matrix(1.0I, 3, 3) * dot(rx - xc, rx - xc) -
            (rx - xc) * (rx - xc)') * deltaS[x]
            #   ^ changed the transpose
    end
    Minv = inv(M) # calculate inverse of M

    for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3

            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                #else
                #polar integration scheme
                end

                for i = 1:3
                    if x != y
                       T[3*y-3+j,3*x-3+i] = -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]

                       T[3*y-3+j,3*y-3+i] -= -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]
                    end

                    B[3*y-3+j,3*x-3+i] = -4*pi/S * normals[j,y] * normals[i,x] * deltaS[x]

                    C1[3*y-3+j,3*x-3+i] =  deltaS[x]/S

                    for k = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,j]*(rx[k] - xc[k])*deltaS[x]*(ry[i] - xc[i]) -
                            Minv[i,j]*(rx[k] - xc[k])*deltaS[x]*(ry[k] - xc[k]) -
                            Minv[k,k]*(rx[j] - xc[j])*deltaS[x]*(ry[i] - xc[i]) +
                            Minv[i,k]*(rx[j] - xc[j])*deltaS[x]*(ry[k] - xc[k])
                    end
                    if i==j
                        for k = 1:3
                        for m = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,k]*(rx[m] - xc[m])*deltaS[x]*(ry[m] - xc[m]) -
                            Minv[k,m]*(rx[k] - xc[k])*deltaS[x]*(ry[m] - xc[m])
                        end
                        end
                    end

                end
            end

        end
    end
    C = C1 + C2

    # w + T*w + 4*pi*C*w + B*w = F
    #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
    w = (Matrix(1.0I, size(T)) + T + (lambda-1)/2*C + (lambda-1)/(8*pi)*B) \ F

    # w' = C*w
    # v = w + (lambda-1)/2 * w'
    varr = w + (lambda-1)/2 * C*w


    velocities = reshape(varr ,3,size(vertices,2))
    return velocities
end

function scalar_magnetic_potential(r::Array{Float64}, M::Float64, angle::Float64,  xlims::Array{Float64}, ylims::Array{Float64}, zlims::Array{Float64})
    # Scalar magnetic potential of a block magnet, bounded by lims (according to https://contattafiles.s3.us-west-1.amazonaws.com/tnt41611/uDDJpl0uRpMsYLf/Ravaud_Lemarquand_2009_Magnetic%20field%20produced%20by%20a%20parallelepipedic%20magnet%20of%20various%20and%20uniform.pdf)
    # R. Ravaud and G. Lemarquand "MAGNETIC FIELD PRODUCED BY A PARALLELEPIPEDIC MAGNET OF VARIOUS AND UNIFORM POLARIZATION"
    # M - magnetization, angle wrt to X axis in XY plane

    function dist(i::Int64, j::Int64, k::Int64, r::Array{Float64}, xlims::Array{Float64}, ylims::Array{Float64}, zlims::Array{Float64})
        # distance function wrt to given faces of the block magnet
        x, y, z = r
        xi, yj, zk = xlims[i], ylims[j], zlims[k]

        return sqrt((x-xi)^2 + (y-yj)^2 + (z-zk)^2)
    end

    function phi_1(i::Int64, j::Int64, k::Int64, r::Array{Float64}, xlims::Array{Float64}, ylims::Array{Float64}, zlims::Array{Float64})
        # helper function, see ref above
        x, y, z = r
        xi, yj, zk = xlims[i], ylims[j], zlims[k]
        d = dist(i, j, k, r, xlims, ylims, zlims)

        return zk + (x-xi)*atan((z-zk)/(x-xi)) + (y-yj)*log(z-zk+d) -
                    (x-xi)*atan( (y-yj)*(z-zk) / ((x-xi)*d) ) + (z-zk)*log(y-yj+d)
    end

    function phi_2(i::Int64, j::Int64, k::Int64, r::Array{Float64}, xlims::Array{Float64}, ylims::Array{Float64}, zlims::Array{Float64})
        # helper function, see ref above
        x, y, z = r
        xi, yj, zk = xlims[i], ylims[j], zlims[k]
        d = dist(i, j, k, r, xlims, ylims, zlims)

        return zk + (y-yj)*atan((z-zk)/(y-yj)) + (x-xi)*log(z-zk+d) -
                    (y-yj)*atan( (x-xi)*(z-zk) / ((y-yj)*d) ) + (z-zk)*log(x-xi+d)
    end

    pot = 0
    for i in 1:2
        for j in 1:2
            for k in 1:2
                pot = pot + (-1)^(i+j+k) * ( cos(angle)*phi_1(i, j, k, r, xlims, ylims, zlims) + sin(angle)*phi_2(i, j, k, r, xlims, ylims, zlims) )

            end
        end
    end

    return pot*M/4pi/(4pi*10^-7)

end

function quadropole_potential(point::Array{Float64})
    x, y, z = point
    M = 1.0
    z_lim = 1.75
    return scalar_magnetic_potential([x, y, z], M, 1pi, [-1.0, -0.5], [0., 0.5], [-z_lim/2, z_lim/2]) +
                 scalar_magnetic_potential([x, y, z], M, 0., [-1.0, -0.5], [-0.5, 0.], [-z_lim/2, z_lim/2]) +
                 scalar_magnetic_potential([x, y, z], M, 1pi, [0.5, 1.], [0., 0.5], [-z_lim/2, z_lim/2]) +
                 scalar_magnetic_potential([x, y, z], M, 0., [0.5, 1.], [-0.5, 0.], [-z_lim/2, z_lim/2])
end

function make_arbitrary_force_velocity_lambda1(forces, points, faces, normals; gaussorder=3)
# velocity due to arbitrary (non-singular) force, when viscosity ratio is λ=1
# v_k = 1/(8π) * ∫ f_i G_{ik} dS
    deltaS = make_dS(points,faces)
    N = size(points, 2)
    velocities = zeros(Float64, 3, N)

    function make_n(x,x1,x2,x3,n1,n2,n3) # normal linear interpolation
        A = [x1 x2 x3] # matrix of vertex radiusvecotrs
        B = [n1 n2 n3] # matrix of vertex normals

        zeta_xi_eta = A \ x # find local triangle parameters

        n = B * zeta_xi_eta
        return n/norm(n)
    end

    # jāturpina te

    return velocities
end

function make_L_sing(points, faces, normals; gaussorder=3)
    deltaS = make_dS(points,faces)
    N = size(points, 2)
    L = zeros(Float64, N)

    function make_n(x,x1,x2,x3,n1,n2,n3) # normal linear interpolation
        A = [x1 x2 x3] # matrix of vertex radiusvecotrs
        B = [n1 n2 n3] # matrix of vertex normals

        zeta_xi_eta = A \ x # find local triangle parameters

        n = B * zeta_xi_eta
        return n/norm(n)
    end


    for ykey in 1:N
        function make_L(x,x1,x2,x3,n1,n2,n3; y=points[:,ykey],ny=normals[:,ykey])
            nx = make_n(x,x1,x2,x3,n1,n2,n3)
            r = y - x

            return dot(r, nx-ny) / norm(r)^3 / 4pi
        end

        function make_L_times_norm_r(x,x1,x2,x3,n1,n2,n3; y=points[:,ykey], ny=normals[:,ykey])
            nx = make_n(x,x1,x2,x3,n1,n2,n3)
            r = y - x

            return dot(r, nx-ny) / norm(r)^2 / 4pi
        end

        ny = normals[:, ykey]
        ry = points[:, ykey]

        for i in 1:size(faces, 2)
            #println("triangle: $i")
            if !(ykey in faces[:,i]) # if not singular triangle
                x1, x2, x3 = [points[:, faces[j,i]] for j in 1:3]
                n1, n2, n3 = [normals[:, faces[j,i]] for j in 1:3]

                L[ykey] += gauss_nonsingular(x->make_L(x,x1,x2,x3,n1,n2,n3), x1,x2,x3,gaussorder)
                #println(x1, x2, x3)
            else # if is singular triangle
                singul_ind = findfirst(ind->ind==ykey,faces[:,i])

                x1 = points[:,faces[singul_ind,i]]
                x2 = points[:,faces[(singul_ind) % 3 + 1,i]]
                x3 = points[:,faces[(singul_ind + 1) % 3 + 1,i]]

                n1 = normals[:,faces[singul_ind,i]]
                n2 = normals[:,faces[(singul_ind) % 3 + 1,i]]
                n3 = normals[:,faces[(singul_ind + 1) % 3 + 1,i]]
                #println("L = $L,eeee $(L[ykey])")
                #L[ykey] +=
                L[ykey] += gauss_weaksingular(x->make_L_times_norm_r(x,x1,x2,x3,n1,n2,n3), x1,x2,x3,gaussorder)
                #println("$Ltemp, $(L[ykey]), $ykey")
            end

        end
    end

    return -L # implement the minus in the end. For a sphere L=-1 for all points
end

function make_deltaH_normal(points, faces, normals, mu, H0; gaussorder=3)
    # calculate the jump in the normal H field on drop surface
    # from D. Das and D. Saintillan "Electrohydrodynamics of viscous drops in strong electric fields: numerical simulations"
    alpha = mu
    deltaS = make_dS(points,faces)
    N = size(points, 2)
    L = make_L_sing(points, faces, normals; gaussorder=gaussorder)
    L = Diagonal(alpha/(alpha-1) .- L) # alpha term should be positive
    F = zeros(Float64, N, N) # integral equation matrix
    H = zeros(Float64, N)

    for ykey in 1:N
        ny = normals[:, ykey]
        ry = points[:, ykey]

        for xkey in 1:N
            if xkey == ykey
                continue
            end
            nx = normals[:, xkey]
            rx = points[:, xkey]
            F[ykey, xkey] +=  dot(ny, ry-rx) / norm(ry-rx)^3 * deltaS[xkey] / (4pi)

        end
    end

    for ykey in 1:N
        H[ykey] = dot(H0, normals[:, ykey]) # no minus
    end

    F_reg = F - Diagonal([sum(F[i, :]) for i in 1:N])
    deltaH_normal = (L - F_reg) \ H

    return deltaH_normal'
end

function make_H_tangential(points, faces, normals, CDE, H0, deltaH_normal; gaussorder = 3)
    # return the magnitude of the tangential magnetic field on the droplet surface
    # we use curved elements for singular triangles
    N = size(points, 2)
    Ht = zeros(3, N) # 3xN array
    deltaS = make_dS(points,faces)

    dHn = deltaH_normal

    for ykey in 1:N
        for i in 1:size(faces, 2)
            function make_n(x,x1,x2,x3,n1,n2,n3) # normal linear interpolation
                A = [x1 x2 x3] # matrix of vertex radiusvecotrs
                B = [n1 n2 n3] # matrix of vertex normals

                zeta_xi_eta = A \ x # find local triangle parameters

                n = B * zeta_xi_etafor
                return n/norm(n)
            end

            function make_n_analy(x,x1,x2,x3,n1,CDE)
                # error in this function
                x_p = project_on_given_paraboloid(CDE, n1, x, x1) # x_p is global
                x_l = to_local(x_p - x1, n1)
                n = [-2*CDE[1]*x_l[1] - CDE[2]*x_l[2],
                    -CDE[2]*x_l[1] - 2*CDE[3]*x_l[2],
                    1]
                return n/norm(n)
            end

            function make_dHn(x,x1,x2,x3,dHn1,dHn2,dHn3) # delta normal field linear interpolation
                A = [x1 x2 x3] # matrix of vertex radiusvecotrs
                B = [dHn1 dHn2 dHn3] # matrix of vertex normals

                zeta_xi_eta = A \ x # find local triangle parameters

                deltaHn = dot(B, zeta_xi_eta)
                return deltaHn
            end

            function make_Ht(x,x1,x2,x3,n1,n2,n3,dHn1,dHn2,dHn3,CDE; y=points[:,ykey],ny=normals[:,ykey])
                nx = make_n(x,x1,x2,x3,n1,n2,n3)
                r = y - x
                dHnx = make_dHn(x,x1,x2,x3,dHn1,dHn2,dHn3)
                temp = dHnx * ny - dHn[ykey] * nx
                Ht_func = cross(temp, r) / norm(r)^3

                return Ht_func/(4pi)
            end

            if !(ykey in faces[:,i]) # if not singular triangle
                x1 = points[:,faces[1,i]]
                x2 = points[:,faces[2,i]]
                x3 = points[:,faces[3,i]]

                n1, dHn1 = normals[:,faces[1,i]], dHn[faces[1,i]]
                n2, dHn2 = normals[:,faces[2,i]], dHn[faces[2,i]]
                n3, dHn3 = normals[:,faces[3,i]], dHn[faces[3,i]]

                Ht[:, ykey] += gauss_nonsingular(x->make_Ht(x,x1,x2,x3,n1,n2,n3,dHn1,dHn2,dHn3,CDE[:,ykey]), x1,x2,x3,gaussorder)
                #Ht[:, ykey] += gauss_curved_pol_vec(x->make_Ht(x,x1,x2,x3,n1,n2,n3,dHn1,dHn2,dHn3), x1,x2,x3,n1,CDE[:,faces[1,i]],gaussorder)
            else # if is singular triangle
                # arrange labels so that singularity is on x1
                # (singul_ind + n - 1) % 3 + 1 shifts index by n
                singul_ind = findfirst(singul_ind->singul_ind==ykey,faces[:,i])
                ind2, ind3 = (singul_ind) % 3 + 1, (singul_ind + 1) % 3 + 1

                x1, x2, x3 = points[:,faces[singul_ind,i]], points[:,faces[ind2,i]], points[:,faces[ind3,i]]

                n1, dHn1 = normals[:,faces[singul_ind,i]], dHn[faces[singul_ind,i]]
                n2, dHn2 = normals[:,faces[ind2,i]], dHn[faces[ind2,i]]
                n3, dHn3 = normals[:,faces[ind3,i]], dHn[faces[ind3,i]]
                Ht[:,ykey] += gauss_curved_pol_vec(x->make_Ht(x,x1,x2,x3,n1,n2,n3,dHn1,dHn2,dHn3,CDE[:,ykey]), x1,x2,x3,n1,CDE[:,ykey],gaussorder)
                #Ht[:,ykey] += gauss_flat_pol_vec(x->make_Ht(x,x1,x2,x3,n1,n2,n3,dHn1,dHn2,dHn3,CDE[:,ykey]),x1,x2,x3,gaussorder)
            end # end if singular

        end # end i for

        Ht[:, ykey] += cross(normals[:, ykey], H0)
    end # end ykey for

    Ht_norms2 = sum(Ht.^2, dims=1)
    Ht_norms = sqrt.(Ht_norms2)

    return Ht_norms
end

function make_H_normal(deltaH_normal,mu)
    return deltaH_normal / (mu-1)
end

function PotentialSimple(points,faces,normals,mu,H0;regularize=true)
    # return the magnetic potential psi on the surface of teh drop

    hmag = mu
    A = zeros(Float64,size(points,2),size(points,2))

    vareas = zeros(Float64,size(points,2))
    for i in 1:size(faces,2)
        v1,v2,v3 = faces[:,i]
        area = norm(cross(points[:,v2]-points[:,v1],points[:,v3]-points[:,v1]))/2
        vareas[v1] += area/3
        vareas[v2] += area/3
        vareas[v3] += area/3
    end

    for xkey in 1:size(points,2)

        x = points[:,xkey]
        nx = normals[:,xkey]

        for ykey in 1:size(points,2)
            if xkey==ykey
                continue
            end
            y = points[:,ykey]
            ny = normals[:,ykey]

            A[ykey,xkey] = dot(y-x,ny)/norm(y-x)^3 * vareas[ykey]
        end
    end

    B = zeros(Float64,size(points,2))
    for xkey in 1:size(points,2)
        B[xkey] = 2*dot(H0,points[:,xkey])/(hmag+1)
    end

    if regularize==true
        A = A'
        reg_A = A - Diagonal(Float64[sum(A[i,:]) for i in 1:size(A,2)])
        psi = (I * (1- (hmag-1)/(hmag+1)) - 1/2/pi * (hmag-1)/(hmag+1) * reg_A) \ B
    else
        A = A*(hmag-1)/(hmag+1)/2/pi
        A = A'
        psi = (I - A)\B
    end

    return psi
end

function HField(points,faces,psi)

    H = Array{Float64}(undef,size(points)...)
    for xkey in 1:size(points,2)

        x = points[:,xkey]
        psix = psi[xkey]
        distances = Float64[]
        dphi = Float64[]

        for ykey in NeighborVertices(xkey,faces)
            y = points[:,ykey]
            distances = [distances; y-x]
            dphi = [dphi; psi[ykey]-psix]
        end
        A, B = distances, dphi
        A = transpose(reshape(A,3,div(length(A),3))) ### This looks unecesarry

        # linear least squares
        H[:,xkey] = inv(transpose(A)*A)*transpose(A)*B
    end
    return H
end

function HtField(points, faces, psi, normals)
    # return the tangential component of the magnetic field on the droplet surface
    # returns a vector
    Ht = Array{Float64}(undef,3,size(points,2))
    H = HField(points,faces,psi)

    for xkey in 1:size(points,2)
        nx = normals[:,xkey]
        P = I - nx*nx'
        Ht[:,xkey] = P*H[:,xkey]
    end

    return Ht
end

function NormalFieldCurrent(points,faces,normals,Ht,mu,H0; eps=0.0001)
    # return normal component of magnetic field on the surface of the droplet
    # returns a scalar

    hmag = mu

    for xkey in 1:size(points,2)
        nx = normals[:,xkey]
        #Ht[:,xkey] = (eye(3) - nx*nx')*Ht[:,xkey]
    end

    vareas = zeros(Float64,size(points,2))
    for i in 1:size(faces,2)
        v1,v2,v3 = faces[:,i]
        area = norm(cross(points[:,v2]-points[:,v1],points[:,v3]-points[:,v1])) /2
        vareas[v1] += area/3
        vareas[v2] += area/3
        vareas[v3] += area/3
    end


    function qs(xi,eta,v1,v2,v3,x,nx,Htx)
        y = (1 - xi - eta)*points[:,v1] + xi*points[:,v2] + eta*points[:,v3]
        Hty = (1 - xi - eta)*Ht[:,v1] + xi*Ht[:,v2] + eta*Ht[:,v3]
        ny =  (1 - xi - eta)*normals[:,v1] + xi*normals[:,v2] + eta*normals[:,v3]
        s = - dot(nx,cross(Hty - Htx,cross(ny,-(y-x)/norm(y-x)^2)))
    end

    NP = 10
    tt, ww = gausslegendre(NP)

    function strquad(q::Function,x1,x2,x3)

        B = dot(x3-x1,x2-x1)/norm(x2-x1)^2
        C = norm(x3-x1)^2/norm(x2-x1)^2
        hS = norm(cross(x2-x1,x3-x1))

        s = 0
        for i in 1:NP
            Chi = pi/4*(1 + tt[i])

            R = 1/(cos(Chi) + sin(Chi))
            si = 0
            for j in 1:NP
                rho = R/2*(1 + tt[j])
                si += q(rho*cos(Chi),rho*sin(Chi))*ww[j]
            end

            s += si*R/2 / sqrt(cos(Chi)^2 + B*sin(2*Chi) + C*sin(Chi)^2) * ww[i]
        end
        s *= pi/4

        return s*hS/norm(x2 - x1)
    end

    Hn = Array{Float64}(undef,size(points,2))

    for xkey in 1:size(points,2)

        nx = normals[:,xkey]
        x = points[:,xkey] + eps*nx
        Htx = Ht[:,xkey]

        s = 0
        for ykey in 1:size(points,2)
            !(xkey==ykey) || continue
            y = points[:,ykey]
            ny = normals[:,ykey]
            Hty = Ht[:,ykey]

            #s += dot(nx,-(Hty-Htx)*dot((y-x)/norm(y-x)^3,ny)) * vareas[ykey]
            s += dot(nx,-(Hty-Htx)*dot((y-x)/norm(y-x)^3,ny)) * vareas[ykey]
            s += -dot(nx,cross(Hty-Htx,cross(ny,-(y-x)/norm(y-x)^3))) * vareas[ykey]
            #s += dot(nx,cross(cross(ny,),-(y-x)/norm(y-x)^3)) * vareas[ykey]
        end

        ### Making a proper hole
        for (v2,v3) in DoubleVertexVRing(xkey,faces)
            area = norm(cross(points[:,v2]-x,points[:,v3]-x))/2

            ny = normals[:,v2]
            y = points[:,v2]
            s -= -dot(nx,cross(Ht[:,v2]-Htx,cross(ny,-(y-x)/norm(y-x)^3))) * area/3

            ny = normals[:,v3]
            y = points[:,v3]
            s -= -dot(nx,cross(Ht[:,v3]-Htx,cross(ny,-(y-x)/norm(y-x)^3))) * area/3

            ### Singular triangle integration

            #s += strquad((xi,eta) -> qs(xi,eta,xkey,v2,v3,x,nx,Htx),x,points[:,v2],points[:,v3],abstol=abs(s/100))[1]
            s += strquad((xi,eta) -> qs(xi,eta,xkey,v2,v3,x,nx,Htx),x,points[:,v2],points[:,v3])
        end

        #println("xkey is $xkey")

        Hn[xkey] = dot(H0,nx)/hmag + 1/4/pi * (1-hmag)/hmag * s
    end

    return Hn
end

function PotentialSimple_par(points,faces,normals,mu,H0;regularize=true)
    # return the magnetic potential psi on the surface of teh drop

    hmag = mu
    A = zeros(Float64,size(points,2),size(points,2))

    vareas = zeros(Float64,size(points,2))
    for i in 1:size(faces,2)
        v1,v2,v3 = faces[:,i]
        area = norm(cross(points[:,v2]-points[:,v1],points[:,v3]-points[:,v1]))/2
        vareas[v1] += area/3
        vareas[v2] += area/3
        vareas[v3] += area/3
    end

    Threads.@threads for xkey in 1:size(points,2)

        x = points[:,xkey]
        nx = normals[:,xkey]

        for ykey in 1:size(points,2)
            if xkey==ykey
                continue
            end
            y = points[:,ykey]
            ny = normals[:,ykey]

            A[ykey,xkey] = dot(y-x,ny)/norm(y-x)^3 * vareas[ykey]
        end
    end

    B = zeros(Float64,size(points,2))
    for xkey in 1:size(points,2)
        B[xkey] = 2*dot(H0,points[:,xkey])/(hmag+1)
    end

    if regularize==true
        A = A'
        reg_A = A - Diagonal(Float64[sum(A[i,:]) for i in 1:size(A,2)])
        psi = (I * (1- (hmag-1)/(hmag+1)) - 1/2/pi * (hmag-1)/(hmag+1) * reg_A) \ B
    else
        A = A*(hmag-1)/(hmag+1)/2/pi
        A = A'
        psi = (I - A)\B
    end

    return psi
end

function HtField_par(points, faces, psi, normals)
    # return the tangential component of the magnetic field on the droplet surface
    # returns a vector
    Ht = Array{Float64}(undef,3,size(points,2))
    H = HField(points,faces,psi)

    Threads.@threads for xkey in 1:size(points,2)
        nx = normals[:,xkey]
        P = I - nx*nx'
        Ht[:,xkey] = P*H[:,xkey]
    end

    return Ht
end

function NormalFieldCurrent_par(points,faces,normals,Ht,mu,H0; eps=0.0001)
    # return normal component of magnetic field on the surface of the droplet
    # returns a scalar

    hmag = mu

    for xkey in 1:size(points,2)
        nx = normals[:,xkey]
        #Ht[:,xkey] = (eye(3) - nx*nx')*Ht[:,xkey]
    end

    vareas = zeros(Float64,size(points,2))
    for i in 1:size(faces,2)
        v1,v2,v3 = faces[:,i]
        area = norm(cross(points[:,v2]-points[:,v1],points[:,v3]-points[:,v1])) /2
        vareas[v1] += area/3
        vareas[v2] += area/3
        vareas[v3] += area/3
    end


    function qs(xi,eta,v1,v2,v3,x,nx,Htx)
        y = (1 - xi - eta)*points[:,v1] + xi*points[:,v2] + eta*points[:,v3]
        Hty = (1 - xi - eta)*Ht[:,v1] + xi*Ht[:,v2] + eta*Ht[:,v3]
        ny =  (1 - xi - eta)*normals[:,v1] + xi*normals[:,v2] + eta*normals[:,v3]
        s = - dot(nx,cross(Hty - Htx,cross(ny,-(y-x)/norm(y-x)^2)))
    end

    NP = 10
    tt, ww = gausslegendre(NP)

    function strquad(q::Function,x1,x2,x3)

        B = dot(x3-x1,x2-x1)/norm(x2-x1)^2
        C = norm(x3-x1)^2/norm(x2-x1)^2
        hS = norm(cross(x2-x1,x3-x1))

        s = 0
        for i in 1:NP
            Chi = pi/4*(1 + tt[i])

            R = 1/(cos(Chi) + sin(Chi))
            si = 0
            for j in 1:NP
                rho = R/2*(1 + tt[j])
                si += q(rho*cos(Chi),rho*sin(Chi))*ww[j]
            end

            s += si*R/2 / sqrt(cos(Chi)^2 + B*sin(2*Chi) + C*sin(Chi)^2) * ww[i]
        end
        s *= pi/4

        return s*hS/norm(x2 - x1)
    end

    Hn = Array{Float64}(undef,size(points,2))

    Threads.@threads for xkey in 1:size(points,2)

        nx = normals[:,xkey]
        x = points[:,xkey] + eps*nx
        Htx = Ht[:,xkey]

        s = 0
        for ykey in 1:size(points,2)
            !(xkey==ykey) || continue
            y = points[:,ykey]
            ny = normals[:,ykey]
            Hty = Ht[:,ykey]

            #s += dot(nx,-(Hty-Htx)*dot((y-x)/norm(y-x)^3,ny)) * vareas[ykey]
            s += dot(nx,-(Hty-Htx)*dot((y-x)/norm(y-x)^3,ny)) * vareas[ykey]
            s += -dot(nx,cross(Hty-Htx,cross(ny,-(y-x)/norm(y-x)^3))) * vareas[ykey]
            #s += dot(nx,cross(cross(ny,),-(y-x)/norm(y-x)^3)) * vareas[ykey]
        end

        ### Making a proper hole
        for (v2,v3) in DoubleVertexVRing(xkey,faces)
            area = norm(cross(points[:,v2]-x,points[:,v3]-x))/2

            ny = normals[:,v2]
            y = points[:,v2]
            s -= -dot(nx,cross(Ht[:,v2]-Htx,cross(ny,-(y-x)/norm(y-x)^3))) * area/3

            ny = normals[:,v3]
            y = points[:,v3]
            s -= -dot(nx,cross(Ht[:,v3]-Htx,cross(ny,-(y-x)/norm(y-x)^3))) * area/3

            ### Singular triangle integration

            #s += strquad((xi,eta) -> qs(xi,eta,xkey,v2,v3,x,nx,Htx),x,points[:,v2],points[:,v3],abstol=abs(s/100))[1]
            s += strquad((xi,eta) -> qs(xi,eta,xkey,v2,v3,x,nx,Htx),x,points[:,v2],points[:,v3])
        end

        #println("xkey is $xkey")

        Hn[xkey] = dot(H0,nx)/hmag + 1/4/pi * (1-hmag)/hmag * s
    end

    return Hn
end

function make_magvelocities_par(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2)
    #without the use of levi civita tensor
    # lambda = int viscosity / ext viscosity
    # Returns vertex velocities from the Stokes flow integral equations
    # using Wielandt's deflation as describeed in Zinchenko 1997, Das 2017
    # w + T*w + 4*pi*C*w + B*w = F
    println("calculating velocities")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2


    S = sum(deltaS)
    xc = [0.,0.,0.]
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        xc = xc + rx * deltaS[x] / S
    end

    F = zeros(3*size(vertices,2),1)
    T = zeros(3*size(vertices,2),3*size(vertices,2))
    #C = C1+C2
    C1 = zeros(3*size(vertices,2),3*size(vertices,2))
    C2 = zeros(3*size(vertices,2),3*size(vertices,2))
    B = zeros(3*size(vertices,2),3*size(vertices,2))

    M = zeros(3,3)
    # fill M
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        # Matrix(1.0I, 3, 3) <- 3x3 Identity matrix
        M += (Matrix(1.0I, 3, 3) * dot(rx - xc, rx - xc) -
            (rx - xc) * (rx - xc)') * deltaS[x]
            #   ^ changed the transpose
    end
    Minv = inv(M) # calculate inverse of M

    Threads.@threads for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3

            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                #else
                #polar integration scheme
                end

                for i = 1:3
                    if x != y
                       T[3*y-3+j,3*x-3+i] = -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]

                       T[3*y-3+j,3*y-3+i] -= -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]
                    end

                    B[3*y-3+j,3*x-3+i] = -4*pi/S * normals[j,y] * normals[i,x] * deltaS[x]

                    C1[3*y-3+j,3*x-3+i] =  deltaS[x]/S

                    for k = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,j]*(rx[k] - xc[k])*deltaS[x]*(ry[i] - xc[i]) -
                            Minv[i,j]*(rx[k] - xc[k])*deltaS[x]*(ry[k] - xc[k]) -
                            Minv[k,k]*(rx[j] - xc[j])*deltaS[x]*(ry[i] - xc[i]) +
                            Minv[i,k]*(rx[j] - xc[j])*deltaS[x]*(ry[k] - xc[k])
                    end
                    if i==j
                        for k = 1:3
                        for m = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,k]*(rx[m] - xc[m])*deltaS[x]*(ry[m] - xc[m]) -
                            Minv[k,m]*(rx[k] - xc[k])*deltaS[x]*(ry[m] - xc[m])
                        end
                        end
                    end

                end
            end

        end
    end
    C = C1 + C2

    # w + T*w + 4*pi*C*w + B*w = F
    #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
    w = (Matrix(1.0I, size(T)) + T + (lambda-1)/2*C + (lambda-1)/(8*pi)*B) \ F

    # w' = C*w
    # v = w + (lambda-1)/2 * w'
    varr = w + (lambda-1)/2 * C*w


    velocities = reshape(varr ,3,size(vertices,2))
    return velocities
end


function make_magvelocities_2_par(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2)
    # early exit for lambda=1
    #without the use of levi civita tensor
    # lambda = int viscosity / ext viscosity
    # Returns vertex velocities from the Stokes flow integral equations
    # using Wielandt's deflation as describeed in Zinchenko 1997, Das 2017
    # w + T*w + 4*pi*C*w + B*w = F
    println("calculating velocities")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2


    S = sum(deltaS)
    xc = [0.,0.,0.]
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        xc = xc + rx * deltaS[x] / S
    end

    F = zeros(3*size(vertices,2),1)
    T = zeros(3*size(vertices,2),3*size(vertices,2))
    #C = C1+C2
    C1 = zeros(3*size(vertices,2),3*size(vertices,2))
    C2 = zeros(3*size(vertices,2),3*size(vertices,2))
    B = zeros(3*size(vertices,2),3*size(vertices,2))



    if lambda == 1.
        println("lambda = 1; calculating v directly")
        Threads.@threads for y = 1:size(vertices,2)
            ry = vertices[:,y]

            for j = 1:3

                for x = 1:size(vertices,2)
                    rx = vertices[:,x]

                    Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                    (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                    if x != y
                       F[3*y-3+j] +=
                           1/(8*pi) *(
                           dot(rx-ry,normals[:,x]) * normals[j,y]+
                           dot(rx-ry,normals[:,y]) * normals[j,x]+
                           (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                           3*(rx[j]-ry[j])/norm(rx-ry)^2*
                           dot(normals[:,x]+normals[:,y],rx-ry)*
                           dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                    # magnetic part
                       F[3*y-3+j] += Bm/(8*pi) *
                       (fbar[x] - fbar[y]) *
                       dot(Gji[j,:], normals[:,x]) * deltaS[x]
                    #else
                    #polar integration scheme
                    end
                end

            end
        end
        # w + T*w + 4*pi*C*w + B*w = F
        #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
        w = (Matrix(1.0I, size(T)) + T) \ F

        # w' = C*w
        # v = w + (lambda-1)/2 * w'
        varr = w


        velocities = reshape(varr ,3,size(vertices,2))
        return velocities
    end



    M = zeros(3,3)
    # fill M
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        # Matrix(1.0I, 3, 3) <- 3x3 Identity matrix
        M += (Matrix(1.0I, 3, 3) * dot(rx - xc, rx - xc) -
            (rx - xc) * (rx - xc)') * deltaS[x]
            #   ^ changed the transpose
    end
    Minv = inv(M) # calculate inverse of M

    Threads.@threads for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3

            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                #else
                #polar integration scheme
                end

                for i = 1:3
                    if x != y
                       T[3*y-3+j,3*x-3+i] = -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]

                       T[3*y-3+j,3*y-3+i] -= -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]
                    end

                    B[3*y-3+j,3*x-3+i] = -4*pi/S * normals[j,y] * normals[i,x] * deltaS[x]

                    C1[3*y-3+j,3*x-3+i] =  deltaS[x]/S

                    for k = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,j]*(rx[k] - xc[k])*deltaS[x]*(ry[i] - xc[i]) -
                            Minv[i,j]*(rx[k] - xc[k])*deltaS[x]*(ry[k] - xc[k]) -
                            Minv[k,k]*(rx[j] - xc[j])*deltaS[x]*(ry[i] - xc[i]) +
                            Minv[i,k]*(rx[j] - xc[j])*deltaS[x]*(ry[k] - xc[k])
                    end
                    if i==j
                        for k = 1:3
                        for m = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,k]*(rx[m] - xc[m])*deltaS[x]*(ry[m] - xc[m]) -
                            Minv[k,m]*(rx[k] - xc[k])*deltaS[x]*(ry[m] - xc[m])
                        end
                        end
                    end

                end
            end

        end
    end
    C = C1 + C2

    # w + T*w + 4*pi*C*w + B*w = F
    #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
    w = (Matrix(1.0I, size(T)) + T + (lambda-1)/2*C + (lambda-1)/(8*pi)*B) \ F

    # w' = C*w
    # v = w + (lambda-1)/2 * w'
    varr = w + (lambda-1)/2 * C*w


    velocities = reshape(varr ,3,size(vertices,2))
    return velocities
end

function make_magvelocities_linearflow_par(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2,G,gamma)
    # G = shear rate
    # early exit for lambda=1
    #without the use of levi civita tensor
    # lambda = int viscosity / ext viscosity
    # Returns vertex velocities from the Stokes flow integral equations
    # using Wielandt's deflation as describeed in Zinchenko 1997, Das 2017
    # w + T*w + 4*pi*C*w + B*w = F
    println("calculating velocities")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2


    S = sum(deltaS)
    xc = [0.,0.,0.]
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        xc = xc + rx * deltaS[x] / S
    end

    F = zeros(3*size(vertices,2),1)
    T = zeros(3*size(vertices,2),3*size(vertices,2))
    #C = C1+C2
    C1 = zeros(3*size(vertices,2),3*size(vertices,2))
    C2 = zeros(3*size(vertices,2),3*size(vertices,2))
    B = zeros(3*size(vertices,2),3*size(vertices,2))



    if lambda == 1.
        println("lambda = 1; calculating v directly")
        Threads.@threads for y = 1:size(vertices,2)
            ry = vertices[:,y]

            for j = 1:3
                F[3*y-3+j] += G* dot(gamma[:,j],ry) # dont forget to change also in lambda != 1

                for x = 1:size(vertices,2)
                    rx = vertices[:,x]

                    Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                    (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                    if x != y
                       F[3*y-3+j] +=
                           1/(8*pi) *(
                           dot(rx-ry,normals[:,x]) * normals[j,y]+
                           dot(rx-ry,normals[:,y]) * normals[j,x]+
                           (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                           3*(rx[j]-ry[j])/norm(rx-ry)^2*
                           dot(normals[:,x]+normals[:,y],rx-ry)*
                           dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                    # magnetic part
                       F[3*y-3+j] += Bm/(8*pi) *
                       (fbar[x] - fbar[y]) *
                       dot(Gji[j,:], normals[:,x]) * deltaS[x]
                    #else
                    #polar integration scheme
                    end
                end

            end
        end
        # w + T*w + 4*pi*C*w + B*w = F
        #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
        w = (Matrix(1.0I, size(T)) + T) \ F

        # w' = C*w
        # v = w + (lambda-1)/2 * w'
        varr = w


        velocities = reshape(varr ,3,size(vertices,2))
        return velocities
    end



    M = zeros(3,3)
    # fill M
    for x = 1:size(vertices,2)
        rx = vertices[:,x]
        # Matrix(1.0I, 3, 3) <- 3x3 Identity matrix
        M += (Matrix(1.0I, 3, 3) * dot(rx - xc, rx - xc) -
            (rx - xc) * (rx - xc)') * deltaS[x]
            #   ^ changed the transpose
    end
    Minv = inv(M) # calculate inverse of M

    Threads.@threads for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3
            F[3*y-3+j] += G* dot(gamma[:,j],ry) # dont forget to change also in lambda = 1
            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                #else
                #polar integration scheme
                end

                for i = 1:3
                    if x != y
                       T[3*y-3+j,3*x-3+i] = -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]

                       T[3*y-3+j,3*y-3+i] -= -(1-lambda)/(8*pi)*
                           (-6*(rx[i]-ry[i])*(rx[j]-ry[j]) / norm(rx-ry)^5 )*
                           dot(rx-ry,normals[:,x])*
                           deltaS[x]
                    end

                    B[3*y-3+j,3*x-3+i] = -4*pi/S * normals[j,y] * normals[i,x] * deltaS[x]

                    C1[3*y-3+j,3*x-3+i] =  deltaS[x]/S

                    for k = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,j]*(rx[k] - xc[k])*deltaS[x]*(ry[i] - xc[i]) -
                            Minv[i,j]*(rx[k] - xc[k])*deltaS[x]*(ry[k] - xc[k]) -
                            Minv[k,k]*(rx[j] - xc[j])*deltaS[x]*(ry[i] - xc[i]) +
                            Minv[i,k]*(rx[j] - xc[j])*deltaS[x]*(ry[k] - xc[k])
                    end
                    if i==j
                        for k = 1:3
                        for m = 1:3
                        C2[3*y-3+j,3*x-3+i] +=
                            Minv[k,k]*(rx[m] - xc[m])*deltaS[x]*(ry[m] - xc[m]) -
                            Minv[k,m]*(rx[k] - xc[k])*deltaS[x]*(ry[m] - xc[m])
                        end
                        end
                    end

                end
            end

        end
    end
    C = C1 + C2

    # w + T*w + 4*pi*C*w + B*w = F
    #w = (eye(size(T)) + T + 4*pi*C + B) \ F; old and bad
    w = (Matrix(1.0I, size(T)) + T + (lambda-1)/2*C + (lambda-1)/(8*pi)*B) \ F

    # w' = C*w
    # v = w + (lambda-1)/2 * w'
    varr = w + (lambda-1)/2 * C*w


    velocities = reshape(varr ,3,size(vertices,2))
    return velocities
end

function make_magvelocities_linearflow_no_Wie(vertices, normals, lambda, Bm, mu, Hn_2, Ht_2,G,gamma)
    # Returns vertex velocities from the Stokes flow integral equations without Wielandt's deflation
    # v = Av + F
    println("calculation velocities, no Wieldandt")
    deltaS = make_dS(points,faces)

    #fbar = mu*(mu-1)/8pi * Hfieldn.^2 + (mu-1)/8pi * Hfieldt.^2
    fbar = mu*(mu-1)/8pi * Hn_2 + (mu-1)/8pi * Ht_2

    F = zeros(3*size(vertices,2),1)
    A = zeros(3*size(vertices,2), 3*size(vertices,2))

    for y = 1:size(vertices,2)
        ry = vertices[:,y]

        for j = 1:3
            F[3*y-3+j] += G* dot(gamma[:,j],ry) # linear flow

            for x = 1:size(vertices,2)
                rx = vertices[:,x]

                Gji = (Matrix(1.0I, (3,3)) / norm(rx-ry) +
                (rx-ry) * (rx-ry)' / norm(rx-ry)^3)

                if x != y
                   F[3*y-3+j] +=
                       1/(8*pi) *(
                       dot(rx-ry,normals[:,x]) * normals[j,y]+
                       dot(rx-ry,normals[:,y]) * normals[j,x]+
                       (1-dot(normals[:,x],normals[:,y])) * (rx[j]-ry[j])-
                       3*(rx[j]-ry[j])/norm(rx-ry)^2*
                       dot(normals[:,x]+normals[:,y],rx-ry)*
                       dot(rx-ry,normals[:,x])) * deltaS[x]/norm(rx-ry)^3

                # magnetic part
                   F[3*y-3+j] += Bm/(8*pi) *
                   (fbar[x] - fbar[y]) *
                   dot(Gji[j,:], normals[:,x]) * deltaS[x]
                end

                # build the A matrix
                for i = 1:3
                    if x != y
                        A[3y-3+j, 3x-3+i] = (1-lambda)/8/pi * (-6 * (rx[i]-ry[i])*(rx[j]-ry[j])) / norm(rx-ry)^5 *
                            dot(rx-ry, normals[:,x]) * deltaS[x]
                        A[3y-3+j, 3y-3+i] += (1-lambda)/8/pi * (-6 * (rx[i]-ry[i])*(rx[j]-ry[j])) / norm(rx-ry)^5 *
                            dot(rx-ry, normals[:,x]) * deltaS[x]


                    end # if x!=y
                end # i for
            end # x for
        end # j for
    end # y for

    varr = (A-I) \ F;  # solve integral eq. for velocity

    velocities = reshape(varr ,3,size(vertices,2))
    return velocities

end # function
