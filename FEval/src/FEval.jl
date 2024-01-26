module FEval

# simple bounding volume hierarchy (BVH) tree for triangles
using PythonCall
using Cubature
using GrundmannMoeller
using StaticArrays

struct leaf
    bbox :: Array{Float64,1}
    idxs :: Vector{Int32}
end

struct idxtree
    left :: Union{idxtree,leaf,Nothing}
    right :: Union{idxtree,leaf,Nothing}
    bbox :: Union{Array{Float64,1},Nothing} # bbox: xmin,xmax,ymin,ymax
end

struct tritree # just need a way to store the array of triangle points
    _idxtree :: idxtree
    points :: Array{Float64,2}
    connections :: Array{Int64,2}
    tripoints :: Array{Float64,3}
end

function isleaf(a::Union{leaf,idxtree})
    return typeof(a) == leaf
end

function compute_SAH(cmins::Vector{Float64},cmaxs::Vector{Float64},csplit::Float64)
    total_interval = cmaxs[end]-cmins[1]
    area_A = (csplit-cmins[1])/total_interval
    area_B = 1 - area_A
    num_tris_A = sum(cmins .< csplit)
    num_tris_B = sum(cmaxs .> csplit)
    # assume node traversal costs 1 and point-tri intersect costs 4 
    #(empirical comparison between bbox test and triangle test)
    return 1 + area_A*num_tris_A*4 + area_B*num_tris_B*4
end

function minimize_SAH(cmins::Vector{Float64},cmaxs::Vector{Float64},num_buckets=8)
    start_minidx = Int(ceil(length(cmins)/2))
    N = size(cmins,1)
    if N <= 8
        return start_minidx # for small amounts of triangles SAH min isn't worth
    end
    default_cost = 4*N
    best_min_SAH = compute_SAH(cmins,cmaxs,cmins[start_minidx])

    splits = LinRange(cmins[1],cmins[end],num_buckets+2)[2:end-1]
    best_i = 4
    for i in 1:num_buckets
        s = splits[i]
        SAH = compute_SAH(cmins,cmaxs,s)
        if SAH <= best_min_SAH
            best_min_SAH = SAH
            best_i = i
        end
    end
    if best_min_SAH < default_cost
        return clamp(searchsortedfirst(cmins,splits[best_i])-1,1,N)
    else
        return nothing
    end
end

function computeBB(xmins,xmaxs,ymins,ymaxs)
    return [minimum(xmins),maximum(xmaxs),minimum(ymins),maximum(ymaxs)]
end

function computeBB(bounds::Matrix{Float64})::Vector{Float64}
    return [minimum(bounds[:,1]),maximum(bounds[:,2]),minimum(bounds[:,3]),maximum(bounds[:,4])]
end

function construct_recursive(idxs::Array{Int64,1},bounds::Array{Float64,2},level::Int64,min_leaf_size::Int64)
    BB = computeBB(bounds)
    if size(idxs,1) <= min_leaf_size
        return leaf(BB,idxs)
    else
        ax = level%2+1
        sort_idx = sortperm(bounds[:,(ax-1)*2+1])
        cmins = bounds[sort_idx,2*ax-1]
        cmaxs = bounds[sort_idx,2*ax]

        split_idx = minimize_SAH(cmins,cmaxs)
        if isnothing(split_idx)
            return leaf(BB,idxs)
        end

        left_idxs = sort_idx[1:split_idx]
        right_idxs = sort_idx[split_idx+1:end]

        if !isempty(left_idxs)                
            left_bounds = bounds[left_idxs,:]
            left_tree = construct_recursive(idxs[left_idxs],left_bounds,level+1,min_leaf_size)
        else
            left_tree = nothing
        end

        if !isempty(right_idxs)
            right_bounds = bounds[right_idxs,:]
            right_tree = construct_recursive(idxs[right_idxs],right_bounds,level+1,min_leaf_size)
        else
            right_tree = nothing
        end
        return idxtree(left_tree,right_tree,BB)
    end
end

function construct_tritree(points::PyArray{Float64,2},connections::PyArray{T,2} where T<:Integer,min_leaf_size=4)
    points = pyconvert(Array{Float64,2},points)
    connections = pyconvert(Array{UInt64,2},connections)

    tripoints = Array{Float64,3}(undef,size(connections,1),size(connections,2),2)
    for i in axes(connections,1)
        for j in axes(connections,2)
            tripoints[i,j,1] = points[connections[i,j],1]
            tripoints[i,j,2] = points[connections[i,j],2]
        end
    end

    xmins = minimum(tripoints[:,:,1],dims=2)
    xmaxs = maximum(tripoints[:,:,1],dims=2)
    ymins = minimum(tripoints[:,:,2],dims=2)
    ymaxs = maximum(tripoints[:,:,2],dims=2)
    bounds = hcat(xmins,xmaxs,ymins,ymaxs)

    idxs = Array(1:size(tripoints,1))
    return tritree(construct_recursive(idxs,bounds,0,min_leaf_size),points,connections,tripoints)
end 

function modify_idx_tree_recursive(_idxtree::Union{idxtree,leaf},scale_factor::Float64)
    _idxtree.bbox .*= scale_factor
    if isleaf(_idxtree)
        return
    end
    if !isnothing(_idxtree.left)
        modify_idx_tree_recursive(_idxtree.left,scale_factor)
    end
    if !isnothing(_idxtree.right)
        modify_idx_tree_recursive(_idxtree.right,scale_factor)
    end
end

function update_tritree(_tritree::tritree,scale_factor::Float64)
    _tritree.points .*= scale_factor # not sure if this will accumulate machine error
    _tritree.tripoints .*= scale_factor
    # traverse the tree and modify the bounding boxes
    modify_idx_tree_recursive(_tritree._idxtree,scale_factor)
end

function inside(point::Vector{Float64},bbox::Vector{Float64})
    return (bbox[1]<=point[1]<=bbox[2]) & (bbox[3]<=point[2]<=bbox[4])
end

function det(u::Vector{Float64},v::Vector{Float64})
    return u[1]*v[2] - u[2]*v[1]
end

function inside(point::Vector{Float64},tri::Array{Float64,2},_eps=1e-12)::Bool
    x,y = point
    dot1 = (tri[2,2]-tri[1,2])*(x-tri[1,1]) + (tri[1,1]-tri[2,1])*(y-tri[1,2])
    if dot1 > _eps
        return false
    end
    dot2 = (tri[3,2]-tri[2,2])*(x-tri[2,1]) + (tri[2,1]-tri[3,1])*(y-tri[2,2])
    if dot2 > _eps
        return false
    end
    dot3 = (tri[1,2]-tri[3,2])*(x-tri[3,1]) + (tri[3,1]-tri[1,1])*(y-tri[3,2])
    if dot3 > _eps
        return false
    end
    return true
end

function query_recursive(point::Union{Vector{Float64},PyArray{Float64,1}},tripoints::Array{Float64,3},_idxtree::Union{idxtree,leaf,Nothing}) :: Int64
    if typeof(point) <: PyArray
        point = pyconvert(Vector{Float64},point)
    end
    if isnothing(_idxtree)
        return 0
    elseif !inside(point,_idxtree.bbox)
        return 0
    elseif isleaf(_idxtree)
        for idx in _idxtree.idxs
            tri = tripoints[idx,:,:]
            if inside(point,tri)
                return idx
            end
        end
        return 0
    else
        left_query = query_recursive(point,tripoints,_idxtree.left)
        if left_query != 0
            return left_query
        end
        right_query = query_recursive(point,tripoints,_idxtree.right)
        if right_query != 0
            return right_query
        else
            return 0
        end
    end
end

function query(point::Union{Vector{Float64},PyArray{Float64,1}},_tritree::tritree) :: Int64
    if typeof(point) <: PyArray
        point = pyconvert(Vector{Float64},point)
    end
    return query_recursive(point,_tritree.tripoints,_tritree._idxtree)
end

### interpolation stuff ##

function N1(u,v)
    2 * (1 - u - v) * (0.5 - u - v)
end
function N2(u)
    2 * u * (u - 0.5)
end
function N3(v)
    2 * v * (v - 0.5)
end
function N4(u,v)
    4 * u * (1 - u - v)
end
function N5(u,v)
    4 * u * v
end
function N6(u,v)
    4 * v * (1 - u - v)
end

function affine_transform_matrix(vertices::Array{Float64,2})
    x21 = vertices[2,1] - vertices[1,1]
    y21 = vertices[2,2] - vertices[1,2]
    x31 = vertices[3,1] - vertices[1,1]
    y31 = vertices[3,2] - vertices[1,2]
    _J = x21*y31-x31*y21
    M = [y31 -x31 ; -y21 x21] ./ _J
    return M
end

function affine_transform_matrix_inv(vertices::Array{Float64,2})
    x21 = vertices[2,1] - vertices[1,1]
    y21 = vertices[2,2] - vertices[1,2]
    x31 = vertices[3,1] - vertices[1,1]
    y31 = vertices[3,2] - vertices[1,2]
    return [x21 x31 ; y21 y31]
end

function apply_affine_transform(vertices::Array{Float64,2},xy::Vector{Float64})
    M = affine_transform_matrix(vertices)
    return M * (xy .- vertices[1,:])
end

function apply_affine_transform_inv(vertices::Array{T,2},uv::Union{Vector{T},SVector{2,T}}) where T <: Real
    Mi = affine_transform_matrix_inv(vertices)
    return Mi * uv .+ vertices[1,:]
end

function jac(vertices::Array{Float64,2})
    x21 = vertices[2,1] - vertices[1,1]
    y21 = vertices[2,2] - vertices[1,2]
    x31 = vertices[3,1] - vertices[1,1]
    y31 = vertices[3,2] - vertices[1,2]
    return x21*y31-x31*y21
end

function get_interp_weights(new_points::PyArray{Float64,2},_tritree::tritree)
    _old_points = _tritree.tripoints
    _new_points = pyconvert(Array{Float64,2},new_points)
    N = size(_new_points,1)
    _weights = Array{Float64}(undef,N,6)
    _triidxs = Array{Int64}(undef,N)

    for i in 1:N
        new_point = _new_points[i,1:2]
        _triidx = query(new_point,_tritree)
        if _triidx != 0
            _triidxs[i] = _triidx
            triverts = _old_points[_triidx,:,:]
            u,v = apply_affine_transform(triverts,new_point)

            _weights[i,1] = N1(u,v)
            _weights[i,2] = N2(u)
            _weights[i,3] = N3(v)
            _weights[i,4] = N4(u,v)
            _weights[i,5] = N5(u,v)
            _weights[i,6] = N6(u,v)
        else
            _weights[i,:] .= 0.
            _triidxs[i] = 0
        end
    end
    return (_triidxs,_weights)
end

function evaluate(point::Union{Vector{Float64},PyArray{Float64,1}},field::Union{PyArray{T,1},Vector{T}},_tritree::tritree) :: T where T<:Union{Float64,ComplexF64}
    dtype = eltype(field)
    if typeof(point) <: PyArray
        point = pyconvert(Vector{Float64},point)
    end
    if typeof(field) <: PyArray
        field = pyconvert(Vector{dtype},field)
    end
    _triidx = query(point,_tritree)
    val = 0.
    if _triidx != 0
        triverts = _tritree.tripoints[_triidx,:,:]
        u,v = apply_affine_transform(triverts,point)
        val += N1(u,v) * field[_tritree.connections[_triidx,1]]
        val += N2(u) * field[_tritree.connections[_triidx,2]]
        val += N3(v) * field[_tritree.connections[_triidx,3]]
        val += N4(u,v) * field[_tritree.connections[_triidx,4]]
        val += N5(u,v) * field[_tritree.connections[_triidx,5]]
        val += N6(u,v) * field[_tritree.connections[_triidx,6]]
    end
    return val
end

function evaluate(point::PyArray{Float64,2},field::Union{PyArray{T,1},Vector{T}},_tritree::tritree) :: T where T<:Union{Float64,ComplexF64}
    dtype = eltype(field)
    if typeof(point) <: PyArray
        point = pyconvert(Vector{Float64},point)
    end
    if typeof(field) <: PyArray
        field = pyconvert(Vector{dtype},field)
    end
    out = Vector{dtype}(undef,size(points,1))

    for i in axes(point,1)
        _point = point[i,:]
        _triidx = query(_point,_tritree)
        val = 0.
        if _triidx != 0
            triverts = _tritree.tripoints[_triidx,:,:]
            u,v = apply_affine_transform(triverts,_point)
            val += N1(u,v) * field[_tritree.connections[_triidx,1]]
            val += N2(u) * field[_tritree.connections[_triidx,2]]
            val += N3(v) * field[_tritree.connections[_triidx,3]]
            val += N4(u,v) * field[_tritree.connections[_triidx,4]]
            val += N5(u,v) * field[_tritree.connections[_triidx,5]]
            val += N6(u,v) * field[_tritree.connections[_triidx,6]]
        end
        out[i] = val
    end
    return out
end

function evaluate(pointsx::PyArray{Float64,1},pointsy::PyArray{Float64,1},field::PyArray{T,1} where T<:Union{Float64,ComplexF64},_tritree::tritree)

    dtype = eltype(field)

    pointsx = pyconvert(Vector{Float64},pointsx)
    pointsy = pyconvert(Vector{Float64},pointsy)
    field = pyconvert(Vector{dtype},field)

    out = Array{dtype,2}(undef,size(pointsx,1),size(pointsy,1))

    for i in eachindex(pointsx)
        for j in eachindex(pointsy)
            point = [pointsx[i],pointsy[j]]
            _triidx = query(point,_tritree)
            val = 0.
            if _triidx != 0
                triverts = _tritree.tripoints[_triidx,:,:]
                u,v = apply_affine_transform(triverts,point)
                val += N1(u,v) * field[_tritree.connections[_triidx,1]]
                val += N2(u) * field[_tritree.connections[_triidx,2]]
                val += N3(v) * field[_tritree.connections[_triidx,3]]
                val += N4(u,v) * field[_tritree.connections[_triidx,4]]
                val += N5(u,v) * field[_tritree.connections[_triidx,5]]
                val += N6(u,v) * field[_tritree.connections[_triidx,6]]
            end
            out[i,j] = val
        end
    end
    return out
end

function evaluate_func(field::PyArray{T,1} where T<:Union{Float64,ComplexF64},_tritree::tritree)
    """ convert a FE field into a function of point [x,y] """
    dtype = eltype(field)
    field = pyconvert(Vector{dtype},field)

    function _inner_(point::Union{Vector{Float64},PyArray{Float64,1}})
        if typeof(point) <: PyArray
            point = pyconvert(Vector{Float64},point)
        end
        return evaluate(point,field,_tritree)
    end
    return _inner_
end

function evaluate(f::Function,xa::PyArray{Float64,1},ya::PyArray{Float64,1})
    xa = pyconvert(Vector{Float64},xa)
    ya = pyconvert(Vector{Float64},ya)
    out = Array{Float64}(undef,size(xa,1),size(ya,1))
    for i in eachindex(xa)
        for j in eachindex(ya)
            x = xa[i]
            y = ya[j]
            out[i,j] = f([x,y])
        end
    end
    return out
end


function diff_func(field1::PyArray{T,1},tree1::tritree,field2::PyArray{T,1},tree2::tritree) where T<:Union{Float64,ComplexF64}
    """ compute the difference (not derivative) function between two fields, field2-field1 """
    dtype = eltype(field1)
    field1 = pyconvert(Vector{dtype},field1)
    field2 = pyconvert(Vector{dtype},field2)

    function _inner_(point::Union{Vector{Float64},PyArray{Float64,1}})
        val1 = evaluate(point,field1,tree1)
        val2 = evaluate(point,field2,tree2)
        return val2-val1
    end
    return _inner_
end

function diff_func(field1::PyArray{T,2},tree1::tritree,field2::PyArray{T,2},tree2::tritree) where T<:Union{Float64,ComplexF64}
    """ vectorized version of diff_func. second axis is along separate fields. """
    dtype = eltype(field1)
    field1 = pyconvert(Array{dtype,2},field1)
    field2 = pyconvert(Array{dtype,2},field2)

    funcs = Vector{Function}(undef,size(field1,2))
    for j in axes(field1,2)
        function _inner_(point::Union{Vector{Float64},PyArray{Float64,1}})
            val1 = evaluate(point,field1[:,j],tree1)
            val2 = evaluate(point,field2[:,j],tree2)
            return val2-val1
        end
        funcs[j] = _inner_
    end
    return funcs
end

function avg_func(field1::PyArray{T,1},tree1::tritree,field2::PyArray{T,1},tree2::tritree) where T<:Union{Float64,ComplexF64}
    """ compute the average function between two fields, 0.5*(field2+field1) """
    dtype = eltype(field1)
    field1 = pyconvert(Vector{dtype},field1)
    field2 = pyconvert(Vector{dtype},field2)

    function _inner_(point::Union{Vector{Float64},PyArray{Float64,1}})
        val1 = evaluate(point,field1,tree1)
        val2 = evaluate(point,field2,tree2)
        return (val1+val2)/2.
    end
    return _inner_
end

function avg_func(field1::PyArray{T,2},tree1::tritree,field2::PyArray{T,2},tree2::tritree) where T<:Union{Float64,ComplexF64}
    """ vectorized version of avg_func. second axis is along separate fields. """
    dtype = eltype(field1)
    field1 = pyconvert(Array{dtype,2},field1)
    field2 = pyconvert(Array{dtype,2},field2)
    funcs = Vector{Function}(undef,size(field1,2))
    for  j in axes(field1,2)
        function _inner_(point::Union{Vector{Float64},PyArray{Float64,1}})
            val1 = evaluate(point,field1[:,j],tree1)
            val2 = evaluate(point,field2[:,j],tree2)
            return (val1+val2)/2.
        end
        funcs[j] = _inner_
    end
    return funcs
end

function inner_product(func1::Function,func2::Function,xmin::PyArray{Float64,1},xmax::PyArray{Float64,1},tol::Float64)
    """ return inner product between two functions of [x,y] """
    xmin = pyconvert(Vector{Float64},xmin)
    xmax = pyconvert(Vector{Float64},xmax)
    integrand = point -> func1(point)*func2(point)
    (val, err) = pcubature(integrand, xmin, xmax; reltol=1e-2, abstol=1e-6, maxevals=0)
    return val
end

function inner_product(func1::Vector{Function},func2::Vector{Function},xmin::PyArray{Float64,1},xmax::PyArray{Float64,1},tol::Float64)
    """ return inner product matrix between two sets of functions of [x,y], integration in polar coords """
    out = Array{ComplexF64}(undef,size(func1,1),size(func2,1))
    for i in axes(func1,1)
        for j in axes(func2,1)
            if i == j
                out[i,j] = 0.
                continue
            end
            out[i,j] = inner_product(func1[i],func2[j],xmin,xmax,tol)
        end
    end
    return out
end

function inner_product_polar(func1::Function,func2::Function,rmax::Float64,tol::Float64)
    """ return inner product between two functions of [x,y], integration in polar coords """
    function integrand(point::Vector{Float64})
        x = point[1]*cos(point[2])
        y = point[1]*sin(point[2])
        p = [x,y]
        return func1(p)*func2(p)*point[1] 
    end
    xmin = [0.,0.]
    xmax = [rmax,2*pi]
    (val, err) = hcubature(integrand, xmin, xmax; reltol=tol, abstol=0, maxevals=0)
    return val
end

function inner_product_polar(func1::Vector{Function},func2::Vector{Function},rmax::Float64,tol::Float64)
    """ return inner product matrix between two sets of functions of [x,y], integration in polar coords """
    out = Array{ComplexF64}(undef,size(func1,1),size(func2,1))
    for i in axes(func1,1)
        for j in axes(func2,1)
            out[i,j] = inner_product_polar(func1[i],func2[j],rmax,tol)
        end
    end
    return out
end

function integrate_product(field1::Vector{Float64},tree1::tritree,field2::Vector{Float64},tree2::tritree)
    """ integrate the product field1 * field2 over the mesh triangles of field1 """
    tot = 0.
    for i in axes(tree1.connections,1)
        idxs = tree1.connections[i,:]
        triverts = tree1.tripoints[i,:,:]
        fps = field1[idxs]

        function F(u,v)
            f1 = fps[1]*N1(u,v) + fps[2]*N2(u) + fps[3]*N3(v) + fps[4]*N4(u,v) + fps[5]*N5(u,v) + fps[6]*N6(u,v)
            xy = apply_affine_transform_inv(triverts,[u,v])
            f2 = evaluate(xy,field2,tree2)
            return f1*f2
        end

        (_int,_err) = pcubature(r->pcubature(s->F(s[1],r[1]), [0], [1-r[1]] ; reltol=1e-3,abstol=0)[1], [0], [1] ; reltol=1e-3,abstol=0)
        tot += _int * jac(triverts)
    end
    return tot
end

function integrate_product_simplex(field1::Vector{Float64},tree1::tritree,field2::Vector{Float64},tree2::tritree,scheme)
    """ integrate the product field1 * field2 over the mesh triangles of field1, using simplicial cubature package """

    tot = 0.
    for i in axes(tree1.connections,1)
        idxs = tree1.connections[i,:]
        triverts = tree1.tripoints[i,:,:]
        fps = field1[idxs]
        function F(uv)
            u = uv[1]
            v = uv[2]
            f1 = fps[1]*N1(u,v) + fps[2]*N2(u) + fps[3]*N3(v) + fps[4]*N4(u,v) + fps[5]*N5(u,v) + fps[6]*N6(u,v)
            xy = apply_affine_transform_inv(triverts,uv)
            f2 = evaluate(xy,field2,tree2)
            return f1*f2
        end
        uv_verts = [[0,0],[1,0],[0,1]]
        _int = integrate(F,scheme,uv_verts)
        tot += _int * jac(triverts)
    end
    return tot
end


function compute_coupling_pcube(field1::PyArray{Float64,2},tree1::tritree,field2::PyArray{Float64,2},tree2::tritree)
    """ estimate cross-coupling matrix """

    out = Array{Float64}(undef,size(field1,2),size(field2,2))

    for j in axes(field1,2)
        for k in 1:j
            if j == k
                out[j,k] = 0
                continue
            end
            val = 0.5 * (integrate_product(field1[:,j],tree1,field2[:,k],tree2) - integrate_product(field1[:,k],tree1,field2[:,j],tree2))
            out[j,k] = val
            out[k,j] = -val
        end
    end
    return out
end

function compute_coupling_simplex(field1::PyArray{Float64,2},tree1::tritree,field2::PyArray{Float64,2},tree2::tritree)
    """ estimate cross-coupling matrix """

    scheme = grundmann_moeller(Float64, Val(2), 5)
    out = Array{Float64}(undef,size(field1,2),size(field2,2))

    for j in axes(field1,2)
        for k in 1:j
            if j == k
                out[j,k] = 0
                continue
            end
            val = 0.5 * (integrate_product_simplex(field1[:,j],tree1,field2[:,k],tree2,scheme) - integrate_product_simplex(field1[:,k],tree1,field2[:,j],tree2,scheme))
            out[j,k] = val
            out[k,j] = -val
        end
    end
    return out
end

function FE_dot(field1::PyArray{Float64,1},tree1::tritree,field2::PyArray{Float64,1},tree2::tritree)
    """ dot product of FE fields """
    field1 = pyconvert(Array{Float64,1},field1)
    field2 = pyconvert(Array{Float64,1},field2)
    scheme = grundmann_moeller(Float64, Val(2), 5)
    tot = 0.
    for i in axes(tree1.connections,1)
        idxs = tree1.connections[i,:]
        triverts = tree1.tripoints[i,:,:]
        fps = field1[idxs]
        function F(uv)
            u = uv[1]
            v = uv[2]
            f1 = fps[1]*N1(u,v) + fps[2]*N2(u) + fps[3]*N3(v) + fps[4]*N4(u,v) + fps[5]*N5(u,v) + fps[6]*N6(u,v)
            xy = apply_affine_transform_inv(triverts,uv)
            f2 = evaluate(xy,field2,tree2)
            return f1*f2
        end
        uv_verts = [[0,0],[1,0],[0,1]]
        _int = integrate(F,scheme,uv_verts)
        tot += _int * jac(triverts)
    end
    return tot
end

function IOR_func(tree::tritree,IORvals::Vector{Float64},IORidxbounds::Array{UInt,2})
    function _inner_(point::Union{Vector{Float64},PyArray{Float64,1}})
        idx = query(point,tree)
        for i in eachindex(IORvals)
            if IORidxbounds[i,1] <= idx <= IORidxbounds[i,end]
                return IORvals[i]
            end
        end
        return minimum(IORvals)
    end
    return _inner_
end

function IORsq_diff_func(tree1::tritree,tree2::tritree,IORvals::Vector{Float64},IORidxbounds1::Array{UInt64,2},IORidxbounds2::Array{UInt64,2})
    f1 = IOR_func(tree1,IORvals,IORidxbounds1)
    f2 = IOR_func(tree2,IORvals,IORidxbounds2)
    function _inner_(point::Union{Vector{Float64},PyArray{Float64,1}})
        return f2(point)^2-f1(point)^2
    end
    return _inner_
end

function integrate_simplex(tree::tritree,field1::Vector{Float64},field2::Vector{Float64},func::Function,scheme)
    tot = 0.
    for i in axes(tree.connections,1)
        idxs = tree.connections[i,:]
        triverts = tree.tripoints[i,:,:]
        fps1 = field1[idxs]
        fps2 = field2[idxs]
        
        function F(uv)
            u = uv[1]
            v = uv[2]
            f1 = fps1[1]*N1(u,v) + fps1[2]*N2(u) + fps1[3]*N3(v) + fps1[4]*N4(u,v) + fps1[5]*N5(u,v) + fps1[6]*N6(u,v)
            f2 = fps2[1]*N1(u,v) + fps2[2]*N2(u) + fps2[3]*N3(v) + fps2[4]*N4(u,v) + fps2[5]*N5(u,v) + fps2[6]*N6(u,v)
            xy = apply_affine_transform_inv(triverts,uv)
            f = func(xy)
            return f1*f2*f
        end

        uv_verts = [[0,0],[1,0],[0,1]]
        #_int = integrate(F,scheme,uv_verts)
        (_int,_err) = hcubature(r->hcubature(s->F([s[1],r[1]]), [0], [1-r[1]] ; reltol=1e-3,abstol=0)[1], [0], [1] ; reltol=1e-3,abstol=1e-6)
        tot += _int * jac(triverts)
    end
    return tot
end

function compute_coupling_pert(field::PyArray{Float64,2},tree1::tritree,IORvals::PyArray{Float64,1},IORidxbounds1::PyArray{UInt64,2},IORidxbounds2::PyArray{UInt64,2},tree2::tritree)

    field = pyconvert(Array{Float64,2},field)
    IORvals = pyconvert(Vector{Float64},IORvals)
    IORidxbounds1 = pyconvert(Array{UInt64,2},IORidxbounds1)
    IORidxbounds2 = pyconvert(Array{UInt64,2},IORidxbounds2)

    scheme = grundmann_moeller(Float64, Val(2), 7)
    Idiff = IORsq_diff_func(tree1,tree2,IORvals,IORidxbounds1,IORidxbounds2)

    out = Array{Float64}(undef,size(field,1),size(field,1))
    for i in axes(field,1)
        for j in axes(field,1)
            out[i,j] = integrate_simplex(tree1,field[i,:],field[j,:],Idiff,scheme)
        end
    end
    return out
end

function compute_cob(field1::PyArray{Float64,2},tree1::tritree,field2::PyArray{Float64,2},tree2::tritree)
    """ estimate change of basis matrix """

    scheme = grundmann_moeller(Float64, Val(2), 11)
    out = Array{Float64}(undef,size(field1,2),size(field2,2))

    for j in axes(field1,2)
        for k in axes(field2,2)
            val = integrate_product_simplex(field1[:,j],tree1,field2[:,k],tree2,scheme) 
            out[j,k] = val
        end
    end
    return out
end

end # module FEval
