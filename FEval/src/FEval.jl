module FEval

# simple bounding volume hierarchy (BVH) tree for triangles
using PythonCall

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

function apply_affine_transform(vertices::Array{Float64,2},xy::Vector{Float64})
    M = affine_transform_matrix(vertices)
    return M * (xy .- vertices[1,:])
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

function evaluate(point::PyArray{Float64,1},field::PyArray{T,1} where T<:Union{Float64,ComplexF64},_tritree::tritree)
    dtype = eltype(field)
    point = pyconvert(Vector{Float64},point)
    field = pyconvert(Vector{dtype},field)
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

function evaluate(point::PyArray{Float64,2},field::PyArray{T,1} where T<:Union{Float64,ComplexF64},_tritree::tritree)
    dtype = eltype(field)

    point = pyconvert(Array{Float64,2},point)
    field = pyconvert(Vector{dtype},field)

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

end # module FEval
