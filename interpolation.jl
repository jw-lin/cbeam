# interpolation module for finite element meshes

using NPZ
using PythonCall

### bounding volume hierarchy (BVH) for faster point location ###

eps = 1e-12 # wiggle room for machine (im)precision

struct leaf
    bbox :: Array{Float64,1}
    idxs :: Vector{Int32}
end

struct idxtree
    left :: Union{idxtree,leaf,Nothing}
    right :: Union{idxtree,leaf,Nothing}
    bbox :: Union{Array{Float64,1},Nothing} # bbox: xmin,xmax,ymin,ymax
end

struct tritree 
    _idxtree :: idxtree
    points :: Array{Float64,2} # array of all points in the mesh
    triidxs :: Array{Int64,2} # list of integer indices for each quadratic triangle element
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
    # assume node traversal costs 1 and point-tri intersect costs 2
    return 1 + area_A*num_tris_A*2 + area_B*num_tris_B*2
end

function minimize_SAH(cmins::Vector{Float64},cmaxs::Vector{Float64})
    start_minidx = Int(ceil(length(cmins)/2))
    j = 1
    sgn = 1
    best_min_SAH = compute_SAH(cmins,cmaxs,cmins[start_minidx])
    # do a shitty gradient descent
    while true
        SAH = compute_SAH(cmins,cmaxs,cmins[start_minidx+sgn*j])
        if SAH < best_min_SAH
            best_min_SAH = SAH
            j+=1
        elseif sgn ==1 & j==1
            sgn=-1
        else
            break
        end
    end
    return start_minidx + sgn*j
end

function computeBB(xmins,xmaxs,ymins,ymaxs)
    return [minimum(xmins),maximum(xmaxs),minimum(ymins),maximum(ymaxs)]
end

function computeBB(bounds::Matrix{Float64})::Vector{Float64}
    return [minimum(bounds[:,1]),maximum(bounds[:,2]),minimum(bounds[:,3]),maximum(bounds[:,4])]
end

function construct_recursive(tripoints::Array{Float64,3},idxs::Array{Int64,1},bounds::Array{Float64,2},level::Int64,leaf_size::Int64)
    BB = computeBB(bounds)
    _tripoints = tripoints[idxs,:,:]
    if size(_tripoints,1) <= leaf_size
        return leaf(BB,idxs)
    else
        ax = level%2+1
        sort_idx = sortperm(bounds[:,(ax-1)*2+1])
        cmins = bounds[sort_idx,2*ax-1]
        cmaxs = bounds[sort_idx,2*ax]

        split_idx = minimize_SAH(cmins,cmaxs)
        left_idxs = sort_idx[1:split_idx]
        right_idxs = sort_idx[split_idx+1:end]

        if !isempty(left_idxs)                
            left_bounds = bounds[left_idxs,:]
            left_tree = construct_recursive(tripoints,idxs[left_idxs],left_bounds,level+1,leaf_size)
        else
            left_tree = nothing
        end

        if !isempty(right_idxs)
            right_bounds = bounds[right_idxs,:]
            right_tree = construct_recursive(tripoints,idxs[right_idxs],right_bounds,level+1,leaf_size)
        else
            right_tree = nothing
        end
        return idxtree(left_tree,right_tree,BB)
    end
end

function construct_tritree(points::Union{Array{Float64,2},PyArray{Float64,2}},triidxs::Union{Array{Int64,2},PyArray{Int64,2}},leaf_size=8)
    if typeof(points) <: PyArray
        points = pyconvert(Array{Float64,2},points)
    end
    if typeof(triidxs) <: PyArray
        triidxs = pyconvert(Array{Int64,2},triidxs)
    end
    tripoints = points[triidxs,:]
    xmins = minimum(tripoints[:,:,1],dims=2)
    xmaxs = maximum(tripoints[:,:,1],dims=2)
    ymins = minimum(tripoints[:,:,2],dims=2)
    ymaxs = maximum(tripoints[:,:,2],dims=2)
    bounds = hcat(xmins,xmaxs,ymins,ymaxs)

    idxs = Array(1:size(tripoints,1))
    return tritree(construct_recursive(tripoints,idxs,bounds,0,leaf_size),points,triidxs)
end 

function inside(point::Vector{Float64},bbox::Vector{Float64})
    return (bbox[1]<=point[1]<=bbox[2]) & (bbox[3]<=point[2]<=bbox[4])
end

function det(u::Vector{Float64},v::Vector{Float64})
    return u[1]*v[2] - u[2]*v[1]
end

function inside(point::Vector{Float64},tri::Array{Float64,2})
    v1 = tri[1,:]
    v2 = tri[2,:] - tri[1,:]
    v3 = tri[3,:] - tri[1,:]

    a = (det(point,v3) - det(v1,v3)) / det(v2,v3)
    b = -(det(point,v2) - det(v1,v2)) / det(v2,v3)

    return (a>=-eps) & (b>=-eps) & (a+b<=1+eps)
end

function query_recursive(point::Union{Vector{Float64},PyArray{Float64,1}},points::Array{Float64,2},triidxs::Array{Int64,2},_idxtree::Union{idxtree,leaf,Nothing})
    if typeof(point) <: PyArray
        point = pyconvert(Vector{Float64},point)
    end
    if isnothing(_idxtree)
        return nothing
    elseif !inside(point,_idxtree.bbox)
        return nothing
    elseif isleaf(_idxtree)
        for idx in _idxtree.idxs
            tri = points[triidxs[idx,:],:]
            if inside(point,tri)
                return idx
            end
        end
        return nothing
    else
        left_query = query_recursive(point,points,triidxs,_idxtree.left)
        if !isnothing(left_query)
            return left_query
        end
        right_query = query_recursive(point,points,triidxs,_idxtree.right)
        if !isnothing(right_query)
            return right_query
        else
            return nothing
        end
    end
end

function query(point::Union{Vector{Float64},PyArray{Float64,1}},_tritree::tritree)
    if typeof(point) <: PyArray
        point = pyconvert(Vector{Float64},point)
    end
    return query_recursive(point,_tritree.points,_tritree.triidxs,_tritree._idxtree)
end

### finite element shape functions ###

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

function make_interpolation_matrix(points::Array{Float64,2},tree::tritree)
    N = size(points,1)
    mat = zeros(Float64,(N,N))
    for i in 1:N
        point = points[i,:]
        idx = query(point,tree)
        if isnothing(idx)
            continue
        end
        _triidx = tree.triidxs[idx,:]
        verts = tree.points[_triidx,:]
        u,v = apply_affine_transform(verts,point)

        mat[_triidx[1],_triidx[1]] += N1(u,v)
        mat[_triidx[2],_triidx[2]] += N2(u)
        mat[_triidx[3],_triidx[3]] += N3(v)
        mat[_triidx[4],_triidx[4]] += N4(u,v)
        mat[_triidx[5],_triidx[5]] += N5(u,v)
        mat[_triidx[6],_triidx[6]] += N6(u,v)
    end
    return mat
end

# interpolate field, sampled on mesh corresponding to tree, to new points
function unstructured_interpolate(field::Vector{Float64},tree::tritree,points::Array{Float64,2})
    N = size(points,1)
    out = zeros(Float64,N)
    for i in 1:N
        point = points[i,:]
        idx = query(point,tree)
        if isnothing(idx)
            continue
        end
        _triidxs = tree.triidxs[idx,:]
        verts = tree.points[_triidxs,:]
        u,v = apply_affine_transform(verts,point)
        out[i] += field[_triidxs[1]]*N1(u,v)
        out[i] += field[_triidxs[2]]*N2(u)
        out[i] += field[_triidxs[3]]*N3(v)
        out[i] += field[_triidxs[4]]*N4(u,v)
        out[i] += field[_triidxs[5]]*N5(u,v)
        out[i] += field[_triidxs[6]]*N6(u,v)
    end
    return out
end