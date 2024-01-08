import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gmsh

class HandlePrint():
    def __init__(self):
        self.initial_stout = sys.stderr

    def blockPrint(self):
        sys.stderr = open(os.devnull, 'w')
        
    # Restore
    def resetPrint(self):
        sys.stderr = self.initial_stout

handleprint = HandlePrint()

def boolean_fragment(geom:pygmsh.occ.Geometry,object,tool):
    """ fragment the tool and the object, and return the fragments in the following order:
        intersection, object_fragment, tool_fragment.
        in some cases one of the later two may be empty
    """
    object_copy = geom.copy(object)
    tool_copy = geom.copy(tool)
    try:
        intersection = geom.boolean_intersection([object_copy,tool_copy])
    except:
        # no intersection - make first element None to signal
        return [None,object,tool]

    object = geom.boolean_difference(object,intersection,delete_first=True,delete_other=False)
    tool = geom.boolean_difference(tool,intersection,delete_first=True,delete_other=False)
    return intersection+object+tool

class prim2D:
    """ a prim2D (2D primitive) is an an array of N (x,y) points, shape (N,2), that denote a closed curve (so, a polygon). 
        inside the closed curve, the primitive has refractive index n. 
    """
    def __init__(self,points,n):
        self.points = points
        self.n = n
        self.res = points.shape[0]
        self.center = None
    
    def make_poly(self):
        with pygmsh.occ.Geometry() as geom:
            poly = geom.add_polygon(self.points)
        return poly
    
    def make_points(self):
        """ this function will generate the polygon as a function of some parameters """
        pass

    def update(self,points,n):
        self.points = points
        self.n = n

    def boundary_dist(self,x,y):
        """ this function computes the distance between the point (x,y) and the boundary of the primitive """
        pass

class circle(prim2D):
    """ a circle primitive, defined by radius, center, and number of sides """
    
    def __init__(self,radius,res,n,center=(0,0),angle=0,resolve_inner=False,mesh_size=None):
        super().__init__(self.make_points(radius,res,center,angle=angle),n)
        self.radius = radius
        self.center = center
        self.resolve_inner = resolve_inner
        self.mesh_size=mesh_size

    def make_points(self,radius,res,center=(0,0),angle=0):
    
        thetas = np.linspace(angle,2*np.pi+angle,res,endpoint=False)
        points = []
        for t in thetas:
            points.append((radius*np.cos(t)+center[0],radius*np.sin(t)+center[1]))
        
        points = np.array(points)
        return points
    
    def boundary_dist(self, x, y):
        if self.resolve_inner:
            return max(0,np.sqrt(np.power(x-self.center[0],2)+np.power(y-self.center[1],2)) - self.radius)
        return np.abs(np.sqrt(np.power(x-self.center[0],2)+np.power(y-self.center[1],2)) - self.radius)

class wedge(prim2D):
    """ pie slice - div is an int, how many slices fit into a circle """
    def __init__(self,radius,div,res,n):
        super().__init__(self.make_points(radius,res,div,res),n)
    
    def make_points(self,radius,div,res):
        thetas = np.linspace(0,2*np.pi/div,res+1,endpoint=True)
        points = [(0,0)]
        for t in thetas:
            points.append((radius*np.cos(t),radius*np.sin(t)))
        points = np.array(points)
        return points

class prim3D:
    """ a prim3D (3D primitive) is a function of z that returns a prim2D. """

    def __init__(self,_prim2D:prim2D,label:str):
        self._prim2D = _prim2D
        self.label = label

    def update(self,z):
        pass

    def make_poly_at_z(self,z):
        self.update(z)
        return self._prim2D.make_poly()

class pipe(prim3D):
    """ a pipe is a 3D primitive with circular cross section at all z. """
    def __init__(self,rfunc,nfunc,cfunc,resfunc,label,angle=0,resolve_inner=False,mesh_size=None):
        """ args: functions for radius, index, and center. last arg is polygon resolution. """
        _circ= circle(rfunc(0),resfunc(0),nfunc(0),cfunc(0),angle=angle,resolve_inner=resolve_inner,mesh_size=mesh_size)
        self.rfunc = rfunc
        self.resfunc = resfunc
        self.nfunc = nfunc
        self.cfunc = cfunc
        super().__init__(_circ,label)
    
    def update(self,z):
        points = self._prim2D.make_points(self.rfunc(z),self.resfunc(z),self.cfunc(z))
        self._prim2D.update(points,self.nfunc(z))

def linear_taper(final_scale,z_ex):
    def _inner_(z):
        return (final_scale - 1)/z_ex * z + 1
    return _inner_

class waveguide:
    skip_layers=[0]
    
    def __init__(self,prim3Dgroups):
        self.prim3Dgroups = prim3Dgroups # these are "layers" for the optical structure. each layer overwrites the next.
        self.IOR_dict = {}
    
    def update(self,z):
        for group in self.prim3Dgroups:
            for prim in group:
                prim.update(z)

    def make_mesh(self,algo=6):
        with pygmsh.occ.Geometry() as geom:
            # make the polygons
            polygons = [[geom.add_polygon(p._prim2D.points) for p in group] for group in self.prim3Dgroups]

            # diff the polygons
            for i in range(0,len(self.prim3Dgroups)-1):

                polys = polygons[i]
                _polys = polygons[i+1]

                for _p in _polys:
                    polys = geom.boolean_difference(polys,_p,delete_other=False,delete_first=True)
            
            for i,el in enumerate(polygons):
                geom.add_physical(el,self.prim3Dgroups[i][0].label)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            return mesh
    
    def make_mesh_bndry_ref(self,algo=6,min_mesh_size=0.05,max_mesh_size=1.,size_scale_fac=0.25,_power=1,_align=False,writeto=None):

        with pygmsh.occ.Geometry() as geom:
            
            # flat array of all 2D primitives, skipping layers as needed
            prims = []
            for i,group in enumerate(self.prim3Dgroups):
                if i in self.skip_layers:
                    continue
                for p in group:
                    prims.append(p._prim2D)

            # make the polygons
            polygons = []
            for i,group in enumerate(self.prim3Dgroups):
                polygroup = []
                for j,p in enumerate(group):
                    if _align and i==1 and j==0:
                        # split the cladding
                        N = len(p._prim2D.points)
                        points0 = p._prim2D.points[:int(N/2)+1]
                        points1 = np.concatenate((p._prim2D.points[int(N/2):],np.array([p._prim2D.points[0]])))
                        polygroup.append(geom.add_polygon(points0))
                        polygroup.append(geom.add_polygon(points1))
                    else:
                        polygroup.append(geom.add_polygon(p._prim2D.points))
                polygons.append(polygroup)

            #polygons = [[geom.add_polygon(p._prim2D.points) for p in group] for group in self.prim3Dgroups]

            # diff the polygons
            for i in range(0,len(self.prim3Dgroups)-1):

                polys = polygons[i]
                _polys = polygons[i+1]

                for _p in _polys:
                    polys = geom.boolean_difference(polys,_p,delete_other=False,delete_first=True)

            for i,el in enumerate(polygons):
                geom.add_physical(el,self.prim3Dgroups[i][0].label)

            # mesh refinement callback
            def callback(dim,tag,x,y,z,lc):
                dists = np.array([p.boundary_dist(x,y) for p in prims])
                mesh_sizes = np.array([min_mesh_size if p.mesh_size is None else p.mesh_size for p in prims])
                #min_dist = min([p.boundary_dist(x,y) for p in prims])
                _size = np.min(np.power(dists,_power) * size_scale_fac + mesh_sizes)
                return min(_size,max_mesh_size)

            geom.env.removeAllDuplicates()
            geom.set_mesh_size_callback(callback)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            if writeto is not None:
                gmsh.write(writeto+".msh")
            gmsh.clear()
            return mesh

    def assign_IOR(self):
        for group in self.prim3Dgroups:
            self.IOR_dict[group[0].label] = group[0]._prim2D.n
        return self.IOR_dict

    def plot_mesh(self,mesh=None,IOR_dict=None, verts=3,alpha=0.3):
        """ plot a mesh and associated refractive index distribution
        Args:
        mesh: the mesh to be plotted. if None, we auto-compute
        IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
        """
        
        fig,ax = plt.subplots(figsize=(5,5))

        if mesh is None:
            mesh = self.make_mesh()
        if IOR_dict is None:
            IOR_dict = self.assign_IOR()
        points = mesh.points
        els = mesh.cells[1].data
        materials = mesh.cell_sets.keys()

        IORs = [ior[1] for ior in IOR_dict.items()]
        n,n0 = max(IORs) , min(IORs)

        for material in materials:       
            cval = IOR_dict[material]/(n-n0) - n0/(n-n0)
            cm = plt.get_cmap("inferno")
            color = cm(cval)

            _els = els[tuple(mesh.cell_sets[material])][0,:,0,:]
            for i,_el in enumerate(_els):
                t=plt.Polygon(points[_el[:verts]][:,:2], facecolor=color)
                t_edge=plt.Polygon(points[_el[:verts]][:,:2], lw=0.5,color='0.5',alpha=alpha,fill=False)
                ax.add_patch(t)
                ax.add_patch(t_edge)


        for point in points:
            plt.plot(point[0],point[1],color='0.5',marker='o',ms=1.5,alpha=alpha)

        plt.xlim(np.min(points[:,0]),np.max(points[:,0]) )
        plt.ylim(np.min(points[:,1]),np.max(points[:,1]) )
        plt.show()
    
    def plot_boundaries(self):
        for group in self.prim3Dgroups:
            for prim in group:
                p = prim._prim2D.points
                p2 = np.zeros((p.shape[0]+1,p.shape[1]))
                p2[:-1] = p[:]
                p2[-1] = p[0]
                plt.plot(p2.T[0],p2.T[1])
        plt.axis('equal')
        plt.show()
    
    def make_intersection_mesh(self,z,dz):
        
        IOR_dict={}

        skip_layers = self.skip_layers
        with pygmsh.occ.Geometry() as geom:
            # make the polygons at z

            self.update(z)
            polygons = [[geom.add_polygon(p._prim2D.points) for p in group] for group in self.prim3Dgroups]

            # make the polygons at z + dz
            self.update(z+dz)
            next_polygons = []
            for i,layer in enumerate(self.prim3Dgroups):
                if i in skip_layers:
                    next_polygons.append([])
                    continue
                next_polygons.append([geom.add_polygon(p._prim2D.points) for p in layer])

            elmnts = []

            #fragment each pair of polygons
            for i in range(len(polygons)):
                if i in skip_layers:
                    elmnts.append(polygons[i])
                    continue

                pieces = []
                for j in range(len(polygons[i])):
                    _p = polygons[i][j]
                    _np = next_polygons[i][j]
                    piece = boolean_fragment(geom,_p,_np)
                    pieces.append(piece)

                elmnts.append(pieces)

            #boolean subtraction of layers 
            for i in range(len(elmnts)-1):
                group = elmnts[i]
                _group = elmnts[i+1]
                for j,fragroup in enumerate(group):
                    for _fragroup in _group:
                        idx = 0
                        _idx = 0
                        if type(fragroup) == list and fragroup[0] is None:
                            idx = 1
                        if type(_fragroup) == list and _fragroup[0] is None:
                            _idx = 1

                        if type(fragroup) == list:
                            if type(_fragroup) == list:
                                fragroup[idx:] = geom.boolean_difference(fragroup[idx:],_fragroup[_idx:],delete_first=True,delete_other=False)
                            else:
                                fragroup[idx:] = geom.boolean_difference(fragroup[idx:],_fragroup,delete_first=True,delete_other=False)
                        else:
                            if type(_fragroup) == list:
                                fragroup = geom.boolean_difference(fragroup,_fragroup[_idx:],delete_first=True,delete_other=False)
                            else:
                                fragroup = geom.boolean_difference(fragroup,_fragroup,delete_first=True,delete_other=False)

            #add labels

            for i,layer in enumerate(elmnts):
                fragroup0 = []
                fragroup1 = []
                fragroup01 = [] #intersection group
                for j,sublayer in enumerate(layer):
                    if type(sublayer) == list:
                        # lists represent contain two or three fragments, formed by the union of the 3D primitives evaled at z and z + dz

                        if len(sublayer)==2:
                            fragroup01.append(sublayer[0])
                            fragroup1.append(sublayer[1])
                        elif len(sublayer)==3:
                            if sublayer[0] is not None:
                                fragroup01.append(sublayer[0])
                            fragroup0.append(sublayer[1])
                            fragroup1.append(sublayer[2])
                        else:
                            raise Exception("wrong number of fragments produced by geom.boolean_fragments()")
                        
                    else:
                        # if the sublayer is just a polygon, its structure is assumed to be fixed from z -> z + dz
                        geom.add_physical(layer,self.prim3Dgroups[i][0].label)
                        IOR_dict[self.prim3Dgroups[i][0].label] = self.prim3Dgroups[i][0]._prim2D.n
                        break
                
                if fragroup0:
                    geom.add_physical(fragroup0,self.prim3Dgroups[i][0].label+"0")
                    IOR_dict[self.prim3Dgroups[i][0].label+"0"] = self.prim3Dgroups[i][0]._prim2D.n
                if fragroup1:
                    geom.add_physical(fragroup1,self.prim3Dgroups[i][0].label+"1")
                    IOR_dict[self.prim3Dgroups[i][0].label+"1"] = self.prim3Dgroups[i-1][0]._prim2D.n
                if fragroup01:
                    geom.add_physical(fragroup01,self.prim3Dgroups[i][0].label+"2")
                    IOR_dict[self.prim3Dgroups[i][0].label+"2"] = self.prim3Dgroups[i][0]._prim2D.n

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=6)
        
        return mesh,IOR_dict
    
    def advance_IOR(self,IOR_dict):
        IORs = [g[0]._prim2D.n for g in self.prim3Dgroups]
        for key in IOR_dict.keys():
            if key[-1]=="1":
                IOR_dict[key] = IOR_dict[key[:-1]+"2"]
            if key[-1]=="0":
                idx = IORs.index(IOR_dict[key])
                IOR_dict[key] = IORs[idx-1]
        return IOR_dict
    
    def backstep_IOR_half(self,IOR_dict):
        IORs = [g[0]._prim2D.n for g in self.prim3Dgroups]
        for key in IOR_dict.keys():
            if key[-1]=="1":
                idx = IORs.index(IOR_dict[key])
                IOR_dict[key] = 0.5*(IOR_dict[key[:-1]+"2"]+IORs[idx-1])
            if key[-1]=="0":
                idx = IORs.index(IOR_dict[key])
                IOR_dict[key] = 0.5*(IOR_dict[key[:-1]+"2"]+IORs[idx])
        return IOR_dict        

class linear_lantern(waveguide):
    def __init__(self,core_pos,rcores,rclad,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res,corecirc_res = 10,clad_res=16,jack_res=32):

        self.skip_layers=[0,2]
        taper_func = linear_taper(taper_factor,z_ex)
        self.taper_func = taper_func
        #resfunc = lambda z: int(np.ceil(core_res*np.sqrt(taper_func(z))))
        resfunc = lambda z: core_res
        cores = []

        def rfunc(r):
            def _inner_(z):
                return taper_func(z)*r
            return _inner_
        
        def nfunc(n):
            def _inner_(z):
                return n
            return _inner_

        def cfunc(c):
            def _inner_(z):
                return taper_func(z)*c[0],taper_func(z)*c[1]
            return _inner_

        i = 0
        for c,r,n in zip(core_pos,rcores,ncores):
            if i == 0:
                angle = 0
            else:
                angle = 2*np.pi/5*i
                i+=1
            cores.append(pipe(rfunc(r),nfunc(n),cfunc(c),resfunc,"core",angle=angle))
        
        corecircs = []
        corecirc_resfunc = lambda z: corecirc_res
        i = 0
        for c,r,n in zip(core_pos,rcores,ncores):
            if i == 0:
                angle = 0
            else:
                angle = 2*np.pi/5*i
                i+=1
            corecircs.append(pipe(rfunc(r*4),nfunc(nclad),cfunc(c),corecirc_resfunc,"_cladding",angle=angle))

        cladrfunc = lambda z: taper_func(z)*rclad
        cladnfunc = lambda z: nclad
        cladcfunc = lambda z: (0,0)
        cladresfunc = lambda z: clad_res
        _clad = pipe(cladrfunc,cladnfunc,cladcfunc,cladresfunc,"cladding")

        jackrfunc = lambda z: taper_func(z)*rjack
        jacknfunc = lambda z: njack
        jackcfunc = lambda z: (0,0)
        jackresfunc = lambda z: jack_res
        _jack = pipe(jackrfunc,jacknfunc,jackcfunc,jackresfunc,"jacket")

        els = [[_jack],[_clad],corecircs,cores]
        
        super().__init__(els)
    
class linear_lantern2(waveguide):
    def __init__(self,core_pos,rcores,rclad,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res,clad_res=64,jack_res=32,core_mesh_size=0.05,clad_mesh_size=0.2):

        self.skip_layers=[0]
        taper_func = linear_taper(taper_factor,z_ex)
        self.taper_func = taper_func
        #resfunc = lambda z: int(np.ceil(core_res*np.sqrt(taper_func(z))))
        resfunc = lambda z: core_res
        cores = []

        def rfunc(r):
            def _inner_(z):
                return taper_func(z)*r
            return _inner_
        
        def nfunc(n):
            def _inner_(z):
                return n
            return _inner_

        def cfunc(c):
            def _inner_(z):
                return taper_func(z)*c[0],taper_func(z)*c[1]
            return _inner_

        i = 0
        for c,r,n in zip(core_pos,rcores,ncores):
            if i == 0:
                angle = 0
            else:
                angle = 2*np.pi/5*i
                i+=1
            cores.append(pipe(rfunc(r),nfunc(n),cfunc(c),resfunc,"core",angle=angle,resolve_inner=True,mesh_size=core_mesh_size))

        cladrfunc = lambda z: taper_func(z)*rclad
        cladnfunc = lambda z: nclad
        cladcfunc = lambda z: (0,0)
        cladresfunc = lambda z: clad_res
        _clad = pipe(cladrfunc,cladnfunc,cladcfunc,cladresfunc,"cladding",resolve_inner=True,mesh_size=clad_mesh_size)

        jackrfunc = lambda z: taper_func(z)*rjack
        jacknfunc = lambda z: njack
        jackcfunc = lambda z: (0,0)
        jackresfunc = lambda z: jack_res
        _jack = pipe(jackrfunc,jacknfunc,jackcfunc,jackresfunc,"jacket")

        els = [[_jack],[_clad],cores]
        
        super().__init__(els)