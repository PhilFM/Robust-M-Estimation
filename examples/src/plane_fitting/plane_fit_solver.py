import numpy as np
import open3d as o3d

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")
    sys.path.append("../../../pypi_package/src/gnc_smoothie_philfm/linear_model")
    sys.path.append("../../../pypi_package/src/gnc_smoothie_philfm/cython_files")

from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch

from plane_fit_orthog_welsch import PlaneFitOrthogWelsch

def show_3d(data: np.array, final_weight: np.array, mesh_size, plane_vertices) -> None:
    assert(len(data) == mesh_size*mesh_size)

    triangles = np.zeros((2*(mesh_size-1)*(mesh_size-1), 3), dtype=np.int32)
    for row in range(mesh_size - 1):
        for col in range(mesh_size - 1):
            # 1st triangle
            triangles[2*(row*(mesh_size-1)+col)] = [row * mesh_size + col,
                                                    (row + 1) * mesh_size + col + 1,
                                                    row * mesh_size + col + 1]

            # 2nd triangle
            triangles[2*(row*(mesh_size-1)+col)+1] = [row * mesh_size + col,
                                                      (row + 1) * mesh_size + col,
                                                      (row + 1) * mesh_size + col + 1]

    # 3. Create TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    max_weight = max(final_weight)
    vertex_colors = np.array([[w/max_weight, 0.0, 1.0-w/max_weight] for w in final_weight])
    #mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # View the mesh
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, front=[-0.15,-0.7,0.3]) # mesh_show_wireframe=True

    # add plane mesh
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(np.reshape(plane_vertices, (len(plane_vertices),3)))
    plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    plane_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    #plane_mesh.paint_uniform_color([0.4, 0.4, 1])
    plane_mesh.compute_vertex_normals()

    # View the plane mesh
    o3d.visualization.draw_geometries([plane_mesh], mesh_show_back_face=True, front=[-0.15,-0.7,0.3]) # mesh_show_wireframe=True

def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y,z] triplets
    plane_gt = [0.5, 0.2, -1.0]

    # first put all points on a mesh
    mesh_size = 9
    sigma_pop = 0.06
    xy_range = 8.0
    data = np.zeros((mesh_size*mesh_size,3))
    for i in range(mesh_size):
        y = (-0.5 + i/(mesh_size-1))*xy_range
        for j in range(mesh_size):
            x = (-0.5 + j/(mesh_size-1))*xy_range
            idx = i*mesh_size+j
            data[idx][0] = x
            data[idx][1] = y
            data[idx][2] = plane_gt[0]*data[idx][0] + plane_gt[1]*data[idx][1] + plane_gt[2] + np.random.normal(0.0, sigma_pop)

    # perturb some of the points
    n_bad_points = 5
    for k in range(n_bad_points):
        i = np.random.randint(mesh_size)
        j = np.random.randint(mesh_size)
        idx = i*mesh_size+j
        data[idx][2] = np.random.rand()
    
    # linear regression fitter z = a*x + b*y + c
    p = 0.6667
    sigma = sigma_pop/p
    sigma_limit = np.max(data[:,2]) - np.min(data[:,2])
    linear_regressor = LinearRegressorWelsch(sigma, sigma_limit, 20, use_slow_version=False, debug=True)
    if linear_regressor.run(data):
        final_plane = linear_regressor.final_model
        final_weight = linear_regressor.final_weight
        debug_plane_list = linear_regressor.debug_model_list

    if not test_run:
        print("Linear regression plane result:", final_plane)
        print("   error: ", final_plane-plane_gt)

        # build vertices on the plane
        plane_vertices = np.copy(data)
        for pv in plane_vertices:
            pv[2] = final_plane[0]*pv[0] + final_plane[1]*pv[1] + final_plane[2]

        show_3d(data, final_weight, mesh_size, plane_vertices)

    # change to True if you want to see the progress of the algorithm
    if False: #not test_run:
        print("Intermediate model values:")
        for plane in debug_plane_list:
            if not test_run:
                print("   ",plane)

    # orthogonal regression fitter a*x + b*y + c*z + d = 0 where a^2+b^2+c^2=1
    plane_fitter_orthog = PlaneFitOrthogWelsch(0.01, 50.0, 20, max_niterations=200, debug=True)
    if plane_fitter_orthog.run(data):
        final_plane_orthog = plane_fitter_orthog.final_plane
        debug_plane_list_orthog = plane_fitter_orthog.debug_plane_list

    if not test_run:
        print("Orthogonal regression result: a,b,c,d=", final_plane_orthog)
        plane_orthog = np.array([-final_plane_orthog[0]/final_plane_orthog[2], -final_plane_orthog[1]/final_plane_orthog[2], -final_plane_orthog[3]/final_plane_orthog[2]])
        print("   error: ", plane_orthog-plane_gt)

    # change to True if you want to see the progress of the algorithm
    if False:
        print("Intermediate model values (orthog):")
        for plane in debug_plane_list_orthog:
            if not test_run:
                print("   ",plane)

    if test_run:
        print("plane_fit_solver OK")

if __name__ == "__main__":
    main(False) # test_run
