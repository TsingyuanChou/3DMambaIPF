
from .quaternion import quaternion_rotate
from .camera import intrinsic_matrix
from .drc import drc_projection, drc_depth_projection
from .vis import save_depth_map_cv2
import os
import math
import numpy as np
# import tensorflow as tf
from functools import reduce
import torch

def euler2mat(z=0, y=0, x=0):
	''' Return matrix for rotations around z, y and x axes

	Uses the z, then y, then x convention above

	Parameters
	----------
	z : scalar
	   Rotation angle in radians around z-axis (performed first)
	y : scalar
	   Rotation angle in radians around y-axis
	x : scalar
	   Rotation angle in radians around x-axis (performed last)

	Returns
	-------
	M : array shape (3,3)
	   Rotation matrix giving same rotation as for given angles

	Examples
	--------
	>>> zrot = 1.3 # radians
	>>> yrot = -0.1
	>>> xrot = 0.2
	>>> M = euler2mat(zrot, yrot, xrot)
	>>> M.shape == (3, 3)
	True

	The output rotation matrix is equal to the composition of the
	individual rotations

	>>> M1 = euler2mat(zrot)
	>>> M2 = euler2mat(0, yrot)
	>>> M3 = euler2mat(0, 0, xrot)
	>>> composed_M = np.dot(M3, np.dot(M2, M1))
	>>> np.allclose(M, composed_M)
	True

	You can specify rotations by named arguments

	>>> np.all(M3 == euler2mat(x=xrot))
	True

	When applying M to a vector, the vector should column vector to the
	right of M.  If the right hand side is a 2D array rather than a
	vector, then each column of the 2D array represents a vector.

	>>> vec = np.array([1, 0, 0]).reshape((3,1))
	>>> v2 = np.dot(M, vec)
	>>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
	>>> vecs2 = np.dot(M, vecs)

	Rotations are counter-clockwise.

	>>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
	>>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
	True
	>>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
	>>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
	True
	>>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
	>>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
	True

	Notes
	-----
	The direction of rotation is given by the right-hand rule (orient
	the thumb of the right hand along the axis around which the rotation
	occurs, with the end of the thumb at the positive end of the axis;
	curl your fingers; the direction your fingers curl is the direction
	of rotation).  Therefore, the rotations are counterclockwise if
	looking along the axis of rotation from positive to negative.
	'''
	Ms = []
	if z:
		cosz = math.cos(z)
		sinz = math.sin(z)
		Ms.append(np.array(
				[[cosz, -sinz, 0],
				 [sinz, cosz, 0],
				 [0, 0, 1]]))
	if y:
		cosy = math.cos(y)
		siny = math.sin(y)
		Ms.append(np.array(
				[[cosy, 0, siny],
				 [0, 1, 0],
				 [-siny, 0, cosy]]))
	if x:
		cosx = math.cos(x)
		sinx = math.sin(x)
		Ms.append(np.array(
				[[1, 0, 0],
				 [0, cosx, -sinx],
				 [0, sinx, cosx]]))
	if Ms:
		return reduce(np.dot, Ms[::-1])
	return np.eye(3)



def pc_perspective_transform(point_cloud, transform, focal_length=None):

    """
    :param point_cloud: [B, N, 3]
    :param transform: [B, 4] if quaternion or [B, 4, 4] if camera matrix
    :param predicted_translation: [B, 3] translation vector
    :return:
    """
    #camera_distance = 2.0  #default相机距离
    #focal_length = 1.875

    camera_distance = 2.0  #default相机距离
    focal_length = 1.0
    pose_quaternion = True

    # transform = tf.constant(transform, dtype=tf.float32)

    if pose_quaternion:
        pc2 = quaternion_rotate(point_cloud, transform) # [V, N, 3]
        # print(f'shape after quaternion_rotate2222: {pc2}')

        xs = pc2[0:, 0:, 2]
        ys = pc2[0:, 0:, 1]
        zs = pc2[0:, 0:, 0]

        # translation part of extrinsic camera
        zs += camera_distance
        # intrinsic transform
        xs *= focal_length
        ys *= focal_length

    xs /= zs
    ys /= zs
    zs -= camera_distance
    xyz2 = torch.cat([zs.unsqueeze(-1), ys.unsqueeze(-1), xs.unsqueeze(-1)], dim=-1)
    return xyz2

skip = 50

x_rot = np.reshape(np.array([ii for ii in range(-180, 180+1, skip)]+\
                        [1 for _ in range(-180, 180+1, skip)]+\
                        [2 for _ in range(-180, 180+1, skip)]+\
                        [45,    -45, -135, 135, 45,  -45, -135, 135])/180.0 *np.pi, [-1,1])
y_rot = np.reshape(np.array([3 for _ in range(-180, 180+1, skip)]+\
                            [ii for ii in range(-180, 180+1, skip)]+\
                            [4 for _ in range(-180, 180+1, skip)]+\
                            [-45,  -135, -135, -45, 45,  135,  135,  45])/180.0 *np.pi, [-1, 1])
z_rot = np.reshape(np.array([5 for _ in range(-180, 180+1, skip)]+\
                            [6 for _ in range(-180, 180+1, skip)]+\
                            [ii for ii in range(-180, 180+1, skip)]+\
                            [-135, -135,  135, 135, -45, -45,  45,   45])/180.0 *np.pi, [-1,1])


trams = np.concatenate([x_rot, y_rot, z_rot, np.ones(x_rot.shape)], 1)
rand_view = np.random.randint(0, trams.shape[0])
trams = np.expand_dims(trams[rand_view], 0)

def pointcloud_project_fast(point_cloud, transform=trams, vox_size=64):
    tr_pc = pc_perspective_transform(point_cloud, transform)
    # print(f'tr_pc.shape: {tr_pc.shape}') # [V, N, 3]

    voxels, _ = pointcloud2voxels3d_fast(tr_pc, vox_size=vox_size)
    # print(f'voxels.shape: {voxels.shape}') # [V, vox_size_z, vox_size, vox_size]

    voxels = torch.unsqueeze(voxels, axis=-1) # [V, vox_size_z, vox_size, vox_size, 1]
    voxels_raw = voxels

    voxels = torch.clamp(voxels, 0.0, 1.0)
    voxels_rgb = None
      
    proj, drc_probs = drc_projection(voxels)
    # print(f'proj shape: {proj.shape}') # [V, vox_size, vox_size, 1]
    drc_probs = torch.flip(drc_probs, [2])
    proj_depth = drc_depth_projection(drc_probs)
    # proj_depth = None

    proj = torch.flip(proj, [1])
    proj_rgb = None

    output_all = {
        "proj": proj,
        "voxels": voxels,
        "tr_pc": tr_pc,
        "voxels_rgb": voxels_rgb,
        "proj_rgb": proj_rgb,
        "drc_probs": drc_probs,
        "proj_depth": proj_depth
    }

    output = output_all['proj']
    voxels = output_all['voxels']
    tr_pc = output_all['tr_pc']
    save_depth_map_cv2(proj_depth,f'gt',os.path.join('/home/lancer/test/202403'))
    return output, voxels, tr_pc, proj_depth

def pointcloud2voxels3d_fast(pc, rgb = None, vox_size=64):  # [B,N,3]
    # vox_size = 137
    vox_size_z = vox_size

    batch_size = pc.shape[0]
    num_points = pc.shape[1]

    has_rgb = rgb is not None

    grid_size = 1.0
    half_size = grid_size / 2

    filter_outliers = True
    valid = torch.logical_and(pc >= -half_size, pc <= half_size).cuda()
    valid = torch.all(valid, axis=-1)

    vox_size_tf = torch.FloatTensor([[[vox_size_z, vox_size, vox_size]]]).cuda()
    # print(pc.shape)
    # print(vox_size_tf.shape)
    pc_grid = (pc + half_size) * (vox_size_tf - 1)

    indices_floor = torch.floor(pc_grid)
    indices_int = indices_floor.int()
    batch_indices = torch.arange(0, batch_size, 1)
    batch_indices = torch.unsqueeze(batch_indices, -1)
    batch_indices = batch_indices.repeat(1, num_points)
    batch_indices = torch.unsqueeze(batch_indices, -1).cuda()

    indices = torch.cat([batch_indices, indices_int], axis=2).cuda()
    indices = torch.reshape(indices, (-1, 4))

    r = pc_grid - indices_floor  # fractional part
    rr = [1.0 - r, r]

    if filter_outliers:
        valid = torch.reshape(valid, (-1, ))
        indices = indices[valid, :]

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][:, :, 0] * rr[pos[1]][:, :, 1] * rr[pos[2]][:, :, 2]
        updates = torch.reshape(updates_raw, (-1,)).cuda()
        if filter_outliers:
            updates = torch.masked_select(updates, valid).cuda()

        indices_loc = indices.cuda()
        indices_shift = torch.Tensor([[0] + pos]).cuda()
        num_updates = indices_loc.shape[0]
        indices_shift = indices_shift.repeat(num_updates, 1)
        indices_loc = indices_loc + indices_shift

        print('--------',indices_loc.shape, updates.dtype,  [batch_size, vox_size_z, vox_size, vox_size])
        indices_loc = indices_loc.to(torch.int64)
        
        voxels = torch.zeros(batch_size * vox_size_z * vox_size * vox_size).cuda().to(torch.float64)
        #voxels.scatter(1,indices_loc,updates)
        import torch_scatter
        #linear_indices = torch.arange(batch_size * vox_size_z * vox_size * vox_size).view(batch_size, vox_size_z, vox_size, vox_size)
        torch.use_deterministic_algorithms(False)
        flattened_indices_loc = (indices_loc[:, 0] * vox_size_z * vox_size * vox_size + indices_loc[:, 1] * vox_size * vox_size + indices_loc[:, 2] * vox_size + indices_loc[:, 3])

        voxels = voxels.scatter(0, flattened_indices_loc, updates).view(batch_size, vox_size_z, vox_size, vox_size)
        voxels.detach().cpu().numpy()
        #for i in range(indices_loc.shape[0]):
        #    voxels[indices_loc[i][0], indices_loc[i][1], indices_loc[i][2], indices_loc[i][3]] = updates[i]
        
        voxels_rgb = None

        return voxels, voxels_rgb

    voxels = []
    voxels_rgb = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                vx, vx_rgb = interpolate_scatter3d([k, j, i])
                voxels.append(vx)
                if vx_rgb  != None:
                    voxels_rgb.append(vx_rgb)

    voxels = sum(voxels)
    voxels_rgb = sum(voxels_rgb) if has_rgb else None
    
    return voxels, voxels_rgb



def render_views(pc, batch_size, p_rgb= None):

	pc = pc.detach().cpu().numpy()
	
	skip = 50

	x_rot = np.reshape(np.array([ii for ii in range(-180, 180+1, skip)]+\
							[1 for _ in range(-180, 180+1, skip)]+\
							[2 for _ in range(-180, 180+1, skip)]+\
							[45,    -45, -135, 135, 45,  -45, -135, 135])/180.0 *np.pi, [-1,1])
	y_rot = np.reshape(np.array([3 for _ in range(-180, 180+1, skip)]+\
								[ii for ii in range(-180, 180+1, skip)]+\
								[4 for _ in range(-180, 180+1, skip)]+\
								[-45,  -135, -135, -45, 45,  135,  135,  45])/180.0 *np.pi, [-1, 1])
	z_rot = np.reshape(np.array([5 for _ in range(-180, 180+1, skip)]+\
								[6 for _ in range(-180, 180+1, skip)]+\
								[ii for ii in range(-180, 180+1, skip)]+\
								[-135, -135,  135, 135, -45, -45,  45,   45])/180.0 *np.pi, [-1,1])


	trams = np.concatenate([x_rot, y_rot, z_rot, np.ones(x_rot.shape)], 1)

	all_result = []
	for ii in range(batch_size):
		output, voxels, tr_pc, proj_depth  = pointcloud_project_fast(torch.unsqueeze(torch.Tensor(pc[ii]).cuda(), 0), trams)
		if ii != 0:
			all_result = torch.concat([all_result, output], 0)
		else:
			all_result = output
	return all_result


def render_single_views(pc, batch_size, p_rgb= None):
	pc = pc.detach().cpu().numpy()
	skip = 360
	x_rot = np.reshape(np.array([ii for ii in range(-180, 180+1, skip)]+\
							[1 for _ in range(-180, 180+1, skip)]+\
							[2 for _ in range(-180, 180+1, skip)]+\
							[45,    -45, -135, 135, 45,  -45, -135, 135])/180.0 *np.pi, [-1,1])
	y_rot = np.reshape(np.array([3 for _ in range(-180, 180+1, skip)]+\
								[ii for ii in range(-180, 180+1, skip)]+\
								[4 for _ in range(-180, 180+1, skip)]+\
								[-45,  -135, -135, -45, 45,  135,  135,  45])/180.0 *np.pi, [-1, 1])
	z_rot = np.reshape(np.array([5 for _ in range(-180, 180+1, skip)]+\
								[6 for _ in range(-180, 180+1, skip)]+\
								[ii for ii in range(-180, 180+1, skip)]+\
								[-135, -135,  135, 135, -45, -45,  45,   45])/180.0 *np.pi, [-1,1])
	trams = np.concatenate([x_rot, y_rot, z_rot, np.ones(x_rot.shape)], 1)
	all_result = []
	for ii in range(batch_size):
		output, voxels, tr_pc  = pointcloud_project_fast(torch.unsqueeze(torch.Tensor(pc[ii]), 0), trams, all_rgb = p_rgb)
		if ii != 0:
			all_result = torch.concat([all_result, output], 0)
		else:
			all_result = output
	return all_result
