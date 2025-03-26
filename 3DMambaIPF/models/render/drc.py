import torch

DTYPE = torch.float32

def slice_axis0(t, idx):
    init = t[idx, :, :, :, :]
    return torch.unsqueeze(init, axis=0)


def drc_event_probabilities_impl(voxels):
    # swap batch and Z dimensions for the ease of processing
    input = voxels.transpose(1, 0)

    drc_tf_cumulative = True
    logsum = True
    dtp = DTYPE

    clip_val = 0.00001
    if logsum:
        input = torch.clamp(input, clip_val, 1.0-clip_val)

    def log_unity(shape):
        return torch.ones(shape)*clip_val

    y = input
    x = 1.0 - y
    if logsum:
        y = torch.log(y)
        x = torch.log(x)
        op_fn = torch.add
        unity_fn = log_unity
        cum_fun = torch.cumsum
    else:
        op_fn = torch.mul
        unity_fn = torch.ones
        cum_fun = torch.cumprod

    v_shape = input.shape
    singleton_shape = [1, v_shape[1], v_shape[2], v_shape[3], v_shape[4]]

    # this part computes tensor of the form,
    # ex. for vox_size=3 [1, x1, x1*x2, x1*x2*x3]
    if  drc_tf_cumulative:
        r = cum_fun(x, axis=0)
    else:
        depth = input.shape[0]
        collection = []
        for i in range(depth):
            current = slice_axis0(x, i)
            if i > 0:
                prev = collection[-1]
                current = op_fn(current, prev)
            collection.append(current)
        r = torch.cat(collection, dim=0).cuda()

    r1 = unity_fn(singleton_shape).cuda()
    p1 = torch.cat([r1, r], axis=0).cuda()  # [1, x1, x1*x2, x1*x2*x3]

    r2 = unity_fn(singleton_shape).cuda()
    p2 = torch.cat([y, r2], axis=0) .cuda() # [(1-x1), (1-x2), (1-x3), 1])

    p = op_fn(p1, p2)  # [(1-x1), x1*(1-x2), x1*x2*(1-x3), x1*x2*x3]
    if logsum:
        p = torch.exp(p)

    return p, singleton_shape, input


def drc_event_probabilities(voxels, cfg):
    p, _, _ = drc_event_probabilities_impl(voxels, cfg)
    return p


def drc_projection(voxels):
    p, singleton_shape, input = drc_event_probabilities_impl(voxels)
    dtp = DTYPE

    # colors per voxel (will be predicted later)
    # for silhouettes simply: [1, 1, 1, 0]
    c0 = torch.ones_like(input, dtype=dtp).cuda()
    c1 = torch.zeros(singleton_shape, dtype=dtp).cuda()
    c = torch.cat([c0, c1], axis=0)

    # \sum_{i=1:vox_size} {p_i * c_i}
    out = torch.sum(p * c, [0]).cuda()
    p = p.cuda()
    return out, p


def drc_depth_grid(z_size):
    camera_distance = 2.0
    max_depth = 10.0
    i_s = torch.arange(0, z_size, 1).float()
    di_s = i_s / z_size - 0.5 + camera_distance
    last = torch.Tensor([max_depth])
    return torch.cat([di_s, last], dim=0)


def drc_depth_projection(p):
    z_size = p.shape[0]
    z_size = z_size - 1  # because p is already of size vox_size + 1
    depth_grid = drc_depth_grid(z_size).cuda()
    psi = torch.reshape(depth_grid, shape=[-1, 1, 1, 1, 1])
    # \sum_{i=1:vox_size} {p_i * psi_i}
    out = torch.sum(p * psi, [0])
    return out
