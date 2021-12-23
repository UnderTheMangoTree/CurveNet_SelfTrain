"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def so3_rotate(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    devices = batch_data.device
    batch_data = batch_data.cpu().numpy()
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle_A = np.random.uniform() * 2 * np.pi
        rotation_angle_B = np.random.uniform() * 2 * np.pi
        rotation_angle_C = np.random.uniform() * 2 * np.pi

        cosval_A = np.cos(rotation_angle_A)
        sinval_A = np.sin(rotation_angle_A)
        cosval_B = np.cos(rotation_angle_B)
        sinval_B = np.sin(rotation_angle_B)
        cosval_C = np.cos(rotation_angle_C)
        sinval_C = np.sin(rotation_angle_C)
        rotation_matrix = np.array([[cosval_B*cosval_C, -cosval_B*sinval_C, sinval_B],
                                    [sinval_A*sinval_B*cosval_C+cosval_A*sinval_C, -sinval_A*sinval_B*sinval_C+cosval_A*cosval_C, -sinval_A*cosval_B],
                                    [-cosval_A*sinval_B*cosval_C+sinval_A*sinval_C, cosval_A*sinval_B*sinval_C+sinval_A*cosval_C, cosval_A*cosval_B]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    rotated_data = torch.from_numpy(rotated_data).to(devices)

    return rotated_data


def azi_rotate(batch_data, rotation_axis="z"):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    '''
    if np.ndim(batch_data) != 6:
        raise ValueError("np.ndim(batch_data) != 3, must be (b, n, 3)")
    if batch_data.shape[2] != 6:
        raise ValueError("batch_data.shape[2] != 3, must be (x, y, z)")
    '''
    device = batch_data.device
    batch_data = batch_data.cpu().numpy()
    rotated_data = np.zeros((batch_data.shape[0], batch_data.shape[1], 3), dtype=np.float32)
    rotated_normal = np.zeros((batch_data.shape[0], batch_data.shape[1], 3), dtype=np.float32)

    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        if rotation_axis == "x":
            rotation_matrix = np.array(
                [[1, 0, 0], [0, cosval, sinval], [0, -sinval, cosval]]
            )
        elif rotation_axis == "y":
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
        elif rotation_axis == "z":
            rotation_matrix = np.array(
                [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
            )
        else:
            raise ValueError("Wrong rotation axis")
        shape_pc = batch_data[k, :, :3]
        # shape_nm = batch_data[k, :, 3:]

        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        # rotated_normal[k, ...] = np.dot(shape_nm.reshape((-1, 3)), rotation_matrix)

    # rotated_data = np.concatenate((rotated_data, rotated_normal), axis=-1)
    rotated_data = torch.from_numpy(rotated_data).to(device)
    return rotated_data


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
