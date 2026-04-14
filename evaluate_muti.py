#!/usr/bin/env python

import pathlib

import numpy as np
import torch
import tqdm

from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_model)
from gaze_estimation.utils import compute_angle_error, load_config, save_config


def compute_pose_mae(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算 Head Pose 的平均绝对误差 (Mean Absolute Error, MAE)，假设输入为弧度，输出转换为角度"""
    mae_radians = torch.abs(predictions - labels).mean(dim=1)
    mae_degrees = mae_radians * 180 / np.pi
    return mae_degrees


def test(model, test_loader, config):
    model.eval()
    device = torch.device(config.device)

    # 存储所有的预测值和真实值
    pred_gazes = []
    pred_poses = []
    gts_gazes = []
    gts_poses = []
    
    with torch.no_grad():
        for images, poses, gazes in tqdm.tqdm(test_loader):
            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)

            # 模型前向传播，针对多任务模型进行修改
            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                raise NotImplementedError("multi-task for MPIIGaze is not implemented")
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                # MPIIFaceGaze 模型返回 gaze 和 pose 的双预测结果
                out_gaze, out_pose = model(images)
            else:
                raise ValueError
            
            # 收集并移动到 CPU 上
            pred_gazes.append(out_gaze.cpu())
            pred_poses.append(out_pose.cpu())
            gts_gazes.append(gazes.cpu())
            gts_poses.append(poses.cpu())

    # 拼接所有的 batch 结果
    pred_gazes = torch.cat(pred_gazes)
    pred_poses = torch.cat(pred_poses)
    gts_gazes = torch.cat(gts_gazes)
    gts_poses = torch.cat(gts_poses)
    
    # 修复：Gaze 算角度误差，Pose 算 MAE
    gaze_angle_error = float(compute_angle_error(pred_gazes, gts_gazes).mean())
    pose_angle_error = float(compute_pose_mae(pred_poses, gts_poses).mean())
    
    # 返回预测值，真实值，以及平均角度误差
    return pred_gazes, pred_poses, gts_gazes, gts_poses, gaze_angle_error, pose_angle_error


def main():
    config = load_config()

    output_rootdir = pathlib.Path(config.test.output_dir)
    checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    output_dir = output_rootdir / checkpoint_name
    output_dir.mkdir(exist_ok=True, parents=True)
    save_config(config, output_dir)

    test_loader = create_dataloader(config, is_train=False)

    model = create_model(config)
    checkpoint = torch.load(config.test.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 接收测试函数返回的多任务结果
    pred_gazes, pred_poses, gts_gazes, gts_poses, gaze_angle_error, pose_angle_error = test(model, test_loader, config)

    # 分别打印出 gaze 和 pose 的平均角度误差
    print(f'The mean gaze angle error (deg): {gaze_angle_error:.2f}')
    print(f'The mean pose MAE (deg): {pose_angle_error:.2f}')

    # 保存多任务的预测结果和真实值到对应的 numpy 数组文件
    output_path = output_dir / 'pred_gazes.npy'
    np.save(output_path, pred_gazes.numpy())
    output_path = output_dir / 'pred_poses.npy'
    np.save(output_path, pred_poses.numpy())
    
    output_path = output_dir / 'gts_gazes.npy'
    np.save(output_path, gts_gazes.numpy())
    output_path = output_dir / 'gts_poses.npy'
    np.save(output_path, gts_poses.numpy())
    
    # 记录详细的 error 文件
    output_path = output_dir / 'error.txt'
    with open(output_path, 'w') as f:
        f.write(f'gaze_angle_error: {gaze_angle_error}\n')
        f.write(f'pose_mae: {pose_angle_error}\n')


if __name__ == '__main__':
    main()
