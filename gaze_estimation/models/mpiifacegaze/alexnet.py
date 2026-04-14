from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yacs.config


class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()
        # 尝试下载和加载官方带有哈希校验的模型
        try:
            # 兼容不同版本的 torchvision
            if hasattr(torchvision.models, 'AlexNet_Weights'):
                 self.feature_extractor = torchvision.models.alexnet(
                     weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
                 ).features
            else:
                 self.feature_extractor = torchvision.models.alexnet(
                     pretrained=True
                 ).features
        except RuntimeError as e:
            if "invalid hash value" in str(e) or "corrupted" in str(e).lower():
                import os
                # 如果是哈希不匹配错误，说明网络波动导致下载的文件损坏了
                # 我们手动把它删掉，然后再重新尝试下载一次
                cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
                model_path = os.path.join(cache_dir, 'alexnet-owt-7be5be79.pth')
                if os.path.exists(model_path):
                    os.remove(model_path)
                print("检测到由于网络波动导致损坏的预训练权重，已删除。正在为您重新下载，请稍候...")
                if hasattr(torchvision.models, 'AlexNet_Weights'):
                     self.feature_extractor = torchvision.models.alexnet(
                         weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
                     ).features
                else:
                     self.feature_extractor = torchvision.models.alexnet(
                         pretrained=True
                     ).features
            else:
                raise e
        except Exception:
             self.feature_extractor = torchvision.models.alexnet(pretrained=False).features

        # While the pretrained models of torchvision are trained using
        # images with RGB channel order, in this repository images are
        # treated as BGR channel order.
        # Therefore, reverse the channel order of the first convolutional
        # layer.
        module = getattr(self.feature_extractor, '0')
        module.weight.data = module.weight.data[:, [2, 1, 0]]

        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(256 * 13**2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

        self._register_hook()
        self._initialize_weight()

    def _initialize_weight(self) -> None:
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv1.bias, val=0.1)
        nn.init.constant_(self.conv2.bias, val=0.1)
        nn.init.constant_(self.conv3.bias, val=1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.005)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.0001)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.0001)
        nn.init.constant_(self.fc1.bias, val=1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def _register_hook(self) -> None:
        n_channels = self.conv1.in_channels

        def hook(
            module: nn.Module, grad_in: Union[Tuple[torch.Tensor, ...],
                                              torch.Tensor],
            grad_out: Union[Tuple[torch.Tensor, ...], torch.Tensor]
        ) -> Optional[torch.Tensor]:
            return tuple(grad / n_channels for grad in grad_in)

        self.conv3.register_backward_hook(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        x = x * y
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = self.fc3(x)
        return x
