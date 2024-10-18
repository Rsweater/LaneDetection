
import torch.nn as nn
from mmcv.cnn import kaiming_init

class CtnetHead(nn.Module):
    def __init__(self, in_channels, heads_dict, final_kernel=1, init_bias=-2.19, use_bias=False):
        super(CtnetHead, self).__init__()
        self.heads_dict = heads_dict

        for cur_name in self.heads_dict:
            output_channels = self.heads_dict[cur_name]['out_channels']
            num_conv = self.heads_dict[cur_name]['num_conv']

            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(in_channels, output_channels, kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True))
            fc = nn.Sequential(*fc_list)

            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m.weight.data)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.heads_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)
        return ret_dict