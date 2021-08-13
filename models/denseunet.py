import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0.0):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=down_size)
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)

    def forward(self, x):
        if self.down_size is not None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))

        return out


class UpBlock(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0.0):
        super(UpBlock, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=1, padding=0)
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x):
        x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')
        x = torch.cat((x, prev_feature_map), dim=1)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))

        return out


class DenseUNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=1, channel_size=16, dropout=False, prob=0.0):
        super(DenseUNet, self).__init__()

        self.down_block1 = DownBlock(input_channels=input_channels, output_channels=channel_size,
                                     down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DownBlock(input_channels=channel_size, output_channels=channel_size,
                                     down_size=2, dropout=dropout, prob=prob)
        self.down_block3 = DownBlock(input_channels=channel_size, output_channels=channel_size,
                                     down_size=2, dropout=dropout, prob=prob)
        self.down_block4 = DownBlock(input_channels=channel_size, output_channels=channel_size,
                                     down_size=2, dropout=dropout, prob=prob)
        self.down_block5 = DownBlock(input_channels=channel_size, output_channels=channel_size,
                                     down_size=2, dropout=dropout, prob=prob)

        self.up_block1 = UpBlock(skip_channels=channel_size, input_channels=channel_size,
                                 output_channels=channel_size, up_stride=2, dropout=dropout,
                                 prob=prob)
        self.up_block2 = UpBlock(skip_channels=channel_size, input_channels=channel_size,
                                 output_channels=channel_size, up_stride=2, dropout=dropout,
                                 prob=prob)
        self.up_block3 = UpBlock(skip_channels=channel_size, input_channels=channel_size,
                                 output_channels=channel_size, up_stride=2, dropout=dropout,
                                 prob=prob)
        self.up_block4 = UpBlock(skip_channels=channel_size, input_channels=channel_size,
                                 output_channels=channel_size, up_stride=2, dropout=dropout,
                                 prob=prob)

        self.push_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels,
                                     kernel_size=1, padding=0)
        self.grasp_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels,
                                      kernel_size=1, padding=0)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.up_block1(x4, x5)
        x7 = self.up_block2(x3, x6)
        x8 = self.up_block3(x2, x7)
        x9 = self.up_block4(x1, x8)

        if self.dropout:
            push_output = self.push_output(self.dropout_pos(x9))
            grasp_output = self.grasp_output(self.dropout_cos(x9))
        else:
            push_output = self.push_output(x9)
            grasp_output = self.grasp_output(x9)

        return push_output, grasp_output
