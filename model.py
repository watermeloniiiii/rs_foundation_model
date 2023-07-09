import torch
import config
import torch.nn as nn


def bn_conv_relu(in_channels, out_channels):
    """
    usually features = [64,96]
    """
    conv = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

    return conv


def upconv(in_channels, out_channels):
    upconv = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU()
    )
    return upconv


class AutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AutoEncoder, self).__init__()
        self.conv1_1 = bn_conv_relu(in_channels, config.hyperparameters['hidden_layer'][0])
        self.conv1_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][0],
                                    config.hyperparameters['hidden_layer'][0])
        self.conv2_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][0],
                                    config.hyperparameters['hidden_layer'][1])
        self.conv2_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][1],
                                    config.hyperparameters['hidden_layer'][1])
        self.conv3_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][1],
                                    config.hyperparameters['hidden_layer'][2])
        self.conv3_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][2],
                                    config.hyperparameters['hidden_layer'][2])
        self.conv4_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][2],
                                    config.hyperparameters['hidden_layer'][3])
        self.conv4_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][3],
                                    config.hyperparameters['hidden_layer'][3])
        self.conv5_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][3],
                                    config.hyperparameters['hidden_layer'][4])
        self.conv5_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][4],
                                    config.hyperparameters['hidden_layer'][4])

        self.upconv4_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][4],
                                      config.hyperparameters['hidden_layer'][4])
        self.upconv4_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][4],
                                      config.hyperparameters['hidden_layer'][3])
        self.unpool4 = upconv(config.hyperparameters['hidden_layer'][3],
                              config.hyperparameters['hidden_layer'][3])

        self.upconv3_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][3],
                                      config.hyperparameters['hidden_layer'][3])
        self.upconv3_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][3],
                                      config.hyperparameters['hidden_layer'][2])
        self.unpool3 = upconv(config.hyperparameters['hidden_layer'][2],
                              config.hyperparameters['hidden_layer'][2])

        self.upconv2_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][2],
                                      config.hyperparameters['hidden_layer'][2])
        self.upconv2_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][2],
                                      config.hyperparameters['hidden_layer'][1])
        self.unpool2 = upconv(config.hyperparameters['hidden_layer'][1],
                              config.hyperparameters['hidden_layer'][1])

        self.upconv1_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][1],
                                      config.hyperparameters['hidden_layer'][1])
        self.upconv1_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][1],
                                      config.hyperparameters['hidden_layer'][0])
        self.unpool1 = upconv(config.hyperparameters['hidden_layer'][0],
                              config.hyperparameters['hidden_layer'][0])

        self.out = torch.nn.Conv2d(config.hyperparameters['hidden_layer'][0], out_channels, kernel_size=1, padding=0)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.avgpool = torch.nn.AvgPool2d(12)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        conv1 = self.conv1_2(self.conv1_1(x))
        maxpool1 = self.maxpool(conv1)
        conv2 = self.conv2_2(self.conv2_1(maxpool1))
        maxpool2 = self.maxpool(conv2)
        conv3 = self.conv3_2(self.conv3_1(maxpool2))
        maxpool3 = self.maxpool(conv3)
        conv4 = self.conv4_2(self.conv4_1(maxpool3))
        maxpool4 = self.maxpool(conv4)
        conv5 = self.conv5_2(self.conv5_1(maxpool4))

        code_vec = torch.squeeze(self.avgpool(conv5))

        upconv4 = self.upconv4_2(self.upconv4_1(conv5))
        unpool4 = self.unpool4(upconv4)
        upconv3 = self.upconv3_2(self.upconv3_1(unpool4))
        unpool3 = self.unpool3(upconv3)
        upconv2 = self.upconv2_2(self.upconv2_1(unpool3))
        unpool2 = self.unpool2(upconv2)
        upconv1 = self.upconv1_2(self.upconv1_1(unpool2))
        unpool1 = self.unpool1(upconv1)
        out = torch.sigmoid(self.out(unpool1))
        return code_vec, out


class DEC(AutoEncoder):
    def __init__(self, in_channels, out_channels):
        super(DEC, self).__init__(in_channels, out_channels)
        self.alpha = 1.0
        self.clusterCenter = torch.nn.Parameter(torch.randn(config.hyperparameters['clusters'],
                                                            config.hyperparameters['hidden_layer'][-1]))

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def updateClusterCenter(self, cc):
        self.clusterCenter.data = torch.from_numpy(cc)

    def getTDistribution(self, code_vec):
        xe = torch.unsqueeze(code_vec, 1).to('cuda') - self.clusterCenter.to('cuda')
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe, xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def getTargetDistribution(self, q):
        weight = q ** 2 / q.sum(0)
        return torch.autograd.Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)


class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv1_1 = bn_conv_relu(in_channels, config.hyperparameters['hidden_layer'][0])
        self.conv1_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][0],
                                    config.hyperparameters['hidden_layer'][0])
        self.conv2_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][0],
                                    config.hyperparameters['hidden_layer'][1])
        self.conv2_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][1],
                                    config.hyperparameters['hidden_layer'][1])
        self.conv3_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][1],
                                    config.hyperparameters['hidden_layer'][2])
        self.conv3_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][2],
                                    config.hyperparameters['hidden_layer'][2])
        self.conv4_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][2],
                                    config.hyperparameters['hidden_layer'][3])
        self.conv4_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][3],
                                    config.hyperparameters['hidden_layer'][3])
        self.conv5_1 = bn_conv_relu(config.hyperparameters['hidden_layer'][4],
                                    config.hyperparameters['hidden_layer'][4])
        self.conv5_2 = bn_conv_relu(config.hyperparameters['hidden_layer'][4],
                                    config.hyperparameters['hidden_layer'][4])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Sequential(
            nn.Linear(36 * config.hyperparameters['hidden_layer'][-1],
                      72 * config.hyperparameters['hidden_layer'][-1]),
            nn.LeakyReLU(0.1),
            nn.Linear(72 * config.hyperparameters['hidden_layer'][-1],
                      72 * config.hyperparameters['hidden_layer'][-1]),
            nn.LeakyReLU(0.1),
            nn.Linear(72 * config.hyperparameters['hidden_layer'][-1],
                      out_channels),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.maxpool(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.maxpool(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.maxpool(x)
        x = x.view(-1, 36 * 128)
        x = self.fc(x)
        return x

