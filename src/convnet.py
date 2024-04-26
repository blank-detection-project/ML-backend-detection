import argparse

import torch
import torch.nn as nn


OUTPUT_SIZE = 10


class ConvNetTwoConvTwoDenseLayersWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(4*4*40, 1000)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)

        self.dropout3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(1000, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 4 * 4 * 40)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.dropout3(x)
        x = self.out(x)
        return x


def parse_command_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="number of epochs", type=int,
                        default=60)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
    parser.add_argument("--wd", help="weight decay", type=float, default=0)
    parser.add_argument("--extend_data", help="use extended training data",
                        action="store_true")
    return parser.parse_args()


def main(args):
    net = ConvNetTwoConvTwoDenseLayersWithDropout()
    num_epochs = args.epochs
    lr = args.lr
    wd = args.wd
    train_loader = choose_train_loader(args)
    train_and_test_network(net, num_epochs=num_epochs, lr=lr, wd=wd,
                           train_loader=train_loader)


if __name__ == "__main__":
    main(parse_command_line_args())