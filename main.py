# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import torch
from torch.nn import functional as F
from torch import nn
from torchvision import datasets, transforms
import pytorch_lightning as pl
import torchvision

class NavyaCNN(pl.LightningModule):
    def __init__(self, hparams):
        super(NavyaCNN, self).__init__()

        self.hparams = hparams
        self.conv1 = nn.Conv2d(1,10, kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def prepare_data(self):
        # download only
        datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        datasets.MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())


    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        if stage == 'fit' or stage is None:
            mnist_train = datasets.MNIST('../data', train=True, transform=transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_train, [55000,5000])
        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST('../data', train=False, transform=transform)



    def forward(self, x):
        output_conv1 = F.relu(F.max_pool2d(self.conv1(x),2))
        output_conv2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(output_conv1)),2))

        input_fc1 = output_conv2.view(-1,320)
        output_fc1 = F.relu(self.fc1(input_fc1))
        output_dropout = F.dropout(output_fc1, training=self.training)
        output_fc2 = self.fc2(output_dropout)
        return F.log_softmax(output_fc2)

    def loss_function(self, output, target):
        return  nn.CrossEntropyLoss()(output.view(-1,10),target)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


    def training_step(self, batch, batch_index):
        x, y = batch
        label = y.view(-1)
        out = self(x)
        loss = self.loss_function(out, label)

        log = {'training_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_index):
        x, y = batch
        label = y.view(-1)
        out = self(x)
        loss = self.loss_function(out, label)

        return {'val_loss': loss}


    def test_step(self, batch, batch_index):
        x, y = batch
        label = y.view(-1)
        out = self(x)
        correct = 0
        total =0
        _, predicted = torch.max(out,1)
        total += label.size(0)
        correct +=(predicted == label).sum().item()
        return {'correct': correct, 'total': total}

    def test_epoch_end(self, outputs):
        total_correct = 0
        total_num = 0
        for x in outputs:
            total_correct += x['correct']
            total_num += x['total']

        correct_rate = total_correct/total_num
        print(correct_rate)
        return {'correct_rate': correct_rate}


    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()

        log = {'avg_val_loss': val_loss}
        return {'log ': log}

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.mnist_train,
            batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.mnist_val,
            batch_size=self.hparams.batch_size)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.mnist_test,
            batch_size=self.hparams.batch_size)
        return test_loader



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=36)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    args = parser.parse_args()

    navya = NavyaCNN(hparams=args)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(navya)

    trainer.test(navya)

    #test function outside the member function
    """
    navya.eval()
    transform_1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = torchvision.datasets.MNIST('../data',
                                              train=False,
                                              transform=transform_1)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=36,
                                               shuffle=False)

    correct = 0
    total = 0
    for x, label in test_loader:
        output = navya(x)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total))
   """


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
