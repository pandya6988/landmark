import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.conv1 = nn.Conv2d(3, 64, 3, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.conv3 = nn.Conv2d(128, 256, 2)
        self.conv4 = nn.Conv2d(256, 512,3, padding='valid')
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(3)
        self.lin1 = nn.Linear(2048, 512)
        self.lin2 = nn.Linear(512, 128)
        self.lin3 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.flatten()
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.maxpool1( F.relu( self.batch_norm1( self.conv1(x))) )
        x = self.maxpool2( F.relu( self.batch_norm2( self.conv2(x))) )
        x = self.maxpool3( F.relu( self.batch_norm3( self.conv3(x))) )
        x = self.dropout1( F.relu( self.conv4(x) ) )
        x = self.flatten (x)
        
        x = F.relu( self.dropout1( self.lin1(x) ) )
        x = F.relu( self.dropout2( self.lin2(x) ) )
        x = self.lin3(x)

        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
