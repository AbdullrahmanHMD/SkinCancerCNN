import torch
import torch.nn as nn


class SkinCancerModel(nn.Module):
    def __init__(self, number_of_classes=7, in_channel=3, image_size=(450, 600), dropout_prob=0.5):
        super(SkinCancerModel, self).__init__()
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # --- First layer ----------------------------------------------------------
        
        kernel_size, stride, padding = 3, 1, 1
        out_channels_1 = 4
        
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels_1,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        # --- Getting the image size after conv_1 --------------------------------------
        new_image_size_ = self.new_image_size(image_dims=image_size, kernel=kernel_size,
                                              padding=padding, stride=stride)
        # ------------------------------------------------------------------------------
        
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
        
        # --- Getting the image size after max_pooling_1 -------------------------------
        new_image_size_ = self.new_image_size(image_dims=new_image_size_, kernel=kernel_size,
                                              padding=0, stride=kernel_size)
        # ------------------------------------------------------------------------------
        
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels_1)
        self.dropout_1 = nn.Dropout(p=dropout_prob)
        
        # --- Second layer ----------------------------------------------------------
        
        kernel_size, stride, padding = 3, 1, 0
        out_channels_2 = 8
        
        self.conv_2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        # --- Getting the image size after conv_2 --------------------------------------
        new_image_size_ = self.new_image_size(image_dims=new_image_size_, kernel=kernel_size,
                                              padding=padding, stride=stride)
        # ------------------------------------------------------------------------------
        
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
        
        # --- Getting the image size after max_pooling_2 -------------------------------
        new_image_size_ = self.new_image_size(image_dims=new_image_size_, kernel=kernel_size,
                                              padding=0, stride=kernel_size)
        # ------------------------------------------------------------------------------
        
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels_2)
        self.dropout_2 = nn.Dropout(p=dropout_prob)
        
        # --- Third layer ----------------------------------------------------------
        
        kernel_size, stride, padding = 5, 1, 0 # 5, 1, 0
        out_channels_3 = 16
        
        self.conv_3 = nn.Conv2d(in_channels=out_channels_2, out_channels=out_channels_3,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        # --- Getting the image size after conv_3 --------------------------------------
        new_image_size_ = self.new_image_size(image_dims=new_image_size_, kernel=kernel_size,
                                              padding=padding, stride=stride)
        # ------------------------------------------------------------------------------
        kernel_size = 3
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
        
        # --- Getting the image size after max_pooling_3 -------------------------------
        new_image_size_ = self.new_image_size(image_dims=new_image_size_, kernel=kernel_size,
                                              padding=0, stride=kernel_size)
        # ------------------------------------------------------------------------------
        
        self.batch_norm_3 = nn.BatchNorm2d(num_features=out_channels_3)
        self.dropout_3 = nn.Dropout(p=dropout_prob)
        
        # --- Fully Connected layer ------------------------------------------------
        in_features = new_image_size_[0] * new_image_size_[1] * out_channels_3
        # in_features = new_image_size_[0] * new_image_size_[1] * out_channels_2
        # hidden_size = 5000
        
        self.fc_1 = nn.Linear(in_features=in_features, out_features=number_of_classes)
        # self.fc_1 = nn.Linear(in_features=in_features, out_features=number_of_classes)
        
        # self.fc_1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        # self.fc_2 = nn.Linear(in_features=hidden_size, out_features=number_of_classes)
        
        # --- Weight initialization ------------------------------------------------
        
        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.kaiming_normal_(self.conv_2.weight)
        nn.init.kaiming_normal_(self.conv_3.weight)
        
        # --- Bias initialization -------------------------------------------------
        
        nn.init.constant_(self.conv_1.bias, 0.0)
        nn.init.constant_(self.conv_2.bias, 0.0)
        nn.init.constant_(self.conv_3.bias, 0.0)
        
    
    def forward(self, x):
        
        # --- First Layer --------------------------
        
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.max_pooling_1(x)
        
        x = self.leaky_relu(x)
        x = self.dropout_1(x)
        
        # --- Second Layer --------------------------
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.max_pooling_2(x)
        
        x = self.leaky_relu(x)
        x = self.dropout_2(x)
        
        # --- Third Layer --------------------------
        
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.max_pooling_3(x)
        
        x = self.leaky_relu(x)
        x = self.dropout_3(x)
        
        # --- Fully Connected ----------------------
        
        # Flattening the image so it can be fed to the fully connected layer:
        x = x.view(x.size(0), -1)
        
        x = self.fc_1(x)
        # x = self.fc_2(x)
        
        return x
    
    
    def new_image_size(self, image_dims : tuple, kernel : int, padding : int, stride : int):
        """ Calculates the new dimensions of an input image after applying
            the convolution operation.

        Args:
            image_dims (tuple): A tuple containing the dimensions of the image with the
                                first element being the height and the second element
                                being the width.
            kernel (int): The kernel size of the filter.
            padding (int): The padding of the image.
            stride (int): The stride of the filter.

        Returns:
            tuple: A tuple containing the new dimension of the image with the new height
                    being the first element and the new width being the second element.
        """
        new_image_height = (image_dims[0] - kernel + 2 * padding) // stride + 1
        new_image_width = (image_dims[1] - kernel + 2 * padding) // stride + 1
        
        return new_image_height, new_image_width