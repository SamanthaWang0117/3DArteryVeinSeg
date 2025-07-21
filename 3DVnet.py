import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DCH(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation='linear'):
        super(Conv3DCH, self).__init__()
        # 'same' padding logic for PyTorch
        if padding == 'same':
            if isinstance(kernel_size, int):
                k = (kernel_size,)*3
            else:
                k = kernel_size
            padding_x = (k[0] - 1) // 2
            padding_y = (k[1] - 1) // 2
            padding_z = (k[2] - 1) // 2
            self.padding_tuple_x = (padding_x, padding_x, padding_y, padding_y, padding_z, padding_z)
            self.padding_tuple_y = (padding_x, padding_x, padding_y, padding_y, padding_z, padding_z)
            self.padding_tuple_z = (padding_x, padding_x, padding_y, padding_y, padding_z, padding_z)
        else:
            self.padding_tuple_x = self.padding_tuple_y = self.padding_tuple_z = (0,0,0,0,0,0) # Handle other padding modes if needed

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        # X-axis convolution (kernel_size: 1xKyxKz)
        self.convx = nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size[1], kernel_size[2]), stride=stride, padding=(0, kernel_size[1]//2, kernel_size[2]//2))
        # Y-axis convolution (kernel_size: Kx1xKz)
        self.convy = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size[0], 1, kernel_size[2]), stride=stride, padding=(kernel_size[0]//2, 0, kernel_size[2]//2))
        # Z-axis convolution (kernel_size: KxKyx1)
        self.convz = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size[0], kernel_size[1], 1), stride=stride, padding=(kernel_size[0]//2, kernel_size[1]//2, 0))

        self.activation = self._get_activation(activation)

    def _get_activation(self, activation_name):
        if activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'linear':
            return nn.Identity() # No activation for linear
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x):
        # Apply padding explicitly if 'same' was specified and kernel size is even for example
        # For 'same' padding, PyTorch automatically handles it if padding is set to kernel_size // 2.
        # If kernel size is even, the logic for 'same' padding might need a slight adjustment
        # in PyTorch to match Keras's behavior exactly, but for odd kernels, it's usually kernel_size // 2.
        # For simplicity, assuming odd kernels for now as is typical for V-Net.

        x_out = self.convx(x)
        y_out = self.convy(x)
        z_out = self.convz(x)
        out = x_out + y_out + z_out # Summing the outputs

        return self.activation(out)
    
    