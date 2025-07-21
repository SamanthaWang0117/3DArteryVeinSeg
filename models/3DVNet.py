import torch
import torch.nn as nn
import torch.nn.functional as F

class VNET(nn.Module):
    def __init__(self, nchannels=1, nlabels=2, nlevels=5, nfeats=16, cross_hair=False, dim=3, activation='tanh'):
        super(VNET, self).__init__()
        self.nlevels = nlevels
        self.nlabels = nlabels
        self.dim = dim
        self.cross_hair = cross_hair
        self.activation_func_name = activation

        # Choose the appropriate convolutional layer
        if dim == 3:
            self.ConvLayer = Conv3DCH if cross_hair else nn.Conv3d
            self.ConvTransposeLayer = nn.ConvTranspose3d
        elif dim == 2:
            self.ConvLayer = Conv2DCH if cross_hair else nn.Conv2d
            self.ConvTransposeLayer = nn.ConvTranspose2d
        else:
            raise ValueError("Dimension must be 2 or 3.")

        kernel = (5,) * dim
        # Define activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity() # Linear activation

        # Encoder path
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.encoder_add_or_concat = nn.ModuleList()

        in_channels_encoder = nchannels
        current_features = nfeats

        for level in range(nlevels):
            steps = level + 1 if level < 2 else 3
            encoder_block_layers = nn.ModuleList()
            for step in range(steps):
                encoder_block_layers.append(self.ConvLayer(in_channels_encoder, current_features, kernel_size=kernel, padding='same', activation=self.activation_func_name))
                in_channels_encoder = current_features # Output channels of previous layer become input for next

            self.encoders.append(nn.Sequential(*encoder_block_layers))

            # Residual/Concatenation connection at the end of each encoder block
            if level == 0:
                if nchannels == 1:
                    self.encoder_add_or_concat.append(self._add_layer) # Custom add function
                else:
                    self.encoder_add_or_concat.append(self._concat_layer)
            else:
                self.encoder_add_or_concat.append(self._add_layer)

            # Downsampling
            if level < nlevels - 1:
                self.downsamples.append(self.ConvLayer(current_features, current_features * 2, kernel_size=kernel, stride=(2,)*dim, padding='same', activation=self.activation_func_name))
                in_channels_encoder = current_features * 2
                current_features *= 2

        # Decoder path
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoder_add = nn.ModuleList() # All are additions in decoder

        for level in reversed(range(nlevels - 1)): # Iterate from nlevels-2 down to 0
            steps = level + 1 if level < 2 else 3
            current_features = nfeats * (2**level) # Adjust features for current decoder level

            # Skip connection concatenation happens *before* decoder convolutions
            # The input channels for the decoder block will be (current_features * 2) from skip + (current_features) from upsampled path
            decoder_in_channels = current_features * 2 # From concatenated skip connection

            decoder_block_layers = nn.ModuleList()
            for step in range(steps):
                decoder_block_layers.append(self.ConvLayer(decoder_in_channels, current_features, kernel_size=kernel, padding='same', activation=self.activation_func_name))
                decoder_in_channels = current_features # Subsequent layers use updated channels

            self.decoders.append(nn.Sequential(*decoder_block_layers))

            self.decoder_add.append(self._add_layer)

            # Upsampling (Transpose Conv)
            if level > 0:
                upsample_out_channels = current_features // 2
                self.upsamples.append(self.ConvTransposeLayer(current_features, upsample_out_channels, kernel_size=kernel, stride=(2,)*dim, padding=kernel[0]//2, output_padding=kernel[0]%2))
            else: # Final output layer
                self.final_conv = self.ConvLayer(current_features, nlabels, kernel_size=(1,)*dim, padding='same', activation='linear')
                self.softmax = nn.Softmax(dim=1) # dim=1 for channels_first


    def _add_layer(self, inputs):
        return inputs[0] + inputs[1]

    def _concat_layer(self, inputs):
        return torch.cat(inputs, dim=1)

    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        encoder_input_for_add = x # Initial input for the first addition

        # Encoder Path
        current_x = x
        for level in range(self.nlevels):
            # Apply encoder block convolutions
            current_x_conv = self.encoders[level](current_x)

            # Add/Concatenate with initial input of the block or previous block's output
            if level == 0:
                if self.nchannels == 1:
                    current_x_final = self.encoder_add_or_concat[level]([encoder_input_for_add, current_x_conv])
                else:
                    current_x_final = self.encoder_add_or_concat[level]([encoder_input_for_add, current_x_conv])
            else:
                current_x_final = self.encoder_add_or_concat[level]([encoder_input_for_add, current_x_conv])

            encoder_outputs.append(current_x_final)

            # Downsample for the next level if not the last level
            if level < self.nlevels - 1:
                current_x = self.downsamples[level](current_x_final)
                encoder_input_for_add = current_x # This becomes the input for the next encoder block's addition

        # Decoder Path (working backwards from the deepest encoder output)
        decoder_outputs = []
        # The last encoder_output is the bottleneck, which starts the decoder path
        current_decoder_input = encoder_outputs[-1] # This is the input to the deepest decoder convolution

        for level in reversed(range(self.nlevels - 1)):
            # Upsample from the previous level's decoder output (or bottleneck)
            # The original V-Net structure implies that the upsampling happens *after* the decoder convolutions,
            # and then *concatenated* with the skip connection for the *next* decoder level.
            # This is a bit tricky to map directly from the provided code's `sort` logic,
            # but usually for U-Net/V-Net, it's skip_connection + upsampled_feature -> decoder block.

            # Let's re-interpret based on common V-Net implementations and the provided code's concatenation logic.
            # The provided code has:
            # {'layer': 'Concatenate', 'inputs': ['encoder_X_final', 'decoder_Y_subsample'], ...}
            # This means the skip connection (encoder_X_final) is concatenated with the upsampled feature from the *next lower* level's decoder.
            # This implies decoder_Y_subsample is an upsampled feature map.

            # Reversing this:
            # 1. Take the current decoder input (from previous upsample or bottleneck)
            # 2. Concatenate with the corresponding encoder skip connection
            # 3. Apply decoder convolutions
            # 4. Add the upsampled input (before concat) for the final block addition
            # 5. Upsample for the next higher level's decoder input

            # Step 1: Get the skip connection from the corresponding encoder level
            skip_connection = encoder_outputs[level]

            # Step 2: Perform upsampling *from the current decoder path*
            # The V-Net paper describes a "deconvolution" for upsampling.
            # Based on the Keras code, the `decoder_level_subsample` is a TransposeConv, which is the upsampling.
            # The `inputs` for the concatenate layer are `encoder_X_final` and `decoder_Y_subsample`.
            # This means the upsampled result from *the level below* is used.

            # Let's adjust the `upsamples` list to align with the reversed loop and current_decoder_input.
            # The first `current_decoder_input` for the highest `level` in the reversed loop
            # is `encoder_outputs[-1]` (the bottleneck).

            # The upsample layer produces the input for the *next* (higher) decoder level.
            if level == self.nlevels - 2: # This is the first upsample from the bottleneck
                 upsampled = self.upsamples[0](current_decoder_input) # First upsample from bottleneck
            else:
                 # Need to index `self.upsamples` correctly for the reversed loop.
                 # If `nlevels = 5`, levels are 4, 3, 2, 1, 0.
                 # The `upsamples` list has `nlevels - 1` elements.
                 # The first element is for level `nlevels-2` (e.g., level 3 if nlevels=5)
                 # The last element is for level `0`.
                 upsampled_idx = (self.nlevels - 2) - level # For level 3, idx is 0. For level 0, idx is 3.
                 upsampled = self.upsamples[upsampled_idx](current_decoder_input)


            # Concatenate skip connection (from encoder) with upsampled feature map
            # This concatenation is `level_X_skip_connection` in the Keras code.
            # The Keras code has: 'inputs': ['encoder_X_final', 'decoder_Y_subsample']
            # So, `encoder_X_final` is `skip_connection`
            # `decoder_Y_subsample` is the `upsampled` feature from the *next lower* level's decoder.
            # The `current_decoder_input` at the start of each decoder loop is the `decoder_Y_subsample` for the *next* level's concatenation.
            # So, the input to the decoder block convolutions is the concatenation.

            concatenated_input = torch.cat((skip_connection, upsampled), dim=1)


            # Apply decoder block convolutions
            decoder_block_output = self.decoders[level](concatenated_input)


            # Add the upsampled feature map (before concatenation) to the output of the decoder block.
            # This corresponds to the `Add` layer after decoder convs in Keras.
            # The Keras code: 'inputs': ['decoder_level+1_subsample', curinputs]
            # `decoder_level+1_subsample` is `upsampled` in our current loop.
            # `curinputs` is `decoder_block_output`.
            final_decoder_output = self.decoder_add[level]([upsampled, decoder_block_output])


            current_decoder_input = final_decoder_output # This becomes the input for the next upsample

            if level == 0:
                # Final convolution and softmax
                output = self.final_conv(current_decoder_input)
                output = self.softmax(output)
                return output