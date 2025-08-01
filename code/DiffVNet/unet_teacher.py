import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Building blocks

class ConvBlock(nn.Module):
    """
    一个卷积块，具有可选的归一化和激活功能。

    该块执行卷积操作，之后是可选的归一化和激活操作。该块支持1D、2D和3D卷积。

    Parameters
    ----------
    ndims : int
        卷积的维度数（1、2或3）。
    input_dim : int
        输入的通道数。
    output_dim : int
        输出的通道数。
    kernel_size : int or tuple
        卷积核的大小。
    stride : int or tuple
        卷积的步幅。
    bias : bool
        是否在卷积中使用偏置项。
    padding : int or tuple, optional
        添加到输入的填充量，默认值为0。
    norm : str, optional
        要应用的归一化类型（'batch'、'instance' 或 'none'）
        默认值为 'none'。
    activation : str, optional
        使用的激活函数（'relu'、'lrelu'、'elu'、'prelu'、'selu'、'tanh' 或 'none'），默认值为 'relu'。
    pad_type : str, optional
        使用的填充类型（'zeros'、'reflect' 等），默认值为 'zeros'。

    """

    def __init__(
        self, ndims, input_dim, output_dim, kernel_size, stride, bias,
        padding=0, norm='none', activation='relu', pad_type='zeros',
    ):
        """
        初始化带有卷积、归一化和激活层的卷积块。

        参数在类的文档字符串中已描述。
        """
        super(ConvBlock, self).__init__()
        self.use_bias = bias
        assert ndims in [1, 2, 3], 'ndims in 1--3. found: %d' % ndims
        Conv = getattr(nn, 'Conv%dd' % ndims)

        # initialize convolution
        self.conv = Conv(
            input_dim, output_dim, kernel_size, stride, bias=self.use_bias,
            padding=padding, padding_mode=pad_type
        )

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = getattr(nn, 'BatchNorm%dd'%ndims)(norm_dim)
        elif norm == 'instance':
            self.norm = getattr(
                nn, 'InstanceNorm%dd'%ndims
            )(norm_dim, track_running_stats=False)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


    def forward(self, x):
        """
        执行ConvBlock的前向传播。

        应用卷积，随后可选的归一化和激活。

        Parameters
        ----------
        x : torch.Tensor
            输入到块的张量。

        Returns
        -------
        应用卷积、归一化和激活后的输出张量。
        """

        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def get_norm_layer(ndims, norm='batch'):
    """
    根据维度数量和归一化类型获取归一化层。

    Parameters
    ----------
    ndims : int
        归一化层的维度数量（1--3）。
    norm : str, optional
        要使用的归一化类型。
        可选项为 'batch'、'instance' 或 'none'。
        默认值为 'batch'。

    Returns
    -------
    Norm : torch.nn.Module or None
        相应的PyTorch归一化层，如果为'none'，则返回None。
    """

    if norm == 'batch':
        Norm = getattr(nn, 'BatchNorm%dd' % ndims)
    elif norm == 'instance':
        Norm = getattr(nn, 'InstanceNorm%dd' % ndims)
    elif norm == 'none':
        Norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return Norm


def get_actvn_layer(activation='relu'):
    """
    根据提供的激活类型获取激活函数层。

    Parameters
    ----------
    activation : str, optional
        要使用的激活函数类型。
        可选项为 'relu'、'lrelu'、'elu'、'prelu'、'selu'、'tanh' 或 'none'。
        默认值为 'relu'。

    Returns
    -------
    Activation : torch.nn.Module or None
        相应的PyTorch激活层，如果为'none'，则返回None。
    """

    if activation == 'relu':
        Activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        Activation = nn.LeakyReLU(0.3, inplace=True)
    elif activation == 'elu':
        Activation = nn.ELU()
    elif activation == 'prelu':
        Activation = nn.PReLU()
    elif activation == 'selu':
        Activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
        Activation = nn.Tanh()
    elif activation == 'none':
        Activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
    return Activation

################
# Network
################
class Unet(nn.Module):
    """
    U-Net架构用于图像到图像的转换。

    该类构建一个具有可配置深度、滤波器大小、归一化和激活层的U-Net。

    参数
    ----------
    dimension : int
        输入和卷积操作的维度数（1、2或3）。
    input_nc : int
        输入图像的通道数。
    output_nc : int
        输出图像的通道数。
    num_downs : int
        U-Net架构中的下采样操作数量。例如，如果 num_downs == 7，输入大小为128x128的图像将在瓶颈处变为1x1。（128x128 64x64 32x32 16x16 8x8 4x4 2x2 1x1）
    ngf : int, optional
        最后一层卷积层中的滤波器数量，默认为24。
    norm : str, optional
        使用的归一化类型（'batch'、'instance' 或 'none'），默认为 'batch'。
    final_act : str, optional
        输出层应用的激活函数，默认为 'none'。
    activation : str, optional
        隐藏层使用的激活函数（'relu'、'lrelu'、'elu' 等），默认为 'relu'。
    pad_type : str, optional
        卷积层使用的填充类型（'reflect'、'zero' 等），默认为 'reflect'。
    doubleconv : bool, optional
        是否在每个块中应用双重卷积，默认为 True。
    residual_connection : bool, optional
        是否在网络中添加残差连接，默认为 False。
    pooling : str, optional
        使用的池化类型（'Max' 或 'Avg'），默认为 'Max'。
    interp : str, optional
        解码器的上采样方法（'nearest' 或 'trilinear'），默认为 'nearest'。
    use_skip_connection : bool, optional
        是否在对应的编码器和解码器层之间使用跳跃连接，默认为 True。

    """

    def __init__(
        self, dimension, input_nc, output_nc, num_downs, ngf=24, norm='batch',
        final_act='none', activation='relu', pad_type='reflect', 
        doubleconv=True, residual_connection=False, 
        pooling='Max', interp='nearest', use_skip_connection=True, num_classes=2,
    ):
        """
        通过从最内层到最外层构建架构来初始化U-Net模型。

        参数在类的文档字符串中有所描述
        """
        super(Unet, self).__init__()
        # Check dims
        ndims = dimension
        assert ndims in [1, 2, 3], 'ndims should be 1--3. found: %d' % ndims
        
        # Decide whether to use bias based on normalization type
        use_bias = norm == 'instance'
        self.use_bias = use_bias

        # Get the appropriate convolution and pooling layers for the given dim
        Conv = getattr(nn, 'Conv%dd' % ndims)
        Pool = getattr(nn, '%sPool%dd' % (pooling,ndims))

        # Initialize normalization, activation, and final activation layers
        Norm = get_norm_layer(ndims, norm)
        Activation = get_actvn_layer(activation)
        FinalActivation = get_actvn_layer(final_act)

        self.residual_connection = residual_connection
        self.res_dest = []  # List to track destination layers for residuals 目标层
        self.res_source  = []  # List to track source layers for residuals 源层

        # Start creating the model
        model = [
            Conv(
                input_nc,
                ngf,
                3,
                stride=1,
                bias=use_bias,
                padding='same',
                padding_mode=pad_type,
            )
        ]
        self.res_source += [len(model)-1]
        if Norm is not None:
            model += [Norm(ngf)]
            
        if Activation is not None:
            model += [Activation]
        self.res_dest += [len(model) - 1]

        # Initialize encoder-related variables初始化与编码器相关的变量
        self.use_skip_connection = use_skip_connection
        self.encoder_idx = []
        in_ngf = ngf
        
        # Create the downsampling (encoder) blocks创建下采样（编码器）块
        for i in range(num_downs):
            if i == 0:
                mult = 1
            else:
                mult = 2
            model += [
                Conv(
                    in_ngf, in_ngf * mult, kernel_size=3, stride=1,
                    bias=use_bias, padding='same', padding_mode=pad_type,
                )
            ]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(in_ngf * mult)]

            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

            if doubleconv:
                model += [
                    Conv(
                        in_ngf * mult, in_ngf * mult, kernel_size=3, stride=1,
                        bias=use_bias, padding='same', padding_mode=pad_type,
                    )
                ]
                self.res_source += [len(model) - 1]
                if Norm is not None:
                    model += [Norm(in_ngf * mult)]
                if Activation is not None:
                    model += [Activation]
                self.res_dest += [len(model) - 1]

            self.encoder_idx += [len(model) - 1]
            model += [Pool(2)]
            in_ngf = in_ngf * mult


        model += [
            Conv(
                in_ngf, in_ngf * 2, kernel_size=3, stride=1, bias=use_bias,
                padding='same', padding_mode=pad_type,
            )
        ]
        self.res_source += [len(model) - 1]
        if Norm is not None:
            model += [Norm(in_ngf * 2)]

        if Activation is not None:
            model += [Activation]
        self.res_dest += [len(model) - 1]

        if doubleconv:
            #self.conv_id += [len(model)]
            model += [
                Conv(
                    in_ngf * 2, in_ngf * 2, kernel_size=3, stride=1,
                    bias=use_bias, padding='same', padding_mode=pad_type,
                )
            ]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(in_ngf * 2)]
    
            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

        # Create the upsampling (decoder) blocks
        self.decoder_idx = []
        mult = 2 ** (num_downs)
        for i in range(num_downs):
            self.decoder_idx += [len(model)]
            model += [nn.Upsample(scale_factor=2, mode=interp)]
            if self.use_skip_connection:  # concatenate encoder/decoder feature
                m = mult + mult // 2
            else:
                m = mult
            model += [
                Conv(
                    ngf * m, ngf * (mult // 2), kernel_size=3, stride=1,
                    bias=use_bias, padding='same', padding_mode=pad_type,
                )
            ]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(ngf * (mult // 2))]
            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

            if doubleconv:
                model += [
                    Conv(
                        ngf * (mult // 2),
                        ngf * (mult // 2),
                        kernel_size=3,
                        stride=1,
                        bias=use_bias,
                        padding='same',
                        padding_mode=pad_type,
                    )
                ]
                self.res_source += [len(model) - 1]
                if Norm is not None:
                    model += [Norm(ngf * (mult // 2))]
   
                if Activation is not None:
                    model += [Activation]
                self.res_dest += [len(model) - 1]

            mult = mult // 2

        print('Encoder skip connect id', self.encoder_idx)
        print('Decoder skip connect id', self.decoder_idx)

        Conv = getattr(nn, 'Conv%dd' % ndims)
        # final conv w/o normalization layer
        model += [
            Conv(
                ngf * mult,
                output_nc,
                kernel_size=3,
                stride=1,
                bias=use_bias,
                padding='same',
                padding_mode=pad_type,
            )
        ]
        if FinalActivation is not None:
            model += [FinalActivation]
        self.model = nn.Sequential(*model)

        # # 对logits加Dense（Linear）
        # self.logits_channels = output_nc
        # self.dense = nn.Linear(self.logits_channels, self.logits_channels)
        self.proj_to_classes = nn.Conv3d(
            in_channels=16,
            out_channels=num_classes,   # 这里要替换成你的num_classes变量
            kernel_size=1
        )
    def forward(self, input, layers=[], encode_only=False, verbose=False):
        # print('use unet model forward')
        if len(layers) == 0:
            """Standard forward"""
            enc_feats = []
            all_encoder_feats = []
            feat = input
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if self.residual_connection and layer_id in self.res_source:
                    feat_tmp = feat
                if self.residual_connection and layer_id in self.res_dest:
                    assert feat_tmp.size() == feat.size()
                    feat = feat + 0.1 * feat_tmp
                if self.use_skip_connection:
                    if layer_id in self.decoder_idx:
                        feat = torch.cat((enc_feats.pop(), feat), dim=1)
                    if layer_id in self.encoder_idx:
                        enc_feats.append(feat)
                        all_encoder_feats.append(feat)
            # # logits: [B, C, D, H, W]
            # B, C, D, H, W = feat.shape
            # feat_flat = feat.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # [B*D*H*W, C]
            # feat_dense = self.dense(feat_flat)  # [B*D*H*W, C]
            # feat_dense = feat_dense.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
            # return feat_dense, all_encoder_feats
            logits = self.proj_to_classes(feat)
            return logits, all_encoder_feats
        else:
            raise NotImplementedError 