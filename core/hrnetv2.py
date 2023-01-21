from turtle import forward
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Upsample, ConvTranspose2d


def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight.data)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  if isinstance(m, nn.ConvTranspose2d):
      nn.init.kaiming_normal_(m.weight.data)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)

class ResidualBasic(torch.nn.Module):
    """
    Residual block
    input_width (int): Number of input conv channels,
    retained through sequentially conducted blocks in one branch.
    """
    def __init__(self, input_width):
        super().__init__()
        self.N = input_width
        self.conv1 = Conv2d(self.N, self.N, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(self.N, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(self.N, self.N, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(self.N, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o += x
        o = self.relu(o)
        return o

class FirstResidualBottleneck(torch.nn.Module):
    """
    1'st Residual block with bottleneck
    with hardcoded out channels,
    input_width (int): Number of input conv channels,
    which are squeezed into 64 for processing by 3x3 conv3
    and then unsqueezed back for output.
    """
    def __init__(self, input_width):
        super().__init__()
        self.N = input_width
        self.conv1 = Conv2d(self.N, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fcconv1 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.fcconv1bn1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu(o)
        o = self.conv3(o)
        o = self.bn3(o)
        o += self.fcconv1bn1(self.fcconv1(x))
        o = self.relu(o)
        return o

class ResidualBottleneck(torch.nn.Module):
    """
    Residual block with bottleneck
    input_width (int): Number of input conv channels,
    which are squeezed into 64 for processing by 3x3 conv3
    and then unsqueezed back for output.
    """
    def __init__(self, input_width):
        super().__init__()
        self.N = input_width
        self.conv1 = Conv2d(self.N, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(64, self.N, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(self.N, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu(o)
        o = self.conv3(o)
        o = self.bn3(o)
        o += x
        o = self.relu(o)
        return o

class Transition(torch.nn.Module):
    """
    Transition block
    input_width (int): Number of input conv channels (from upper branch).
    output_width (int): Number of output conv channels (to lower branch).
    downsample (bool): True when feature map is transited to lower branch. Default: None.
    """
    def __init__(self, input_width, output_width, downsample=None):
        super().__init__()
        self.i_N = input_width
        self.o_N = output_width
        self.downsample = downsample
        if self.downsample:
            self.conv1 = Conv2d(self.i_N, self.o_N, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            self.conv1 = Conv2d(self.i_N, self.o_N, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(self.o_N, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        return o

class Fusion(torch.nn.Module):
    """
    Fusion block
    input_width (int): Number of input conv channels (from feature map used for fusion)
    output_width (int): Number of output conv channels (from target feature map to fuse with)
    upsample (bool): True when feature map is fused with feature map from upper branch. Default: None.
    downsample (bool): True when feature map is fused with feature map from lower branch. Default: None.
    """
    def __init__(self, input_width, output_width, scale_factor=2, upsample=None, downsample=None):
        super().__init__()
        self.i_N = input_width
        self.o_N = output_width
        self.upsample = upsample
        self.downsample = downsample
        self.scale_factor = scale_factor
        if self.upsample:
            assert downsample == None
            self.conv1 = Conv2d(self.i_N, self.o_N, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.upsample1 = Upsample(scale_factor=self.scale_factor)
        if self.downsample:
            assert upsample == None
            self.conv1 = Conv2d(self.i_N, self.o_N, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.relu = torch.nn.ReLU(inplace=True)
        self.bn1 = BatchNorm2d(self.o_N, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        if self.upsample:
            o = self.conv1(x)
            o = self.bn1(o)
            o = self.upsample1(o)
        if self.downsample:
            o = self.conv1(x)
            o = self.bn1(o)
            o = self.relu(o)
        return o

class HRModule(torch.nn.Module):
    """Multi-Resolution Module for HRNet.
    In this module, every branch has 4 BasicBlocks/Bottlenecks.
    stage (int): Num of stage of HRNetv2.
    transition_stage (bool): True when HRModule of stage 3 or 4 should use Transition block,
    (stage 1 and 2 use it by default and controlled by this param). Default: None.
    """

    def __init__(self, stage, transition_stage=None):
        super().__init__()
        self.stage = stage
        self.transition_stage = transition_stage
        if self.stage == 1:
            self.s1_b1_block1 = FirstResidualBottleneck(64)
            self.s1_b1_block2 = ResidualBottleneck(256)
            self.s1_b1_block3 = ResidualBottleneck(256)
            self.s1_b1_block4 = ResidualBottleneck(256)
            # to 18
            self.transition_s1_256_18 = Transition(256, 18)
            # to 36
            self.transition_s1_256_36 = Transition(256, 36, downsample=True)
        if self.stage == 2:
            self.s2_b1_block1 = ResidualBasic(18)
            self.s2_b1_block2 = ResidualBasic(18)
            self.s2_b1_block3 = ResidualBasic(18)
            self.s2_b1_block4 = ResidualBasic(18)
            # to 36
            self.fuse_s2_18_36 = Fusion(18, 36, downsample=True)
            # to 72
            self.transition_s2_18 = Transition(18, 18, downsample=True)
            self.transition_s2_18_72 = Transition(18, 72, downsample=True)

            self.s2_b2_block1 = ResidualBasic(36)
            self.s2_b2_block2 = ResidualBasic(36)
            self.s2_b2_block3 = ResidualBasic(36)
            self.s2_b2_block4 = ResidualBasic(36)
            # to 18
            self.fuse_s2_36_18 = Fusion(36, 18, upsample=True)
            # to 72
            self.transition_s2_36_72 = Transition(36, 72, downsample=True)
        if self.stage == 3:
            self.s3_b1_block1 = ResidualBasic(18)
            self.s3_b1_block2 = ResidualBasic(18)
            self.s3_b1_block3 = ResidualBasic(18)
            self.s3_b1_block4 = ResidualBasic(18)
            # to 36
            self.down_fuse_s3_18_36 = Fusion(18, 36, downsample=True)
            # to 72
            self.down_fuse_s3_18 = Fusion(18, 18, downsample=True)
            self.down_fuse_s3_18_72 = Fusion(18, 72, downsample=True)
            

            self.s3_b2_block1 = ResidualBasic(36)
            self.s3_b2_block2 = ResidualBasic(36)
            self.s3_b2_block3 = ResidualBasic(36)
            self.s3_b2_block4 = ResidualBasic(36)
            # to 18
            self.up_fuse_s3_36_18 = Fusion(36, 18, upsample=True)
            # to 72
            self.down_fuse_s3_36_72 = Fusion(36, 72, downsample=True)

            self.s3_b3_block1 = ResidualBasic(72)
            self.s3_b3_block2 = ResidualBasic(72)
            self.s3_b3_block3 = ResidualBasic(72)
            self.s3_b3_block4 = ResidualBasic(72)
            # to 18
            self.up_fuse_s3_72_72 = Fusion(72, 72, upsample=True)
            self.up_fuse_s3_72_18 = Fusion(72, 18, upsample=True)
            # to 36
            self.up_fuse_s3_72_36 = Fusion(72, 36, upsample=True)

            if self.transition_stage:
                # 18 to 144
                self.transition_s3_18_36 = Transition(18, 36, downsample=True) #
                self.transition1_s3_36_72 = Transition(36, 72, downsample=True) #
                self.transition1_s3_72_144 = Transition(72, 144, downsample=True) #
                # 36 to 144
                self.transition2_s3_36_72 = Transition(36, 72, downsample=True) #
                self.transition2_s3_72_144 = Transition(72, 144, downsample=True) #
                # 72 to 144
                self.transition3_s3_72_144 = Transition(72, 144, downsample=True)
        if self.stage == 4:
            self.s4_b1_block1 = ResidualBasic(18)
            self.s4_b1_block2 = ResidualBasic(18)
            self.s4_b1_block3 = ResidualBasic(18)
            self.s4_b1_block4 = ResidualBasic(18)
            # to 36
            self.down_fuse_s4_18_36 = Fusion(18, 36, downsample=True)
            # to 72
            self.down1_fuse_s4_18 = Fusion(18, 18, downsample=True)
            self.down_fuse_s4_18_72 = Fusion(18, 72, downsample=True)
            # to 144
            self.down2_fuse_s4_18 = Fusion(18, 18, downsample=True)
            self.down1_fuse_s4_18_144 = Fusion(18, 18, downsample=True)
            self.down2_fuse_s4_18_144 = Fusion(18, 144, downsample=True)

            self.s4_b2_block1 = ResidualBasic(36)
            self.s4_b2_block2 = ResidualBasic(36)
            self.s4_b2_block3 = ResidualBasic(36)
            self.s4_b2_block4 = ResidualBasic(36)
            # to 18
            self.up_fuse_s4_36_18 = Fusion(36, 18, upsample=True)
            # to 72
            self.down_fuse_s4_36_72 = Fusion(36, 72, downsample=True)
            # to 144
            self.down2_fuse_s4_36 = Fusion(36, 36, downsample=True)
            self.down_fuse_s4_36_144 = Fusion(36, 144, downsample=True)

            self.s4_b3_block1 = ResidualBasic(72)
            self.s4_b3_block2 = ResidualBasic(72)
            self.s4_b3_block3 = ResidualBasic(72)
            self.s4_b3_block4 = ResidualBasic(72)
            # to 18
            self.up_fuse_s4_72_72 = Fusion(72, 72, upsample=True)
            self.up_fuse_s4_72_18 = Fusion(72, 18, upsample=True)
            # to 36
            self.up_fuse_s4_72_36 = Fusion(72, 36, upsample=True)
            # to 144
            self.up_fuse_s4_72_144 = Fusion(72, 144, downsample=True)

            self.s4_b4_block1 = ResidualBasic(144)
            self.s4_b4_block2 = ResidualBasic(144)
            self.s4_b4_block3 = ResidualBasic(144)
            self.s4_b4_block4 = ResidualBasic(144)
            # to 18
            self.up1_fuse_s4_144_144 = Fusion(144, 144, upsample=True)
            self.up2_fuse_s4_144_144 = Fusion(144, 144, upsample=True)
            self.up_fuse_s4_144_18 = Fusion(144, 18, upsample=True)
            # to 36
            self.up3_fuse_s4_144_144 = Fusion(144, 144, upsample=True)
            self.up_fuse_s4_144_36 = Fusion(144, 36, upsample=True)
            # to 72
            self.up_fuse_s4_144_72 = Fusion(144, 72, upsample=True)

        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        """
        x (list): list of inputs (len=1 at stage=1, len>1 at stages>1)
        """
        outputs = []
        if self.stage == 1:
            # at this stage, 1 HRModule should be initialized
            # x[0] = 64 channels feature map from backbone beginning
            o = self.s1_b1_block1(x[0])
            o = self.s1_b1_block2(o)
            o = self.s1_b1_block3(o)
            o = self.s1_b1_block4(o)
            # outputs of stage 1: [18 channels feature map, 
            #                       36 channels feature map]
            outputs.append(self.transition_s1_256_18(o))
            outputs.append(self.transition_s1_256_36(o))
        if self.stage == 2:
            # at this stage, 1 HRModule should be initialized
            # x[0] = 18 channels feature map from stage 1 output
            o = self.s2_b1_block1(x[0])
            o = self.s2_b1_block2(o)
            o = self.s2_b1_block3(o)
            o_18 = self.s2_b1_block4(o)

            # x[1] = 36 channels feature map from stage 1 output
            o = self.s2_b2_block1(x[1])
            o = self.s2_b2_block2(o)
            o = self.s2_b2_block3(o)
            o_36 = self.s2_b2_block4(o)

            # to 18
            outputs.append(self.fuse_s2_36_18(o) + o_18) #
            # to 36
            outputs.append(self.fuse_s2_18_36(o_18) + o_36)
            # fused (transited 18 to 72 + transited 36 to 72) 
            outputs.append(self.transition_s2_18_72(self.transition_s2_18(o_18)) \
                + self.transition_s2_36_72(o_36))

            # outputs of stage 2: [fused 18 channels feature map,
            #                       fused 36 channels feature map,
            #                       fused(transited 18 to 72 channels feature map 
            #                       + transited 36 to 72 channels feature map)]
        if self.stage == 3:
            # at this stage, 4 subsequent HRModules should be initialized in model definition
            # x[0] = fused 18 channels feature map
            o = self.s3_b1_block1(x[0])
            o = self.s3_b1_block2(o)
            o = self.s3_b1_block3(o)
            o_18 = self.s3_b1_block4(o)

            # x[1] = fused 36 channels feature map
            o = self.s3_b2_block1(x[1])
            o = self.s3_b2_block2(o)
            o = self.s3_b2_block3(o)
            o_36 = self.s3_b2_block4(o)

            # x[2] = fused(transited 18 to 72 channels feature map + transited 36 to 72 channels feature map 
            # (when first HRModule of 3 stage),
            # x[2] = fused 72 channels feature map (when 2,3 or 4 subsequent HRModule of 3 stage)
            o = self.s3_b3_block1(x[2])
            o = self.s3_b3_block2(o)
            o = self.s3_b3_block3(o)
            o_72 = self.s3_b3_block4(o)

            # to 18 (fused 36 and 72 to 18)
            outputs.append(self.up_fuse_s3_36_18(o_36) + self.up_fuse_s3_72_18(self.up_fuse_s3_72_72(o_72)))
            # to 36 (fused 18 and 72 to 36)
            outputs.append(self.down_fuse_s3_18_36(o_18) + self.up_fuse_s3_72_36(o_72))
            # to 72 (fused 18 and 36 to 72)
            outputs.append(self.down_fuse_s3_18_72(self.down_fuse_s3_18(o_18)) + self.down_fuse_s3_36_72(o_36))

            # outputs of each HRModule (except 4th) of stage 3: [fused 36 and 72 feature maps to 18 feature map,
            #                                                   fused 18 and 72 feature maps to 36 feature map,
            #                                                   fused 18 and 36 feature maps to 72 feature map]

            # last (4th HRmodule) of 3th stage should also fuse feature maps to 144 channels
            if self.transition_stage:
                # 18 to 144
                o_18_to_144 = self.transition1_s3_72_144(self.transition1_s3_36_72(self.transition_s3_18_36(o_18)))
                # 36 to 144
                o_36_to_144 = self.transition2_s3_72_144(self.transition2_s3_36_72(o_36))
                # 72 to 144
                o_72_to_144 = self.transition3_s3_72_144(o_72)
                o_144 = o_18_to_144 + o_36_to_144 + o_72_to_144
                outputs.append(o_144)
            
                # outputs of last (4th) HRmodule of stage 3 : [fused 36 and 72 feature maps to 18 feature map,
                #                                             fused 18 and 72 feature maps to 36 feature map,
                #                                             fused 18 and 36 feature maps to 72 feature map,
                #                                             fused 18, 36 and 72 feature maps and transited to 144 feature map]
        if self.stage == 4:
            # at this stage, 3 subsequent HRModules should be initialized in model definition
            # x[0] = fused 36 and 72 feature maps to 18 feature map
            o = self.s4_b1_block1(x[0])
            o = self.s4_b1_block2(o)
            o = self.s4_b1_block3(o)
            o_18 = self.s4_b1_block4(o)

            # x[1] = fused 18 and 72 feature maps to 36 feature map
            o = self.s4_b2_block1(x[1])
            o = self.s4_b2_block2(o)
            o = self.s4_b2_block3(o)
            o_36 = self.s4_b2_block4(o)

            # x[2] = fused 18 and 36 feature maps to 72 feature map
            o = self.s4_b3_block1(x[2])
            o = self.s4_b3_block2(o)
            o = self.s4_b3_block3(o)
            o_72 = self.s4_b3_block4(o)

            # x[3] = fused 18 and 36 feature maps to 72 feature map
            o = self.s4_b4_block1(x[3])
            o = self.s4_b4_block2(o)
            o = self.s4_b4_block3(o)
            o_144 = self.s4_b4_block4(o)

            # fused 18, 36, 72 and 144 feature maps into 18 feature map
            outputs.append(self.up_fuse_s4_36_18(o_36) \
                + self.up_fuse_s4_72_18(self.up_fuse_s4_72_72(o_72)) \
                    + self.up_fuse_s4_144_18(self.up2_fuse_s4_144_144(self.up1_fuse_s4_144_144(o_144)))+ o_18)
            
            # fused 18, 36, 72 and 144 feature maps into 36 feature map
            outputs.append(self.down_fuse_s4_18_36(o_18) \
                + self.up_fuse_s4_72_36(o_72) \
                    + self.up_fuse_s4_144_36(self.up3_fuse_s4_144_144(o_144)) + o_36)
            
            # fused 18, 36, 72 and 144 feature maps into 72 feature map
            outputs.append(self.down_fuse_s4_18_72(self.down1_fuse_s4_18(o_18)) \
                + self.down_fuse_s4_36_72(o_36) \
                    + self.up_fuse_s4_144_72(o_144) + o_72)
            
            # fused 18, 36, 72 and 144 feature maps into 144 feature map
            outputs.append(self.down2_fuse_s4_18_144(self.down1_fuse_s4_18_144(self.down2_fuse_s4_18(o_18))) \
                + self.down_fuse_s4_36_144(self.down2_fuse_s4_36(o_36)) \
                    + self.up_fuse_s4_72_144(o_72) + o_144)
            # outputs of each HRmodule of stage 4: [fused 18,36,72 and 144 feature maps into 18 feature map,
            #                                      fused 18,36,72 and 144 feature maps into 36 feature map,
            #                                      fused 18,36,72 and 144 feature maps into 72 feature map,
            #                                      fused 18,36,72 and 144 feature maps into 144 feature map]
        
        return outputs
    
class HRNetv2Head(torch.nn.Module):
    """
    Build HRNetv2 representation head to output mask prediction
    """

    def __init__(self):
        super().__init__()
        # upsample 36 to 18 spatial size (1/4 of original image), 
        # without changing channels count
        self.upsample_36 = Fusion(36, 36, upsample=True)

        # upsample 72 to 18 spatial size (1/4 of original image), 
        # without changing channels count
        self.upsample1_72 = Fusion(72, 72, upsample=True)
        self.upsample2_72 = Fusion(72, 72, upsample=True)

        # upsample 144 to 18 spatial size (1/4 of original image), 
        # without changing channels count
        self.upsample1_144 = Fusion(144, 144, upsample=True)
        self.upsample2_144 = Fusion(144, 144, upsample=True)
        self.upsample3_144 = Fusion(144, 144, upsample=True)

        # after feature maps concatenation channels should be 270 (15C, while C=18)
        #self.upsample1_output = Upsample(scale_factor=2)
        #self.upsample2_output = Upsample(scale_factor=2)
        self.upsample1_output = ConvTranspose2d(270, 270, kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = BatchNorm2d(270, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv = Conv2d(270, 270, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn = BatchNorm2d(270, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activate = torch.nn.ReLU(inplace=True)
        # predictions: mask([0]) and borders([1])
        self.conv_seg  = Conv2d(270, 2, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        # x[0] is 18 feature map (1/4 spatial size of original image)
        # upsample 36 to 18 spatial size (1/4 of original image), 
        # without changing channels count
        u_36 = self.upsample_36(x[1])
        # upsample 72 to 18 spatial size (1/4 of original image), 
        # without changing channels count
        u_72 = self.upsample2_72(self.upsample1_72(x[2]))
        # upsample 144 to 18 spatial size (1/4 of original image), 
        # without changing channels count
        u_144 = self.upsample3_144(self.upsample2_144(self.upsample1_144(x[3])))
        # after feature maps concatenation channels should be 270 (15C, while C=18)
        o = torch.cat([x[0], u_36, u_72, u_144], dim=1)
        o = self.upsample1_output(o)
        o = self.bn1(o)
        o = self.activate(o)
        o = self.conv(o)
        o = self.bn(o)
        o = self.activate(o)
        # output mask with shape 2xHxW (H and W - original input image height and width)
        # 2 channels - mask and borders
        o = self.conv_seg(o)

        return o

class HRNetv2(torch.nn.Module):
    """
    Build HRNetv2 backbone
    """

    def __init__(self, img_channels):
        super().__init__()
        self.img_channels = img_channels
        # input stem to downscale image to 1/4 of original size
        self.conv1 = Conv2d(img_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)

        # stage 1 containing 1 HRModule
        self.stage_1 = HRModule(stage=1)

        # stage 2 containing 1 HRModule
        self.stage_2 = HRModule(stage=2)

        # stage 3 containing 4 HRModule's
        self.stage_3_1 = HRModule(stage=3)
        self.stage_3_2 = HRModule(stage=3)
        self.stage_3_3 = HRModule(stage=3)
        # HRModule with transition stage (transiting 144 feature maps to stage 4)
        self.stage_3_4 = HRModule(stage=3, transition_stage=True)

        # stage 4 containing 3 HRModule's
        self.stage_4_1 = HRModule(stage=4)
        self.stage_4_2 = HRModule(stage=4)
        self.stage_4_3 = HRModule(stage=4)

        self.decode_head = HRNetv2Head()
    def forward(self, x):
        # input stem to downscale image to 1/4 of original size
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu(o)

        # append output feature map of stem to list,
        # because HRModule works with list of values
        input_list = []
        input_list.append(o)

        outputs_list = self.stage_1(input_list)
        # outputs of stage 1: [18 channels feature map, 
        #                       36 channels feature map]

        outputs_list = self.stage_2(outputs_list)
        # outputs of stage 2: [fused 18 channels feature map,
        #                       fused 36 channels feature map,
        #                       fused(transited 18 to 72 channels feature map 
        #                       + transited 36 to 72 channels feature map)]

        outputs_list = self.stage_3_1(outputs_list)
        outputs_list = self.stage_3_2(outputs_list)
        outputs_list = self.stage_3_3(outputs_list)
        # outputs of each HRModule (except 4th) of stage 3: [fused 36 and 72 feature maps to 18 feature map,
        #                                                   fused 18 and 72 feature maps to 36 feature map,
        #                                                   fused 18 and 36 feature maps to 72 feature map]
        outputs_list = self.stage_3_4(outputs_list)
        # outputs of last (4th) HRmodule of stage 3 : [fused 36 and 72 feature maps to 18 feature map,
        #                                             fused 18 and 72 feature maps to 36 feature map,
        #                                             fused 18 and 36 feature maps to 72 feature map,
        #                                             fused 18, 36 and 72 feature maps and transited to 144 feature map]

        outputs_list = self.stage_4_1(outputs_list)
        outputs_list = self.stage_4_2(outputs_list)
        outputs_list = self.stage_4_3(outputs_list)
        # outputs of each HRmodule of stage 4: [fused 18,36,72 and 144 feature maps into 18 feature map,
        #                                      fused 18,36,72 and 144 feature maps into 36 feature map,
        #                                      fused 18,36,72 and 144 feature maps into 72 feature map,
        #                                      fused 18,36,72 and 144 feature maps into 144 feature map]

        o = self.decode_head(outputs_list)

        return o