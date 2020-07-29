from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56
from .resnet import resnet110, resnet8x4, resnet8x4_double, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_16_2_double
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .vgg_double import vgg8 as vgg8_double, vgg13 as vgg13_double
from .mobilenetv2 import mobile_half, mobile_half_double
from .ShuffleNetv1 import ShuffleV1, ShuffleV1Double
from .ShuffleNetv2 import ShuffleV2, ShuffleV2Double

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet8x4_double': resnet8x4_double,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_16_2_double': wrn_16_2_double,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg8_double': vgg8_double,
    'vgg13_double': vgg13_double,
    'MobileNetV2': mobile_half,
    'MobileNetV2Double': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV1Double': ShuffleV1Double,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2Double': ShuffleV2Double,
}
