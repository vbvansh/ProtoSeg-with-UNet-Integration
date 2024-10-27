from typing import Optional
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import Pytorch_UNet_master.unet.unet_model as UN
from settings import log
from receptive_field import compute_proto_layer_rf_info_v2

base_architecture_to_features = {
    'deeplabv2_resnet101': lambda **kwargs: UN.UNet(n_channels=3, n_classes=19)
}

@gin.configurable(allowlist=['bottleneck_stride', 'patch_classification'])
class PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 bottleneck_stride: Optional[int] = None,
                 patch_classification: bool = False):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.epsilon = 1e-4
        self.bottleneck_stride = bottleneck_stride
        self.patch_classification = patch_classification
        # self.features = features
        # self.channel_adjustment = nn.Conv2d(19, 512, kernel_size=1)
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape), requires_grad=True)
        self.prototype_activation_function = prototype_activation_function

        # Initialize prototype class identities
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        num_prototypes_per_class = self.num_prototypes // num_classes
        for i in range(num_classes):
            self.prototype_class_identity[i * num_prototypes_per_class:(i + 1) * num_prototypes_per_class, i] = 1

        self.num_prototypes_per_class = num_prototypes_per_class
        self.proto_layer_rf_info = proto_layer_rf_info

        # Initialize features (UNet)
        self.features = features
        self.channel_adjustment = nn.Conv2d(19, 512, kernel_size=1)
   
        # Determine input channels for add-on layers
        if isinstance(self.features, UN.UNet):
            first_add_on_layer_in_channels = 512  # UNet's output channels
        else:
            last_conv = [m for m in features.modules() if isinstance(m, nn.Conv2d)][-1]
            first_add_on_layer_in_channels = last_conv.out_channels

        # Build add-on layers
        add_on_layers = []
        if add_on_layers_type == 'bottleneck':
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.extend([
                    nn.Conv2d(current_in_channels, current_out_channels, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(current_out_channels, current_out_channels, kernel_size=1),
                    nn.Sigmoid() if current_out_channels == self.prototype_shape[1] else nn.ReLU()
                ])
                current_in_channels = current_in_channels // 2
                if current_out_channels == self.prototype_shape[1]:
                    break

        self.add_on_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )        
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape), requires_grad=True)
        if init_weights:
            self._initialize_weights()

    @property
    def prototype_shape(self):
        return self.prototype_vectors.shape

    @property
    def num_prototypes(self):
        return self.prototype_vectors.shape[0]

    @property
    def num_classes(self):
        return self.prototype_class_identity.shape[1]

    def conv_features(self, x):
        x = self.features(x)
        x = self.channel_adjustment(x)
        
        x = self.add_on_layers(x)
        return x

    def _l2_convolution(self, x):
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        
        if self.patch_classification:
            batch_size, num_prototypes, h, w = distances.shape
            distances_reshaped = distances.permute(0, 2, 3, 1).reshape(-1, num_prototypes)
            prototype_activations = self.distance_2_similarity(distances_reshaped)
            logits = self.last_layer(prototype_activations)
            return logits.reshape(batch_size, h, w, -1)
        else:
            min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
            min_distances = min_distances.view(-1, self.num_prototypes)
            prototype_activations = self.distance_2_similarity(min_distances)
            logits = self.last_layer(prototype_activations)
            return logits

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        self.last_layer.weight.data.copy_(
            1 * positive_one_weights_locations +
            incorrect_strength * negative_one_weights_locations)

@gin.configurable(denylist=['img_size'])
def construct_PPNet(
        img_size=224,
        base_architecture=gin.REQUIRED,
        pretrained=True,
        prototype_shape=(2000, 512, 1, 1),
        num_classes=200,
        prototype_activation_function='log',
        add_on_layers_type='bottleneck'
):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = [], [], []

    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2]
    )

    return PPNet(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type
    )

if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Initialize UNet model with required arguments
    unet_model = UN.UNet(n_channels=3, n_classes=512)
    
    # Compute proto_layer_rf_info
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=256,
        layer_filter_sizes=[3, 3, 3, 3, 3],
        layer_strides=[1, 2, 2, 2, 2],
        layer_paddings=[1, 1, 1, 1, 1],
        prototype_kernel_size=1
    )
    
    # Initialize PPNet with the UNet
    model = PPNet(
        features=unet_model,
        img_size=256,
        prototype_shape=(1995, 512, 1, 1),
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=19,
        prototype_activation_function='log'
    ).to(device)
    
    # Run forward pass
    output = model(dummy_input)
    print("Model output shape:", output.shape)