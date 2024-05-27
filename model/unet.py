import torch
import torch.nn as nn
from .eff_encoder import EfficientNetEncoder
from .decoder import UnetDecoder
from .head import SegmentationHead
from typing import Optional, Union, List

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class unet(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b7",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = "scse",
        in_channels: int = 1,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = "sigmoid",
    ):
        super().__init__()
        eff_params=efficient_net_encoders[encoder_name]['params']
        eff_params['depth']=encoder_depth
        eff_params['in_channels'] = in_channels
        if encoder_weights:
            eff_params['weights']=encoder_weights

        self.encoder = EfficientNetEncoder(**eff_params)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center= False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        self.name = "u-{}".format(encoder_name)
        self.initialize()


    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)


    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x,extract_features=True):


        if extract_features:
            self.check_input_shape(x)
            features = self.encoder(x)
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)
        else:
            decoder_output = self.decoder(*x)
            masks = self.segmentation_head(decoder_output)


        return masks

    @torch.no_grad()
    def predict(self, x):

        if self.training:
            self.eval()

        x = self.forward(x)

        return x


efficient_net_encoders = {

    "efficientnet-b4": {
        "params": {
            "out_channels": (3, 48, 32, 56, 160, 448),
            "stage_idxs": (6, 10, 22, 32),
            "model_name": "efficientnet-b4",
        },
    },

    "efficientnet-b7": {
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (11, 18, 38, 55),
            "model_name": "efficientnet-b7",
        },
    },
}