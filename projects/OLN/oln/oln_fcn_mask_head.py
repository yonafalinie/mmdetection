from torch import Tensor

from mmdet.models import FCNMaskHead
from mmdet.registry import MODELS


@MODELS.register_module()
class OLNFCNMaskHead(FCNMaskHead):
    def forward(self, x: Tensor):
        """Forward features from the upstream network.

        Args:
            x (Tensor): Extract mask RoI features.

        Returns:
            Tensor: Predicted foreground masks.
        """
        for conv in self.convs:
            x = conv(x)
        _x = x
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_preds = self.conv_logits(x)
        return mask_preds, _x
