from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    def before_train(self, runner):
        # Access the model and freeze the backbone
        backbone = runner.model.backbone
        for param in backbone.parameters():
            param.requires_grad = False
        runner.logger.info("Backbone (ViT) parameters frozen successfully")