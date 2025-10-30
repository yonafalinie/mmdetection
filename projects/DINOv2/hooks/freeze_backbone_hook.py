from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    def before_train(self, runner):
        model = runner.model
        # Freeze the backbone parameters
        for param in model.backbone.parameters():
            param.requires_grad = False


# Register the hook
custom_hooks = [
    dict(type='FreezeBackboneHook')
]