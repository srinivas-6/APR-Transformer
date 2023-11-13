from .posenet.PoseNet import PoseNet
from  .aprtransformer.APRTransformer import APRTransformer
from .dinotransposenet.DinoTransPoseNet import DinoTransPoseNet
from .dino.DinoResNet import DINOPoseNet

def get_model(model_name, config=None):
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param config: (dict) config file
    :return: instance of the model (nn.Module)
    """
    if model_name == 'posenet':
        return PoseNet(config)
    elif model_name == 'apr-transformer':
        return APRTransformer(config)
    elif model_name == 'dino-posenet':
        return DINOPoseNet(config)
    elif model_name == 'dino-transformer':
        return DinoTransPoseNet(config)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")