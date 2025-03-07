from torch_frame.nn import Trompt, ExcelFormer, MLP, TabNet, TabTransformer, FTTransformer

from src.model.lightgbm import LightGBMHandler
from src.model.tab_inception_v3 import TabInceptionV3Handler, TabBinaryInceptionV3Handler 
from src.model.pytorch_frame_lgm import PTFrame_LightGBMHandler
from src.model.pytorch_frame_xgboost import PTFrame_XGBoostHandler
from src.model.inception_v3 import BinaryInceptionV3_Handler, InceptionV3_Handler
from src.model.pytorch_frame_handler import TorchFrameHandler

class ModelHandlerFactory:
    MODEL_HANDLERS = {
        "LightGBM": LightGBMHandler,
        "tab_inception_v3": TabInceptionV3Handler,
        "tab_binary_inception_v3": TabBinaryInceptionV3Handler,
        "pt_frame_lgm": PTFrame_LightGBMHandler,
        "pt_frame_xgboost": PTFrame_XGBoostHandler,
        "binary_inception_v3": BinaryInceptionV3_Handler,
        "inception_v3": InceptionV3_Handler,
    }

    # Dynamically handle all PyTorch Frame models
    TORCH_FRAME_MODELS = {
        "Trompt": Trompt,
        "MLP": MLP,
        "ExcelFormer": ExcelFormer,
        "TabNet": TabNet,
        "TabTransformer": TabTransformer,
        "FTTransformer": FTTransformer,
    }

    @staticmethod
    def get_handler(args):
        model_type = args.model_type
        if model_type in ModelHandlerFactory.MODEL_HANDLERS:
            return ModelHandlerFactory.MODEL_HANDLERS[model_type](args)
        
        if model_type in ModelHandlerFactory.TORCH_FRAME_MODELS:
            model_class = ModelHandlerFactory.TORCH_FRAME_MODELS[model_type]
            return TorchFrameHandler(args, model_class)
        
        raise ValueError(f"Unsupported model type: {model_type}")