from src.model.lightgbm import LightGBMHandler
from src.model.tab_inception_v3 import TabInceptionV3Handler, TabBinaryInceptionV3Handler 
from src.model.pytorch_frame_lgm import PTFrame_LightGBMHandler
from src.model.pytorch_frame_xgboost import PTFrame_XGBoostHandler
from src.model.inception_v3 import BinaryInceptionV3_Handler, InceptionV3_Handler

class ModelHandlerFactory:
    MODEL_HANDLERS = {
        "lightgbm": LightGBMHandler,
        "tab_inception_v3": TabInceptionV3Handler,
        "tab_binary_inception_v3": TabBinaryInceptionV3Handler,
        "pt_frame_lgm": PTFrame_LightGBMHandler,
        "pt_frame_xgboost": PTFrame_XGBoostHandler,
        "binary_inception_v3": BinaryInceptionV3_Handler,
        "inception_v3": InceptionV3_Handler
    }

    @staticmethod
    def get_handler(model_type):
        if model_type not in ModelHandlerFactory.MODEL_HANDLERS:
            raise ValueError(f"Unsupported model type: {model_type}")
        return ModelHandlerFactory.MODEL_HANDLERS[model_type]
