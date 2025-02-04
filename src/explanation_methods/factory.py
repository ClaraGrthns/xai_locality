from src.explanation_methods.gradient import IntegratedGradientsHandler, SmoothGradHandler
from src.explanation_methods.lime import LimeHandler
class ExplanationMethodHandlerFactory:
    METHOD_HANDLERS = {
        "IG": IntegratedGradientsHandler,
        "IG+SmoothGrad": SmoothGradHandler,
        "LIME": LimeHandler
    }

    @staticmethod
    def get_handler(method, **kwargs):
        if method not in ExplanationMethodHandlerFactory.METHOD_HANDLERS:
            raise ValueError(f"Unsupported explanation method: {method}")
        return ExplanationMethodHandlerFactory.METHOD_HANDLERS[method](**kwargs)
