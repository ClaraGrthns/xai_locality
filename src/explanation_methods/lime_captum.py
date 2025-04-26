from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import LimeBase
import torch
from src.explanation_methods.lime_analysis.lime_captum_local_classifier import (compute_feature_attributions, 
                                                                                compute_lime_preds_for_all_kNN,
                                                                                compute_lime_regressionpreds_for_all_kNN)
from torch import Tensor

import os.path as osp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
import time 
from src.utils.sampling import uniform_ball_sample

from captum.attr._core.lime import LimeBase
#!/usr/bin/env python3
import inspect
import math
import warnings
from typing import Any, Callable, cast, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
)
from captum._utils.progress import progress
from captum._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.log import log_usage
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

class LimeCaptumHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = LimeWithBias(model)

    def explain_instance(self, **kwargs):
        coefs, bias = self.explainer.attribute(kwargs["input"], target=kwargs["target"])
        return coefs, bias
    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        tst_feat_for_expl_loader = DataLoader(tst_data, batch_size=self.args.chunk_size, shuffle=False)
        device = torch.device("cpu")
        feature_attribution_folder = osp.join(results_path,
                                    "feature_attribution")
        bias_feature_attribution_file_path = osp.join(feature_attribution_folder, f"bias_feature_attribution_{self.args.gradient_method}_random_seed-{self.args.random_seed}.h5")
        coefs_feature_attribution_file_path = osp.join(feature_attribution_folder, f"coefs_feature_attribution_{self.args.gradient_method}_random_seed-{self.args.random_seed}.h5")

        bias_feature_attributions = None
        coefs_feature_attributions = None

        print("Looking for LIME explanations (coefficients and bias) in: ", feature_attribution_folder)

        # Check if both files exist and force is not set
        files_exist = osp.exists(bias_feature_attribution_file_path) and osp.exists(coefs_feature_attribution_file_path)
        should_load = files_exist and (not self.args.force or self.args.downsample_analysis != 1.0)

        if should_load:
            print(f"Using precomputed LIME explanations from: {feature_attribution_folder}")
            try:
                with h5py.File(bias_feature_attribution_file_path, "r") as f:
                    bias_feature_attributions = f["bias_feature_attribution"][:]
                bias_feature_attributions = torch.tensor(bias_feature_attributions).float().to(device)

                with h5py.File(coefs_feature_attribution_file_path, "r") as f:
                    coefs_feature_attributions = f["coefs_feature_attribution"][:]
                coefs_feature_attributions = torch.tensor(coefs_feature_attributions).float().to(device)
                print("Successfully loaded precomputed explanations.")
            except Exception as e:
                print(f"Error loading precomputed explanations: {e}. Recomputing...")
                bias_feature_attributions = None
                coefs_feature_attributions = None
                should_load = False # Force recomputation

        if not should_load:
            print("Precomputed LIME explanations not found or loading failed/forced. Computing explanations for the test set...")
            if not osp.exists(feature_attribution_folder):
                os.makedirs(feature_attribution_folder)

            coefs, biases = compute_feature_attributions(self.explainer, predict_fn, tst_feat_for_expl_loader)

            coefs_feature_attributions = coefs.float().to(device)
            bias_feature_attributions = biases.float().to(device)
            print(f"Saving computed bias to: {bias_feature_attribution_file_path}")
            with h5py.File(bias_feature_attribution_file_path, "w") as f:
                f.create_dataset("bias_feature_attribution", data=bias_feature_attributions.cpu().numpy())
            print(f"Saving computed coefficients to: {coefs_feature_attribution_file_path}")
            with h5py.File(coefs_feature_attribution_file_path, "w") as f:
                f.create_dataset("coefs_feature_attribution", data=coefs_feature_attributions.cpu().numpy())
            print("Finished computing and saving explanations.")
        if coefs_feature_attributions is None or bias_feature_attributions is None:
             raise RuntimeError("Failed to load or compute LIME explanations.")
        return coefs_feature_attributions, bias_feature_attributions
    
    def get_experiment_setting(self, fractions, max_radius):
        df_setting = "dataset_test"
        df_setting += "_val" if self.args.include_val else ""
        df_setting += "_trn" if self.args.include_trn else ""
        setting = f"{self.args.method}_{df_setting}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_random_seed-{self.args.random_seed}_accuracy_fraction"
        # else:
        #     setting = f"grad_method-{self.args.gradient_method}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_accuracy_fraction"
        if self.args.downsample_analysis != 1.0:
            setting = f"downsample-{np.round(self.args.downsample_analysis, 2)}_" + setting
        if self.args.sample_around_instance:
            setting = f"sampled_at_point_max_R-{np.round(max_radius, 2)}_" + setting
        else:
            setting = f"fractions-0-{np.round(fractions, 2)}_"+setting   
        if self.args.regression:
            setting = "regression_" + setting
        return setting


    def process_chunk(self, 
                      batch, 
                      tst_chunk_dist, 
                      df_feat_for_expl, 
                      explanations_chunk, 
                      predict_fn, 
                      n_points_in_ball, 
                      tree, 
                      max_radius):
        """
        Process a single chunk of data for gradient-based methods.
        """
        tst_chunk = batch  # For gradient methods, batch is already in the right format
        proba_output = self.args.model_type in ["LightGBM", "XGBoost", "LightGBM", "pt_frame_xgb", "LogReg"]
        dist, idx = tree.query(tst_chunk_dist, k=n_points_in_ball, return_distance=True, sort_results=True)
        dist = np.array(dist)
        # 1. Get all the kNN samples from the analysis dataset
        samples_in_ball = [[df_feat_for_expl[idx] for idx in row] for row in idx]
        samples_in_ball = torch.stack([torch.stack(row, dim=0) for row in samples_in_ball], dim=0)  

        if self.args.regression:
            model_preds, local_preds = compute_lime_regressionpreds_for_all_kNN(
               explanation = explanations_chunk, 
                predict_fn = predict_fn, 
                samples_in_ball = samples_in_ball,
            )
            return model_preds, local_preds, dist
        else:
            with torch.no_grad():
                predictions = predict_fn(tst_chunk)
            if not proba_output:
                if predictions.shape[-1] == 1:
                    predictions_sm = torch.sigmoid(predictions)
                    predictions_sm = torch.cat([1 - predictions_sm, predictions_sm], dim=-1)
                else:
                    predictions_sm = torch.softmax(predictions, dim=-1)
            else:
                if predictions.shape[-1] == 1:
                    predictions_sm = torch.cat([1 - predictions, predictions], dim=-1)
                else:
                    predictions_sm = predictions
            top_labels = torch.argmax(predictions_sm, dim=1).tolist()
            model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs = compute_lime_preds_for_all_kNN(
                explanation = explanations_chunk,
                predict_fn = predict_fn, 
                samples_in_ball = samples_in_ball,
                top_labels = top_labels, 
                pred_threshold=None,
                proba_output=proba_output
            )
            return model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, dist
 

class LimeWithBias(LimeBase):
    """
    A wrapper class for captum.attr.LimeBase that modifies the attribute
    method to return both the interpretable model representation (e.g., coefficients)
    and the bias (intercept) term of the fitted surrogate model.

    Assumes the provided `interpretable_model` stores the fitted model
    in a way that the intercept can be accessed (e.g., via a `.model.intercept_`
    attribute, common when using SkLearnLinearModel wrapper).
    """

    def __init__(self, forward_func, interpretable_model, **kwargs):
        """
        Initializes LimeWithBias.

        Args:
            forward_func (callable): The forward function of the model or
                        any modification of it.
            interpretable_model (callable): A function or model instance that
                        takes inputs (interpretable input, original output,
                        weights), trains an interpretable model, and returns
                        a representation of the interpretable model. Common examples
                        include wrappers around sklearn.linear_model (e.g.,
                        captum._utils.models.linear_model.SkLearnLinearModel).
            **kwargs: Additional arguments are passed to the LimeBase constructor.
                      Refer to LimeBase documentation for details (similarity_func,
                      perturb_func, etc.).
        """
        super().__init__(forward_func, interpretable_model, **kwargs)
    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above.
        It trains an interpretable model and returns a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        A batch of inputs can be provided as inputs only if forward_func
        returns a single value per batch (e.g. loss).
        The interpretable feature representation should still have shape
        1 x num_interp_features, corresponding to the interpretable
        representation for the full batch, and perturbations_per_eval
        must be set to 1.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. For all other types,
                        the given argument is used for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_samples (int, optional): The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False
            **kwargs (Any, optional): Any additional arguments necessary for
                        sampling and transformation functions (provided to
                        constructor).
                        Default: None

        Returns:
            **interpretable model representation**:
            - **interpretable model representation** (*Any*):
                    A representation of the interpretable model trained. The return
                    type matches the return type of train_interpretable_model_func.
                    For example, this could contain coefficients of a
                    linear surrogate model.

        Examples::

            >>> # SimpleClassifier takes a single input tensor of
            >>> # float features with size N x 5,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>>
            >>> # We will train an interpretable model with the same
            >>> # features by simply sampling with added Gaussian noise
            >>> # to the inputs and training a model to predict the
            >>> # score of the target class.
            >>>
            >>> # For interpretable model training, we will use sklearn
            >>> # linear model in this example. We have provided wrappers
            >>> # around sklearn linear models to fit the Model interface.
            >>> # Any arguments provided to the sklearn constructor can also
            >>> # be provided to the wrapper, e.g.:
            >>> # SkLearnLinearModel("linear_model.Ridge", alpha=2.0)
            >>> from captum._utils.models.linear_model import SkLearnLinearModel
            >>>
            >>>
            >>> # Define similarity kernel (exponential kernel based on L2 norm)
            >>> def similarity_kernel(
            >>>     original_input: Tensor,
            >>>     perturbed_input: Tensor,
            >>>     perturbed_interpretable_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         # kernel_width will be provided to attribute as a kwarg
            >>>         kernel_width = kwargs["kernel_width"]
            >>>         l2_dist = torch.norm(original_input - perturbed_input)
            >>>         return torch.exp(- (l2_dist**2) / (kernel_width**2))
            >>>
            >>>
            >>> # Define sampling function
            >>> # This function samples in original input space
            >>> def perturb_func(
            >>>     original_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         return original_input + torch.randn_like(original_input)
            >>>
            >>> # For this example, we are setting the interpretable input to
            >>> # match the model input, so the to_interp_rep_transform
            >>> # function simply returns the input. In most cases, the interpretable
            >>> # input will be different and may have a smaller feature set, so
            >>> # an appropriate transformation function should be provided.
            >>>
            >>> def to_interp_transform(curr_sample, original_inp,
            >>>                                      **kwargs):
            >>>     return curr_sample
            >>>
            >>> # Generating random input with size 1 x 5
            >>> input = torch.randn(1, 5)
            >>> # Defining LimeBase interpreter
            >>> lime_attr = LimeBase(net,
                                     SkLearnLinearModel("linear_model.Ridge"),
                                     similarity_func=similarity_kernel,
                                     perturb_func=perturb_func,
                                     perturb_interpretable_space=False,
                                     from_interp_rep_transform=None,
                                     to_interp_rep_transform=to_interp_transform)
            >>> # Computes interpretable model, returning coefficients of linear
            >>> # model.
            >>> attr_coefs = lime_attr.attribute(input, target=1, kernel_width=1.1)
        """
        with torch.no_grad():
            inp_tensor = (
                cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
            )
            device = inp_tensor.device

            interpretable_inps = []
            similarities = []
            outputs = []

            curr_model_inputs = []
            expanded_additional_args = None
            expanded_target = None
            perturb_generator = None
            if inspect.isgeneratorfunction(self.perturb_func):
                perturb_generator = self.perturb_func(inputs, **kwargs)

            if show_progress:
                attr_progress = progress(
                    total=math.ceil(n_samples / perturbations_per_eval),
                    desc=f"{self.get_name()} attribution",
                )
                attr_progress.update(0)

            batch_count = 0
            for _ in range(n_samples):
                if perturb_generator:
                    try:
                        curr_sample = next(perturb_generator)
                    except StopIteration:
                        warnings.warn(
                            "Generator completed prior to given n_samples iterations!"
                        )
                        break
                else:
                    curr_sample = self.perturb_func(inputs, **kwargs)
                batch_count += 1
                if self.perturb_interpretable_space:
                    interpretable_inps.append(curr_sample)
                    curr_model_inputs.append(
                        self.from_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                else:
                    curr_model_inputs.append(curr_sample)
                    interpretable_inps.append(
                        self.to_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                curr_sim = self.similarity_func(
                    inputs, curr_model_inputs[-1], interpretable_inps[-1], **kwargs
                )
                similarities.append(
                    curr_sim.flatten()
                    if isinstance(curr_sim, Tensor)
                    else torch.tensor([curr_sim], device=device)
                )

                if len(curr_model_inputs) == perturbations_per_eval:
                    if expanded_additional_args is None:
                        expanded_additional_args = _expand_additional_forward_args(
                            additional_forward_args, len(curr_model_inputs)
                        )
                    if expanded_target is None:
                        expanded_target = _expand_target(target, len(curr_model_inputs))

                    model_out = self._evaluate_batch(
                        curr_model_inputs,
                        expanded_target,
                        expanded_additional_args,
                        device,
                    )

                    if show_progress:
                        attr_progress.update()

                    outputs.append(model_out)

                    curr_model_inputs = []

            if len(curr_model_inputs) > 0:
                expanded_additional_args = _expand_additional_forward_args(
                    additional_forward_args, len(curr_model_inputs)
                )
                expanded_target = _expand_target(target, len(curr_model_inputs))
                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )
                if show_progress:
                    attr_progress.update()
                outputs.append(model_out)

            if show_progress:
                attr_progress.close()

            combined_interp_inps = torch.cat(interpretable_inps).float()
            combined_outputs = (
                torch.cat(outputs)
                if len(outputs[0].shape) > 0
                else torch.stack(outputs)
            ).float()
            combined_sim = (
                torch.cat(similarities)
                if len(similarities[0].shape) > 0
                else torch.stack(similarities)
            ).float()
            dataset = TensorDataset(
                combined_interp_inps, combined_outputs, combined_sim
            )
            self.interpretable_model.fit(DataLoader(dataset, batch_size=batch_count))
            return self.interpretable_model.representation(), self.interpretable_model.model.bias()


