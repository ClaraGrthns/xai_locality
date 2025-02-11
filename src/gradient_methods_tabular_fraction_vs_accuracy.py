import os.path as osp
import numpy as np
import argparse
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import os
import h5py
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch

from src.utils.misc import get_non_zero_cols, set_random_seeds, get_path
from captum.attr import IntegratedGradients, NoiseTunnel
from src.explanation_methods.gradient_methods.local_classifier import compute_gradmethod_accuracy_per_fraction, compute_saliency_maps
from src.model.factory import ModelHandlerFactory
from src.explanation_methods.factory import ExplanationMethodHandlerFactory

from sklearn.metrics.pairwise import cosine_similarity

def cosine_distance(x, y):
    cosine_sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
    return 1 - cosine_sim

def validate_distance_measure(distance_measure):
    valid_distance_measures = BallTree.valid_metrics + ["cosine"]
    assert distance_measure in valid_distance_measures, f"Invalid distance measure: {distance_measure}. Valid options are: {valid_distance_measures}"


def main(args):
    print(f"Running analysis, with following arguments: {args}")
    set_random_seeds(args.random_seed)

    results_path = get_path(args.results_folder, args.results_path, args.setting)
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)
    
    model_handler = ModelHandlerFactory.get_handler(args.model_type)
    model = model_handler.model
    dataset = model_handler.load_data()
    predict_fn = model_handler.predict_fn

    if args.method == "lime" and args.kernel_width is None:
        args.kernel_width = np.round(np.sqrt(dataset[4].shape[1]) * .75, 2)  # Default value
    
    explanation_handler = ExplanationMethodHandlerFactory.get_handler(args.method)
    explanation_handler.set_explainer(forward_func=model, 
                                      multiply_by_inputs=False,
                                      dataset=dataset,
                                      class_names=model_handler.get_class_names(),
                                      args = args,
                                    )

    tst_feat_for_dist, df_feat_for_dist, tst_feat_for_expl, df_feat_for_expl = explanation_handler.process_data(dataset, model_handler, args)
    explanations = explanation_handler.compute_explanations(results_path=results_path, 
                                                            predict_fn=predict_fn, 
                                                            tst_data=tst_feat_for_expl, 
                                                            args=args)
    
    validate_distance_measure(args.distance_measure)
    distance_measure = "pyfunc" if args.distance_measure == "cosine" else args.distance_measure
    
    tree = BallTree(df_feat_for_dist, metric=distance_measure) if args.distance_measure != "cosine" else BallTree(df_feat_for_dist, metric=distance_measure, func=cosine_distance)
    n_points_in_ball = np.linspace(20, int(args.max_frac* len(df_feat_for_dist)), args.num_frac, dtype=int)
    fractions = n_points_in_ball / len(df_feat_for_dist)
    
    experiment_setting = explanation_handler.get_experiment_setting(fractions, args)
    explanation_handler.run_analysis(
                     tst_feat_for_expl, 
                     tst_feat_for_dist, 
                     df_feat_for_expl, 
                     df_feat_for_dist,
                     explanations, 
                     n_points_in_ball, 
                     predict_fn, 
                     tree,
                     results_path,
                     experiment_setting)

    print("Finished computing accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--gradient_method", type=str, default = "IG", help="Which Gradient Method to use: [IG, IG+SmoothGrad]")
    parser.add_argument("--results_folder", type=str, default = "/home/grotehans/xai_locality/results/gradient_methods/integrated_gradient", help="Path to the results folder")
    parser.add_argument("--model_type", type=str, default="binary_inception_v3", help="binary_inception_v3 or inception_v3")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--max_frac", type=float, default=0.1, help="Until when to compute the fraction of points in the ball")
    parser.add_argument("--num_frac", type=int, default=100, help="Number of fractions to compute")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--predict_threshold", type=float, default = -20, help="Threshold for classifying sample as top prediction")
    parser.add_argument("--max_test_points", type=int, default = 200)
    parser.add_argument("--chunk_size", type=int, default = 2)
                                    
    args = parser.parse_args()

    main(args)