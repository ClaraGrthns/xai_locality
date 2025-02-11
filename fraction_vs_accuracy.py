import os.path as osp
import numpy as np
import argparse
from sklearn.neighbors import BallTree
import os
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.misc import set_random_seeds, get_path
from src.model.factory import ModelHandlerFactory
from src.explanation_methods.factory import ExplanationMethodHandlerFactory
from src.config.handler import ConfigHandler

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

    model_handler = ModelHandlerFactory.get_handler(args.model_type)(args)
    model = model_handler.model
    dataset = model_handler.load_data()
    predict_fn = model_handler.predict_fn
    
    if args.method == "lime" and args.kernel_width is None:
        args.kernel_width = np.round(np.sqrt(dataset[4].shape[1]) * .75, 2)  # Default value

    method = args.method if args.method != "gradient" else args.gradient_method
    explainer_handler = ExplanationMethodHandlerFactory.get_handler(method=method)(args)
    explainer_handler.set_explainer(dataset=dataset,
                                    class_names=model_handler.get_class_names(),
                                    model=model)
    
    df_feat = model_handler.load_feature_vectors()
    tst_feat_for_dist, df_feat_for_dist, tst_feat_for_expl, df_feat_for_expl = explainer_handler.prepare_data_for_analysis(dataset,
                                                                                                            df_feat)
    
    explanations = explainer_handler.compute_explanations(results_path=results_path, 
                                                          predict_fn=predict_fn, 
                                                          tst_data=tst_feat_for_expl)
    
    validate_distance_measure(args.distance_measure)
    distance_measure = "pyfunc" if args.distance_measure == "cosine" else args.distance_measure
    
    tree = BallTree(df_feat_for_dist, metric=distance_measure) if args.distance_measure != "cosine" else BallTree(df_feat_for_dist, metric=distance_measure, func=cosine_distance)
    n_points_in_ball = np.linspace(20, int(args.max_frac * len(df_feat_for_dist)), args.num_frac, dtype=int)

    
    explainer_handler.run_analysis(
                     tst_feat_for_expl = tst_feat_for_expl, 
                     tst_feat_for_dist = tst_feat_for_dist, 
                     df_feat_for_expl = df_feat_for_expl, 
                     explanations = explanations, 
                     n_points_in_ball = n_points_in_ball, 
                     predict_fn = predict_fn, 
                     tree = tree,
                     results_path = results_path,
                     )
    print("Finished computing accuracy and fraction of points in the ball")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locality Analyzer")

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Data and model paths
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--results_folder", type=str,help="Path to the results folder" )
    parser.add_argument("--setting", type=str, help="Setting of the experiment")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--results_path", type=str,  help="Path to save results")
    
    # Model and method configuration
    parser.add_argument("--model_type", type=str, help="Model type: lightgbm, tab_inception_v3, pt_frame_lgm, pt_frame_xgboost, binary_inception_v3, inception_v3")
    parser.add_argument("--method", type=str,help="Explanation method to use (lime or gradient)")
    parser.add_argument("--gradient_method", type=str, help="Which Gradient Method to use: [IG, IG+SmoothGrad]")
    
    # Analysis parameters
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--max_frac", type=float, default=1.0, help="Until when to compute the fraction of points in the ball")
    parser.add_argument("--num_frac", type=int, default=10, help="Number of fractions to compute")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=2, help="Chunk size of test set computed at once")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # LIME-specific parameters
    parser.add_argument("--kernel_width", type=float, default=None, help="Kernel size for the locality analysis")
    parser.add_argument("--model_regressor", type=str, default="ridge", help="Model regressor for LIME")
    parser.add_argument("--num_lime_features", type=int, default=10, help="Number of features for LIME explanation")
    
    # Other parameters
    parser.add_argument("--predict_threshold", type=float, default=None, help="Threshold for classifying sample as top prediction")
    parser.add_argument("--max_test_points", type=int, default=200)
    
    args = parser.parse_args()
    config_handler = ConfigHandler(args.config)
    args = config_handler.update_args(args)


    # Validate arguments
    if args.method == "lime":
        assert (args.data_folder and args.setting and args.model_folder and args.results_folder) or (args.data_path and args.model_path and args.results_path), "You must provide either data_folder, model_folder, results_folder, and setting or data_path, model_path, and results_path."
    print("Starting the experiment with the following arguments: ", args)
    main(args)



