import os.path as osp
import numpy as np
import argparse
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
from src.utils.misc import  set_random_seeds, get_path
from src.explanation_methods.lime_analysis.lime_local_classifier import compute_lime_accuracy_per_fraction
import os
from src.explanation_methods.factory import ExplanationMethodHandlerFactory
from src.model.factory import ModelHandlerFactory
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

    data_path = get_path(args.data_folder, args.data_path, args.setting)
    model_path = get_path(args.model_folder, args.model_path, args.setting, suffix="final_model_")
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    if args.num_test_splits > 1:
        results_path = osp.join(results_path, f"test_splits")
    
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)

    # Add .npz extension specifically for data_path
    if args.data_folder is not None:
        data_path += ".npz"

    model_handler = ModelHandlerFactory.get_handler(args.model_type, model_path)
    dataset = model_handler.load_data(data_path)
    trn_feat = dataset[4] if type(dataset) == tuple else dataset
    
    predict_fn = model_handler.predict_fn
    
    class_names = model_handler.get_class_names()
    explanation_handler = ExplanationMethodHandlerFactory.get_handler(method="LIME",
                                                        trn_feat=trn_feat,
                                                       feature_names=np.arange(trn_feat.shape[1]),
                                                       class_names=class_names,
                                                       discretize_continuous=True,
                                                       random_state=args.random_seed,
                                                       kernel_width=args.kernel_width)
    explainer = explanation_handler.explainer

    tst_feat_for_dist, df_feat_for_dist, tst_feat_for_expl, df_feat_for_expl = explanation_handler.process_data(dataset, model_handler, args)


    validate_distance_measure(args.distance_measure)
    distance_measure = "pyfunc" if args.distance_measure == "cosine" else args.distance_measure
    
    tree = BallTree(df_feat_for_dist, metric=distance_measure) if args.distance_measure != "cosine" else BallTree(df_feat_for_dist, metric=distance_measure, func=cosine_distance)
    n_points_in_ball = np.linspace(20, int(args.max_frac* len(df_feat_for_dist)), args.num_frac, dtype=int)
    
    fractions = n_points_in_ball / len(df_feat_for_dist)
    
    if args.kernel_width is None:
        args.kernel_width = np.round(np.sqrt(trn_feat.shape[1]) * .75, 2)  # Default value

    explanations = explanation_handler.compute_explanations(results_path=results_path, 
                                                            explainer=explainer, 
                                                            predict_fn=predict_fn, 
                                                            tst_data=tst_feat_for_dist, 
                                                            device=None, 
                                                            args=args)
    experiment_setting = explanation_handler.get_experiment_setting(fractions, args)
    
    
    num_samples = len(tst_feat_for_dist)
    num_fractions = len(fractions)
    results = {
        "accuracy": np.zeros((num_fractions, num_samples)),
        "radius": np.zeros((num_fractions, num_samples)),
        "fraction_points_in_ball": fractions,
        "ratio_all_ones": np.zeros((num_fractions, num_samples)),
    }
    chunk_size = args.chunk_size
    predict_threshold = args.predict_threshold
    for i in range(0, len(tst_feat_for_dist), chunk_size):
        if args.debug:
            # Normal for loop for easier debugging
            chunk_end = min(i + chunk_size, len(tst_feat_for_dist))
            explanations_chunk = explanations[i:chunk_end]
            for n_closest in n_points_in_ball:
                print(f"Processing fraction {n_closest}")
                chunk_results = compute_lime_accuracy_per_fraction(
                    tst_feat_for_dist[i:i+chunk_size], df_feat_for_dist, explanations_chunk, explainer, predict_fn, n_closest, tree, predict_threshold
                )
                fraction_idx = np.where(n_points_in_ball == n_closest)[0][0]
                results["accuracy"][fraction_idx, i:i+chunk_size] = chunk_results[1]
                results["radius"][fraction_idx, i:i+chunk_size] = chunk_results[-1]
            np.savez(osp.join(results_path, experiment_setting), **results)
        else:
            print(f"Processing chunk {i//chunk_size + 1}/{(len(tst_feat_for_dist) + chunk_size - 1)//chunk_size}")
            # Parallel processing for normal execution
            chunk_end = min(i + chunk_size, len(tst_feat_for_dist))
            tst_chunk = tst_feat_for_dist[i:chunk_end]
            explanations_chunk = explanations[i:chunk_end]
            chunk_results = Parallel(n_jobs=-1)(
                    delayed(compute_lime_accuracy_per_fraction)(
                        tst_chunk, df_feat_for_dist, explanations_chunk, explainer, predict_fn, n_closest, tree, predict_threshold
                    )
                    for n_closest in n_points_in_ball
                )
            # Unpack results directly into the correct positions in the arrays
            for n_closest, acc, ratio, R in chunk_results:
                fraction_idx = np.where(n_points_in_ball == n_closest)[0][0]
                results["accuracy"][fraction_idx, i:chunk_end] = acc
                results["ratio_all_ones"][fraction_idx, i:chunk_end] = ratio
                results["radius"][fraction_idx, i:chunk_end] = R

        np.savez(osp.join(results_path, experiment_setting), **results)
        print(f"Processed chunk {i//chunk_size + 1}/{(len(tst_feat_for_dist) + chunk_size - 1)//chunk_size}")

    print("Finished computing LIME accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")#, default = "/home/grotehans/xai_locality/data/synthetic_data")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")#, default = "/home/grotehans/xai_locality/pretrained_models/lightgbm/synthetic_data" )
    parser.add_argument("--results_folder", type=str, help="Path to the results folder")#, default="/home/grotehans/xai_locality/results/lightgbm/synthetic_data")
    parser.add_argument("--setting", type=str, help="Setting of the experiment")#, default= "n_feat50_n_informative20_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class5_class_sep0.9_flip_y0.01_random_state42")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--model_type", type=str, default="lightgbm", help="Model type, so far supported: lightgbm, tab_inception_v3, pt_frame_lgm, pt_frame_xgboost")
    parser.add_argument("--results_path", type=str, help="Path to save results")#, default=" ")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--max_frac", type=float, default=1.0, help="Until when to compute the fraction of points in the ball")
    parser.add_argument("--num_frac", type=int, default=10, help="Number of fractions to compute")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=20, help="Chunk size of test set computed at once")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--kernel_width", type=float, default=None, help="Kernel size for the locality analysis")
    parser.add_argument("--model_regressor", type=str, default="ridge", help="Model regressor for LIME")
    parser.add_argument("--num_test_splits",  type=int, default = 0, help="Number of test splits for analysis")
    parser.add_argument("--split_idx", type=int, default = 0, help="Index of the test split")
    parser.add_argument("--num_lime_features", type=int, default = 10, help="Index of the test split")
    parser.add_argument("--predict_threshold", type=float, default = None, help="Threshold for classifying sample as top prediction")
    parser.add_argument("--max_test_points", type=int, default = 200)

                                    
    args = parser.parse_args()

    # Validate arguments
    if (args.data_folder and args.setting and args.model_folder and args.results_folder) or (args.data_path and args.model_path and args.results_path):
        print("Starting the experiment with the following arguments: ", args)
        main(args)
    else:
        parser.error("You must provide either data_folder, model_folder, results_folder, and setting or data_path, model_path, and results_path.")