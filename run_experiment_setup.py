import argparse
import os
import os.path as osp
from pathlib import Path
import sys
import copy
import time

from src.train.custom_data_frame_benchmark import main_deep_models, main_gbdt
from fraction_vs_accuracy import main as main_fraction_vs_accuracy
from knn_on_model_preds import main as main_knn_on_model_preds
from src.utils.misc import set_random_seeds

CUSTOM_MODELS = ["LogisticRegression"]
GBT_MODELS = ["LightGBM", "XGBoost", "CatBoost"]
BASEDIR = str(Path(__file__).resolve().parent)
print(f"Base directory: {BASEDIR}")

def parse_args():
    parser = argparse.ArgumentParser(description="XAI Locality Experiment Suite")

    parser.add_argument("--config", type=str, 
                        help="Path to configuration file")

    # Basic configuration
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_folder", type=str,  
                        help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, default=BASEDIR + "/pretrained_models",
                        help="Path to save/load models")
    parser.add_argument("--results_folder", type=str, default=BASEDIR + "/results",
                        help="Path to save results")
    parser.add_argument("--setting", type=str, help="Setting name for the experiment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, 
                        choices=["LightGBM", "XGBoost", "ExcelFormer", "MLP", "TabNet", "Trompt", "FTTransformer", "ResNet", "LogisticRegression"],
                        help="Type of model to train/use")
    parser.add_argument("--skip_training", action="store_true", 
                        help="Skip model training if the model already exists")
    parser.add_argument("--force_training", action="store_true", 
                        help="Force training even if the model exists")
    
    # Benchmark dataset configuration
    parser.add_argument("--use_benchmark", action="store_true", 
                        help="Use benchmark dataset instead of synthetic data")
    parser.add_argument("--task_type", type=str, 
                        choices=["binary_classification", "multiclass_classification", "regression"],
                        help="Task type for benchmark dataset")
    parser.add_argument("--scale", type=str, 
                        choices=["small", "medium", "large"],
                        help="Scale of benchmark dataset")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of benchmark dataset")
    parser.add_argument("--num_trials", type=int,  help="Number of trials for training")
    parser.add_argument("--num_repeats", type=int, help="Number of repeats for training")

    # Train configuration
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--optimize', action='store_true', help='Use Optuna for hyperparameter optimization')
    
    # Synthetic data generation (if needed)
    parser.add_argument("--n_features", type=int, help="Number of features")
    parser.add_argument("--n_informative", type=int,  help="Number of informative features")
    parser.add_argument("--n_redundant", type=int,  help="Number of redundant features")
    parser.add_argument("--n_repeated", type=int,  help="Number of repeated features")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--n_clusters_per_class", type=int, default=2, help="Number of clusters per class")
    parser.add_argument("--class_sep", type=float,  help="Class separation")
    parser.add_argument("--flip_y", type=float, help="Fraction of samples with random labels")
    parser.add_argument("--hypercube", action="store_true", help="If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.")
    parser.add_argument("--test_size", type=float, default=0.4, help="Test size for train-test split")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size for train-validation split")
    
    # KNN analysis parameters
    parser.add_argument("--distance_measure", type=str, help="Distance measure for KNN")
    parser.add_argument("--distance_measures", nargs='+', default=["euclidean", "manhattan", "cosine"], 
                        help="List of distance measures to use")
    parser.add_argument("--min_k", type=int, default=1, help="Minimum k for KNN")
    parser.add_argument("--max_k", type=int, default=25, help="Maximum k for KNN")
    parser.add_argument("--k_step", type=int, default=1, help="Step size for k in KNN")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size for processing")
    parser.add_argument("--max_test_points", type=int, default=200, help="Maximum number of test points")
    parser.add_argument("--force_overwrite", action="store_true", help="Force overwrite existing results")
    
    # Explanation method parameters
    parser.add_argument("--method", type=str, choices=["lime", "gradient_methods"], 
                        help="Explanation method to use")
    parser.add_argument("--gradient_method", type=str,
                        choices=["IG", "IG+SmoothGrad"], 
                        help="Gradient-based explanation method")
    parser.add_argument("--kernel_width", type=str, default="default", 
                        choices=["default", "double", "half"], 
                        help="Kernel width for LIME")
    parser.add_argument("--model_regressor", type=str, default="ridge", help="Model regressor for LIME")
    parser.add_argument("--num_lime_features", type=int, default=10, 
                        help="Number of features to use in LIME explanation")
    parser.add_argument("--predict_threshold", type=float,)
    
    # Process steps control
    parser.add_argument("--skip_knn", action="store_true", help="Skip KNN analysis")
    parser.add_argument("--skip_fraction", action="store_true", help="Skip fraction vs accuracy analysis")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing results")
    
    return parser.parse_args()

def check_model_exists(args):
    """Check if the model file exists."""
    if args.use_benchmark:# For benchmark datasets
        if args.setting is None:
            from src.train.data_frame_benchmark import get_dataset_name
            dataset_name = get_dataset_name(args.task_type, args.scale, args.idx)
            model_path = osp.join(args.model_folder, args.model_type,
                                f"{args.model_type}_normalized_binary_{dataset_name}_results.pt")
            args.setting = dataset_name  # Use dataset name as setting
        else:
            from src.train.data_frame_benchmark import get_dataset_specs
            if not all([args.task_type, args.scale, args.idx]):
                args.task_type, args.scale, args.idx = get_dataset_specs(args.setting)
            model_path = osp.join(args.model_folder, args.model_type,
                                f"{args.model_type}_{args.setting}_results.pt")
    else:
        setting_name = (f"n_feat{args.n_features}_n_informative{args.n_informative}_"
                        f"n_redundant{args.n_redundant}_n_repeated{args.n_repeated}_"
                        f"n_classes{args.n_classes}_n_samples{args.n_samples}_"
                        f"n_clusters_per_class{args.n_clusters_per_class}_"
                        f"class_sep{args.class_sep}_flip_y{args.flip_y}_"
                        f"random_state{args.random_seed}")
        if not args.hypercube:
            setting_name += f'_hypercube{args.hypercube}'
        model_path = osp.join(args.model_folder, args.model_type, "synthetic_data", 
                             f"{args.model_type}_{setting_name}_results.pt")
        args.setting = setting_name
    
    return osp.exists(model_path), model_path

def get_data_path(args):
    """Get data path based on model type and dataset."""
    if args.use_benchmark:
        return osp.join(args.data_folder, f"{args.model_type}_{args.setting}_normalized_data.pt")
    else:
        # For synthetic data, check if it's ExcelFormer which has a special path format
        is_ExcelFormer_str = "ExcelFormer_" if args.model_type == 'ExcelFormer' else ""
        return osp.join(args.data_folder, "synthetic_data", 
                      f"{is_ExcelFormer_str}{args.setting}_normalized_tensor_frame.pt")

def get_results_path(args, step):
    """Get results path for a specific step (train, knn, fraction)."""
    base_path = "" if args.use_benchmark else "synthetic_data"

    if step == "knn":
        return osp.join(args.results_folder, "knn_model_preds", args.model_type, 
                      base_path, args.setting)
    elif step == "train":
        return osp.join(args.model_folder, "pretrained_models", args.model_type, "synthetic_data" if not args.use_benchmark else "",
                          f"{args.model_type}_{args.setting}_results.pt")
    elif step == "fraction":
        # For fraction analysis, the path depends on the explanation method
        method_subdir = args.gradient_method if args.method == "gradient_methods" else ""
        return osp.join(args.results_folder, args.method, 
                      method_subdir, args.model_type, base_path, args.setting)
    return None

def get_dataset_specific_settings(args):
    """Get dataset-specific settings like include_trn and include_val."""
    include_trn = False
    include_val = False
    
    if args.setting == "jannis":
        include_trn = True
        include_val = True
    elif args.setting == "MiniBooNE":
        include_val = True
    return include_trn, include_val

def train_model(args):
    """Train model based on model type and dataset choice."""
    print(f"Training {args.model_type} model...")
    train_args = copy.deepcopy(args)
    train_args.results_path = get_results_path(args, "train")
    train_args.results_folder = osp.join(args.model_folder, args.model_type)

    if args.use_benchmark:
        from src.train.data_frame_benchmark import main_deep_models as benchmark_deep_models
        from src.train.data_frame_benchmark import main_gbdt as benchmark_gbdt
        from src.train.train_pytorch_model import main as benchmark_custom_models
        if args.model_type in GBT_MODELS:
            benchmark_gbdt(train_args)
        elif args.model_type in CUSTOM_MODELS:
            benchmark_custom_models(train_args)
        else:
            benchmark_deep_models(train_args)
    else:
        train_args.data_folder = osp.join(train_args.data_folder, "synthetic_data")
        train_args.results_folder = osp.join(train_args.results_folder, "synthetic_data")
        # For synthetic data
        if args.model_type in GBT_MODELS:
            main_gbdt(train_args)
        elif args.model_type in CUSTOM_MODELS:
            from src.train.train_pytorch_model import main as custom_models
            custom_models(train_args)
        else:
            main_deep_models(train_args)

def run_knn_analysis(args):
    """Run KNN analysis on model predictions."""
    print("Running KNN analysis on model predictions...")
    knn_args = copy.deepcopy(args)
    knn_args.results_path = get_results_path(args, "knn")
    knn_args.data_folder = osp.join(args.data_folder, "synthetic_data") if not args.use_benchmark else args.data_folder

    main_knn_on_model_preds(knn_args)

def run_fraction_analysis(args):
    """Run fraction vs accuracy analysis."""
    print("Running fraction vs accuracy analysis...")
    fraction_args = copy.deepcopy(args)
    fraction_args.results_path = get_results_path(args, "fraction")
    if not hasattr(fraction_args, 'distance_measure') or not fraction_args.distance_measure:
        fraction_args.distance_measure = fraction_args.distance_measures[0] if fraction_args.distance_measures else "euclidean"
    main_fraction_vs_accuracy(fraction_args)

def main():
    # Parse arguments and set random seed
    args = parse_args()
    set_random_seeds(args.random_seed)
    args.seed = args.random_seed
    if args.model_folder is None:
        args.model_folder = os.path.join(BASEDIR, "pretrained_models")
    if args.data_folder is None:
        args.data_folder = os.path.join(BASEDIR, "data")
    if args.results_folder is None:
        args.results_folder = os.path.join(BASEDIR, "results")
    
    # args.model_type = "LightGBM"
    # args.setting = "n_feat55_n_informative30_n_redundant5_n_repeated5_n_classes2_n_samples100000_n_clusters_per_class10_class_sep0.5_flip_y0.1_random_state42_hypercubeFalse"
    # args.method = "lime"
    # args.distance_measure = "euclidean"
    # args.n_features = 55
    # args.n_informative = 30
    # args.n_redundant = 5
    # args.n_repeated = 5
    # args.n_classes = 2
    # args.n_samples = 100000
    # args.n_clusters_per_class = 10
    # args.class_sep = 0.5
    # args.flip_y = 0.1
    # args.random_seed = 42
    # args.kernel_width = "default"
    # args.num_lime_features = 10
    # args.num_trials = 25
    # args.num_repeats = 5
    # args.force = True
    # args.chunk_size = 200

    model_exists, model_path = check_model_exists(args)
    args.model_path = model_path
    args.data_path = get_data_path(args)
    include_trn, include_val = get_dataset_specific_settings(args)
    args.include_trn = include_trn
    args.include_val = include_val
    print(args)

    if not args.config:
        is_synthetic = "" if args.use_benchmark else "synthetic_data"
        args.config = f"{BASEDIR}/configs/{args.method}/{args.gradient_method or ''}/{args.model_type}/{is_synthetic}/{args.setting}/config.yaml"
    
    if (not model_exists or args.force_training) and not args.skip_training:
        print("Starting with model training...")
        start_time = time.time()
        train_model(args)
        print(f"Model training completed in {(time.time() - start_time)/60:.2f} minutes.")
    else:
        print(f"Model already exists at {args.model_path}")
    
    if not args.skip_knn:
        print("Starting KNN analysis...")
        start_time = time.time()
        run_knn_analysis(args)
        print(f"KNN analysis completed in {(time.time() - start_time)/60:.2f} minutes.")
    
    if not args.skip_fraction:
        print("Starting fraction vs accuracy analysis...")
        start_time = time.time()
        run_fraction_analysis(args)
        print(f"Fraction vs accuracy analysis completed in {(time.time() - start_time):.2f} seconds.")
    print("Experiment complete!")

if __name__ == "__main__":
    main()