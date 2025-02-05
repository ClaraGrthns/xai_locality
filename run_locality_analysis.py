import argparse
import os
from src import gradient_methods_tabular_fraction_vs_accuracy 
from src import lime_tabular_fraction_vs_accuracy

def get_common_parser():
    """Create parser with arguments common to all analysis methods."""
    parser = argparse.ArgumentParser(description="Locality Analysis Runner")
    
    # Analysis method selection
    parser.add_argument("--method", type=str, required=True, choices=["lime", "gradient"],
                       help="Which explanation method to use: lime or gradient")
    
    # Common arguments between both methods
    parser.add_argument("--model_type", type=str, required=True,
                       help="Model type to analyze")
    parser.add_argument("--results_folder", type=str,
                       help="Path to save results")
    parser.add_argument("--distance_measure", type=str, default="euclidean",
                       help="Distance measure to use")
    parser.add_argument("--max_frac", type=float, default=0.1,
                       help="Maximum fraction of points to analyze")
    parser.add_argument("--num_frac", type=int, default=50,
                       help="Number of fractions to compute")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--predict_threshold", type=float, default=None,
                       help="Threshold for classifying sample as top prediction")
    parser.add_argument("--max_test_points", type=int, default=200,
                       help="Maximum number of test points to analyze")
    parser.add_argument("--chunk_size", type=int, default=20,
                       help="Chunk size for batch processing")
    
    return parser

def add_lime_specific_args(parser):
    """Add LIME-specific arguments to parser."""
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--setting", type=str, help="Setting of the experiment")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--results_path", type=str, help="Path to save results")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--kernel_width", type=float, default=None,
                       help="Kernel size for the locality analysis")
    parser.add_argument("--model_regressor", type=str, default="ridge",
                       help="Model regressor for LIME")
    parser.add_argument("--num_test_splits", type=int, default=0,
                       help="Number of test splits for analysis")
    parser.add_argument("--split_idx", type=int, default=0,
                       help="Index of the test split")
    parser.add_argument("--num_lime_features", type=int, default=10,
                       help="Number of features for LIME explanation")

def add_gradient_specific_args(parser):
    """Add gradient method-specific arguments to parser."""
    parser.add_argument("--gradient_method", type=str, default="IG",
                       choices=["IG", "IG+SmoothGrad"],
                       help="Which Gradient Method to use")

def validate_lime_args(args):
    """Validate LIME-specific arguments."""
    if not ((args.data_folder and args.setting and args.model_folder and args.results_folder) or 
            (args.data_path and args.model_path and args.results_path)):
        raise ValueError("For LIME analysis: You must provide either:\n"
                       "1. data_folder, model_folder, results_folder, and setting, OR\n"
                       "2. data_path, model_path, and results_path")

def main():
    # Get base parser with common arguments
    parser = get_common_parser()
    
    # Parse known arguments first (to check the method)
    known_args, remaining_args = parser.parse_known_args()

    # Add method-specific arguments **after** determining the method
    if known_args.method == "lime":
        add_lime_specific_args(parser)
    elif known_args.method == "gradient":
        add_gradient_specific_args(parser)

    args = parser.parse_args()

    if args.method == "lime":
        validate_lime_args(args)
        lime_tabular_fraction_vs_accuracy.main(args)
    elif args.method == "gradient":
        gradient_methods_tabular_fraction_vs_accuracy.main(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()