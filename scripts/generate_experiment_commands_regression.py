import os
import argparse
from pathlib import Path
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

def get_setting_name_regression(regression_mode,
                                n_features,
                                n_informative, 
                                n_samples, 
                                noise,
                                bias,
                                tail_strength,
                                coef,
                                effective_rank,
                                random_seed):
    setting_name = (f'regression_{regression_mode}_n_feat{n_features}_n_informative{n_informative}_n_samples{n_samples}_'
                    f'noise{noise}_bias{bias}_random_state{random_seed}')
    
    # Add additional parameters if they differ from defaults
    if effective_rank is not None:
        setting_name += f'_effective_rank{effective_rank}_tail_strength{tail_strength}'
    if coef:
        setting_name += '_coefTrue'
    return setting_name

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Python commands for running experiments")
    parser.add_argument('--output_dir', type=str, default='experiment_commands_regression',
                        help='Directory to save the generated command files')
    parser.add_argument('--base_dir', type=str, default=str(Path(__file__).resolve().parent.parent),
                        help='Base directory for the experiment')
    parser.add_argument('--skip_training', action='store_true',
                        help='Add --skip_training flag to commands')
    parser.add_argument('--force_training', action='store_true',
                        help='Add --force_training flag to commands')
    parser.add_argument('--skip_knn', action='store_true',
                        help='Add --skip_knn flag to commands')
    parser.add_argument('--skip_fraction', action='store_true',
                        help='Add --skip_fraction flag to commands')
    return parser.parse_args()

def create_command_file(output_dir, model, setting, method, distance_measure, kernel_width, num_lime_features,
                        is_synthetic, skip_training, force_training, skip_knn, skip_fraction, gradient_method=None,
                        synthetic_params=None):
    """Create a file containing the Python command for a specific configuration"""
    
    # Create directory structure - organize by method first
    if method == "gradient_methods" and gradient_method:
        if gradient_method == "IG":
            method_dir = os.path.join(output_dir, method, "integrated_gradient")
        else:
            method_dir = os.path.join(output_dir, method, "smooth_grad")
    else:
        method_dir = os.path.join(output_dir, method)
    
    # Then by model
    model_dir = os.path.join(method_dir, model)
    
    # Then by dataset
    if is_synthetic:
        dataset_dir = os.path.join(model_dir, "regression_synthetic_data" ,  setting)
    else:
        dataset_dir = os.path.join(model_dir, setting)
    
    # Finally by distance measure
    model_dir = os.path.join(dataset_dir, distance_measure)
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Format args for the command
    base_args = f"--model_type {model} --setting {setting} --method {method} --distance_measure {distance_measure} --regression --force"
    
    if is_synthetic:
        # Use the synthetic parameters directly instead of parsing them from the setting name
        if synthetic_params is not None:
            # Extract all parameters with proper defaults
            regression_mode = synthetic_params.get('regression_mode', True)
            n_features = synthetic_params.get('n_features', 100)
            n_informative = synthetic_params.get('n_informative', 10)
            n_samples = synthetic_params.get('n_samples', 100000)
            noise = synthetic_params.get('noise', 0.0)
            bias = synthetic_params.get('bias', 0.0)
            random_seed = synthetic_params.get('random_seed', 42)
            data_folder = synthetic_params.get('data_folder', 'data')
            test_size = synthetic_params.get('test_size', 0.4)
            val_size = synthetic_params.get('val_size', 0.1)
            tail_strength = synthetic_params.get('tail_strength', 0.5)
            coef = synthetic_params.get('coef', False)
            effective_rank = synthetic_params.get('effective_rank', None)
            
            # Add synthetic data parameters
            synthetic_args = (f" --regression_mode {regression_mode}"
                             f" --n_features {n_features}"
                             f" --n_informative {n_informative}"
                             f" --n_samples {n_samples}"
                             f" --noise {noise}"
                             f" --bias {bias}"
                             f" --random_seed {random_seed}"
                             f" --data_folder {data_folder}"
                             f" --test_size {test_size}"
                             f" --val_size {val_size}"
                             f" --tail_strength {tail_strength}"
                             )
            if coef:
                synthetic_args += " --coef"
            if effective_rank is not None:
                synthetic_args += f" --effective_rank {effective_rank}"
            base_args += synthetic_args
    else:
        # For benchmark datasets
        if setting == "airlines_DepDelay_1M":
            base_args += " --use_benchmark --task_type regression --scale large --idx 0 --num_trials 5 --num_repeats 1 --epochs 25"
        elif setting == "delays_zurich_transport":
            base_args += " --use_benchmark --task_type regression --scale large --idx 1 --num_trials 5 --num_repeats 1 --epochs 25"
        elif setting == "nyc-taxi-green-dec-2016":
            base_args += " --use_benchmark --task_type regression --scale large --idx 2 --num_trials 5 --num_repeats 1 --epochs 25"
        elif setting == "microsoft":
            base_args += " --use_benchmark --task_type regression --scale large --idx 3 --num_trials 5 --num_repeats 1 --epochs 25"
        elif setting == "yahoo":
            base_args += " --use_benchmark --task_type regression --scale large --idx 4 --num_trials 5 --num_repeats 1 --epochs 25"
        elif setting == "year":
            base_args += " --use_benchmark --task_type regression --scale large --idx 5 --num_trials 5 --num_repeats 1 --epochs 25"
    
    if method == "lime":
        base_args += f" --kernel_width {kernel_width} --num_lime_features {num_lime_features}"
    elif method == "gradient_methods" and gradient_method:
        # For IG, make sure to use the correct command line name
        base_args += f" --gradient_method {gradient_method}"
    
    # Add optional flags
    if skip_training:
        base_args += " --skip_training"
    if force_training:
        base_args += " --force_training"
    if skip_knn:
        base_args += " --skip_knn"
    if skip_fraction:
        base_args += " --skip_fraction"
    
    # Create the full command
    command = f"/home/grotehans/miniconda3/envs/tab_models/bin/python -u run_experiment_setup.py {base_args}"
    
    # Define filename - include distance measure to distinguish files
    distance_suffix = f"_{distance_measure}"
    
    if method == "lime":
        filename = f"lime_{kernel_width}{distance_suffix}.sh"
    elif method == "gradient_methods" and gradient_method:
        if gradient_method == "IG":
            filename = f"gradient_integrated_gradient{distance_suffix}.sh"
        else:
            filename = f"gradient_{gradient_method}{distance_suffix}.sh"
    else:
        filename = f"{method}{distance_suffix}.sh"
    
    # Write command to file
    file_path = os.path.join(model_dir, filename)
    with open(file_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Model: {model}, Dataset: {setting}, Method: {method}{' '+gradient_method if gradient_method else ''}\n")
        f.write(f"# Distance: {distance_measure}{', Kernel: '+kernel_width if kernel_width else ''}\n\n")
        f.write(f"{command}\n")
    
    # Make file executable
    os.chmod(file_path, 0o755)
    
    print(f"Created {file_path}")
    return file_path
def main():
    args = parse_arguments()
    output_dir = os.path.join(args.base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define synthetic configurations with explicit parameters
    synthetic_configs = [
        {
            'regression_mode': "linear",
            'n_features': 100,
            'n_informative': 10,
            'n_samples': 100000,
            'bias': 1.0,
            'noise': 0.0,
            'random_seed': 42,
            'tail_strength': 0.5,
            'effective_rank': None, 
            'coef': True
        },
        
    ]
    
    # Generate setting names from configurations
    synthetic_settings = []
    for config in synthetic_configs:
        setting = get_setting_name_regression(regression_mode=config['regression_mode'],
                                              n_features=config['n_features'],
                                              n_informative=config['n_informative'],
                                              n_samples=config['n_samples'],
                                              noise=config['noise'],
                                              bias=config['bias'],
                                              tail_strength=config['tail_strength'],
                                              effective_rank=config['effective_rank'],
                                              coef=config['coef'],
                                              random_seed=config['random_seed'])
        synthetic_settings.append((setting, config))
    
    models = ["LightGBM", "MLP", "LinReg",  "TabNet", "FTTransformer", "ResNet", "TabTransformer"]
    standard_settings = ["airlines_DepDelay_1M", "delays_zurich_transport", "nyc-taxi-green-dec-2016", "microsoft", "yahoo", "year"]
    methods = ["lime", "gradient_methods"]
    distance_measures = ["euclidean"]
    
    # Generate all command files
    created_files = []
    
    # First, handle standard benchmark datasets
    for model in models:
        for setting in standard_settings:
            for method in methods:
                if method == "gradient_methods":
                    if model =="LightGBM":
                        continue
                    gradient_method = "IG"  # Integrated Gradient
                    
                    for distance_measure in distance_measures:
                        file = create_command_file(
                            output_dir=output_dir,
                            model=model,
                            setting=setting,
                            method=method,
                            distance_measure=distance_measure,
                            kernel_width=None,
                            num_lime_features=10,
                            is_synthetic=False,
                            skip_training=args.skip_training,
                            force_training=args.force_training,
                            skip_knn=args.skip_knn,
                            skip_fraction=args.skip_fraction,
                            gradient_method=gradient_method
                        )
                        created_files.append(file)
                else:  # lime
                    kernel_width = "default"
                    
                    for distance_measure in distance_measures:
                        file = create_command_file(
                            output_dir=output_dir,
                            model=model,
                            setting=setting,
                            method=method,
                            distance_measure=distance_measure,
                            kernel_width=kernel_width,
                            num_lime_features=10,
                            is_synthetic=False,
                            skip_training=args.skip_training,
                            force_training=args.force_training,
                            skip_knn=args.skip_knn,
                            skip_fraction=args.skip_fraction
                        )
                        created_files.append(file)
    
    # Then, handle synthetic datasets
    for model in models:
        for setting_info in synthetic_settings:
            setting, config = setting_info
            
            for method in methods:
                if method == "gradient_methods":
                    gradient_method = "IG"  # Integrated Gradient
                    
                    for distance_measure in distance_measures:
                        file = create_command_file(
                            output_dir=output_dir,
                            model=model,
                            setting=setting,
                            method=method,
                            distance_measure=distance_measure,
                            kernel_width=None,
                            num_lime_features=10,
                            is_synthetic=True,
                            skip_training=args.skip_training,
                            force_training=args.force_training,
                            skip_knn=args.skip_knn,
                            skip_fraction=args.skip_fraction,
                            gradient_method=gradient_method,
                            synthetic_params=config
                        )
                        created_files.append(file)
                else:  # lime
                    kernel_width = "default"
                    
                    for distance_measure in distance_measures:
                        file = create_command_file(
                            output_dir=output_dir,
                            model=model,
                            setting=setting,
                            method=method,
                            distance_measure=distance_measure,
                            kernel_width=kernel_width,
                            num_lime_features=10,
                            is_synthetic=True,
                            skip_training=args.skip_training,
                            force_training=args.force_training,
                            skip_knn=args.skip_knn,
                            skip_fraction=args.skip_fraction,
                            synthetic_params=config
                        )
                        created_files.append(file)
    
    # Create method-specific run_all.sh files
    for method in methods:
        method_dir = os.path.join(output_dir, method)
        
        # Special handling for gradient_methods with subdirectories
        if method == "gradient_methods":
            gradient_dirs = ["integrated_gradients"]
            for gradient_dir in gradient_dirs:
                gradient_method_dir = os.path.join(method_dir, gradient_dir)
                gradient_files = [f for f in created_files if f"{method}/{gradient_dir}/" in f]
                
                if gradient_files:
                    gradient_run_all = os.path.join(gradient_method_dir, "run_all.sh")
                    with open(gradient_run_all, "w") as f:
                        f.write("#!/bin/bash\n\n")
                        f.write(f"# Run all experiments for {method}/{gradient_dir}\n\n")
                        for file in gradient_files:
                            f.write(f"{file}\n")
                    
                    os.chmod(gradient_run_all, 0o755)
                    print(f"Created method runner: {gradient_run_all}")
        else:
            method_files = [f for f in created_files if f"/{method}/" in f]
            if method_files:
                method_run_all = os.path.join(method_dir, "run_all.sh")
                with open(method_run_all, "w") as f:
                    f.write("#!/bin/bash\n\n")
                    f.write(f"# Run all experiments for method: {method}\n\n")
                    for file in method_files:
                        f.write(f"{file}\n")
                
                os.chmod(method_run_all, 0o755)
                print(f"Created method runner: {method_run_all}")
    
    # Create model-specific run_all.sh files within each method
    for method in methods:
        if method == "gradient_methods":
            gradient_dirs = ["integrated_gradients"]
            for gradient_dir in gradient_dirs:
                for model in models:
                    model_dir = os.path.join(output_dir, method, gradient_dir, model)
                    model_files = [f for f in created_files if f"{method}/{gradient_dir}/{model}/" in f]
                    
                    if not model_files:
                        continue
                        
                    model_run_all = os.path.join(model_dir, "run_all.sh")
                    with open(model_run_all, "w") as f:
                        f.write("#!/bin/bash\n\n")
                        f.write(f"# Run all experiments for {method}/{gradient_dir}/{model}\n\n")
                        for file in model_files:
                            f.write(f"{file}\n")
                    
                    os.chmod(model_run_all, 0o755)
                    print(f"Created model runner: {model_run_all}")
        else:
            for model in models:
                model_dir = os.path.join(output_dir, method, model)
                model_files = [f for f in created_files if f"{method}/{model}/" in f]
                
                if not model_files:
                    continue
                    
                model_run_all = os.path.join(model_dir, "run_all.sh")
                with open(model_run_all, "w") as f:
                    f.write("#!/bin/bash\n\n")
                    f.write(f"# Run all experiments for {method}/{model}\n\n")
                    for file in model_files:
                        f.write(f"{file}\n")
                
                os.chmod(model_run_all, 0o755)
                print(f"Created model runner: {model_run_all}")
    
    # Create dataset-specific run_all.sh files within each method/model
    for method in methods:
        if method == "gradient_methods":
            gradient_dirs = ["integrated_gradients"]
            for gradient_dir in gradient_dirs:
                for model in models:
                    # Standard datasets
                    for dataset in standard_settings:
                        dataset_dir = os.path.join(output_dir, method, gradient_dir, model, dataset)
                        dataset_files = [f for f in created_files if f"{method}/{gradient_dir}/{model}/{dataset}/" in f]
                        
                        if dataset_files:
                            dataset_run_all = os.path.join(dataset_dir, "run_all.sh")
                            with open(dataset_run_all, "w") as f:
                                f.write("#!/bin/bash\n\n")
                                f.write(f"# Run all experiments for {method}/{gradient_dir}/{model}/{dataset}\n\n")
                                for file in dataset_files:
                                    f.write(f"{file}\n")
                            
                            os.chmod(dataset_run_all, 0o755)
                            print(f"Created dataset runner: {dataset_run_all}")
                    
                    # Synthetic datasets
                    for dataset, _ in synthetic_settings:
                        dataset_dir = os.path.join(output_dir, method, gradient_dir, model, "regression_synthetic_data",  dataset)
                        dataset_files = [f for f in created_files if f"{method}/{gradient_dir}/{model}/regression_synthetic_data/{dataset}/" in f]
                        
                        if dataset_files:
                            dataset_run_all = os.path.join(dataset_dir, "run_all.sh")
                            with open(dataset_run_all, "w") as f:
                                f.write("#!/bin/bash\n\n")
                                f.write(f"# Run all experiments for {method}/{gradient_dir}/{model}/regression_synthetic_data/{dataset}\n\n")
                                for file in dataset_files:
                                    f.write(f"{file}\n")
                            
                            os.chmod(dataset_run_all, 0o755)
                            print(f"Created dataset runner: {dataset_run_all}")
        else:
            for model in models:
                # Standard datasets
                for dataset in standard_settings:
                    dataset_dir = os.path.join(output_dir, method, model, dataset)
                    dataset_files = [f for f in created_files if f"{method}/{model}/{dataset}/" in f]
                    
                    if dataset_files:
                        dataset_run_all = os.path.join(dataset_dir, "run_all.sh")
                        with open(dataset_run_all, "w") as f:
                            f.write("#!/bin/bash\n\n")
                            f.write(f"# Run all experiments for {method}/{model}/{dataset}\n\n")
                            for file in dataset_files:
                                f.write(f"{file}\n")
                        
                        os.chmod(dataset_run_all, 0o755)
                        print(f"Created dataset runner: {dataset_run_all}")
                
                # Synthetic datasets
                for dataset, _ in synthetic_settings:
                    dataset_dir = os.path.join(output_dir, method, model, "regression_synthetic_data",  dataset)
                    dataset_files = [f for f in created_files if f"{method}/{model}/regression_synthetic_data/{dataset}/" in f]
                    
                    if dataset_files:
                        dataset_run_all = os.path.join(dataset_dir, "run_all.sh")
                        with open(dataset_run_all, "w") as f:
                            f.write("#!/bin/bash\n\n")
                            f.write(f"# Run all experiments for {method}/{model}/regression_synthetic_data/{dataset}\n\n")
                            for file in dataset_files:
                                f.write(f"{file}\n")
                        
                        os.chmod(dataset_run_all, 0o755)
                        print(f"Created dataset runner: {dataset_run_all}")
                        
    # Create a master run_all.sh file
    run_all_path = os.path.join(output_dir, "run_all.sh")
    with open(run_all_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# This script runs all generated experiment commands\n\n")
        
        # Run each method's run_all.sh
        for method in methods:
            if method == "gradient_methods":
                for gradient_dir in ["integrated_gradients"]:
                    method_run_all = os.path.join(output_dir, method, gradient_dir, "run_all.sh")
                    if os.path.exists(method_run_all):
                        f.write(f"echo 'Running experiments for {method}/{gradient_dir}...'\n")
                        f.write(f"{method_run_all}\n\n")
            else:
                method_run_all = os.path.join(output_dir, method, "run_all.sh")
                if os.path.exists(method_run_all):
                    f.write(f"echo 'Running experiments for {method}...'\n")
                    f.write(f"{method_run_all}\n\n")
    
    os.chmod(run_all_path, 0o755)
    print(f"\nCreated master runner: {run_all_path}")
    
    print(f"\nCreated {len(created_files)} command files in {output_dir}")
    print("\nTo run all commands, you can use:")
    print(f"{run_all_path}")
    print("\nOr run experiments for a specific method:")
    print(f"<method>/run_all.sh")
    print("\nOr for a specific model within a method:")
    print(f"<method>/<model>/run_all.sh")
if __name__ == "__main__":
    main()
