import os
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Python commands for running experiments")
    parser.add_argument('--output_dir', type=str, default='experiment_commands',
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
            method_dir = os.path.join(output_dir, method, "integrated_gradients")
        else:
            method_dir = os.path.join(output_dir, method, "smooth_grad")
    else:
        method_dir = os.path.join(output_dir, method)
    
    # Then by model
    model_dir = os.path.join(method_dir, model)
    
    # Then by dataset
    if is_synthetic:
        dataset_dir = os.path.join(model_dir, "synthetic_data", setting)
    else:
        dataset_dir = os.path.join(model_dir, setting)
    
    # Finally by distance measure
    model_dir = os.path.join(dataset_dir, distance_measure)
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Format args for the command
    base_args = f"--model_type {model} --setting {setting} --method {method} --distance_measure {distance_measure}"
    
    if is_synthetic:
        # Use the synthetic parameters directly instead of parsing them from the setting name
        if synthetic_params is not None:
            # Extract all parameters with proper defaults
            n_features = synthetic_params.get('n_features', 50)
            n_informative = synthetic_params.get('n_informative', 10)
            n_redundant = synthetic_params.get('n_redundant', 30)
            n_repeated = synthetic_params.get('n_repeated', 0)
            n_classes = synthetic_params.get('n_classes', 2)
            n_samples = synthetic_params.get('n_samples', 100000)
            n_clusters_per_class = synthetic_params.get('n_clusters_per_class', 3)
            class_sep = synthetic_params.get('class_sep', 0.9)
            flip_y = synthetic_params.get('flip_y', 0.1)
            random_seed = synthetic_params.get('random_seed', 42)
            hypercube = synthetic_params.get('hypercube', False)
            
            # Add synthetic data parameters
            synthetic_args = (f" --n_features {n_features}"
                             f" --n_informative {n_informative}"
                             f" --n_redundant {n_redundant}"
                             f" --n_repeated {n_repeated}"
                             f" --n_classes {n_classes}"
                             f" --n_samples {n_samples}"
                             f" --n_clusters_per_class {n_clusters_per_class}"
                             f" --class_sep {class_sep}"
                             f" --flip_y {flip_y}"
                             f" --random_seed {random_seed}"
                             f" --num_trials 25"
                             f" --num_repeats 5")
            
            # Add hypercube flag ONLY if it's True
            if hypercube:
                synthetic_args += " --hypercube"
                
            base_args += synthetic_args
        else:
            # If no synthetic_params provided, parse from the setting name
            params = {}
            for param in setting.split('_'):
                if param.startswith('n_feat'):
                    params['n_features'] = param[6:]
                elif param.startswith('n_informative'):
                    params['n_informative'] = param[13:]
                elif param.startswith('n_redundant'):
                    params['n_redundant'] = param[11:]
                elif param.startswith('n_repeated'):
                    params['n_repeated'] = param[10:]
                elif param.startswith('n_classes'):
                    params['n_classes'] = param[9:]
                elif param.startswith('n_samples'):
                    params['n_samples'] = param[9:]
                elif param.startswith('n_clusters_per_class'):
                    params['n_clusters_per_class'] = param[20:]
                elif param.startswith('class_sep'):
                    params['class_sep'] = param[9:]
                elif param.startswith('flip_y'):
                    params['flip_y'] = param[6:]
                elif param.startswith('random_state'):
                    params['random_seed'] = param[12:]
                elif param.startswith('hypercube'):
                    params['hypercube'] = param[9:]
            
            # Add synthetic data parameters from parsed setting
            synthetic_args = (f" --n_features {params.get('n_features', 50)}"
                             f" --n_informative {params.get('n_informative', 10)}"
                             f" --n_redundant {params.get('n_redundant', 30)}"
                             f" --n_repeated {params.get('n_repeated', 0)}"
                             f" --n_classes {params.get('n_classes', 2)}"
                             f" --n_samples {params.get('n_samples', 100000)}"
                             f" --n_clusters_per_class {params.get('n_clusters_per_class', 3)}"
                             f" --class_sep {params.get('class_sep', 0.9)}"
                             f" --flip_y {params.get('flip_y', 0.1)}"
                             f" --random_seed {params.get('random_seed', 42)}"
                             f" --num_trials 25"
                             f" --num_repeats 5")
            
            if params.get('hypercube', "False").lower() == "true":
                synthetic_args += " --hypercube"
            
            base_args += synthetic_args
    else:
        # For benchmark datasets
        if setting == "jannis":
            base_args += " --include_trn --include_val  --scale medium --idx 6  --num_trials 5 --num_repeats 5 --epochs 25"
        elif setting == "MiniBooNE":
            base_args += " --include_val --scale medium --idx 3 --num_trials 5 --num_repeats 5 --epochs 25"
        elif setting == "higgs":
            base_args += " --use_benchmark --task_type binary_classification --scale large --idx 0 --num_trials 5 --num_repeats 5 --epochs 10"
        base_args += " --use_benchmark --task_type binary_classification"
    
    # Method-specific parameters
    if method == "lime":
        base_args += f" --kernel_width {kernel_width} --num_lime_features {num_lime_features}"
    elif method == "gradient_methods" and gradient_method:
        # For IG, make sure to use the correct command line name
        cmd_gradient_method = "integrated_gradient" if gradient_method == "IG" else gradient_method
        base_args += f" --gradient_method {cmd_gradient_method}"
    
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
    command = f"python run_experiment_setup.py {base_args}"
    
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
            'n_features': 50, 
            'n_informative': 2, 
            'n_redundant': 30, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 2, 
            'class_sep': 0.9, 
            'flip_y': 0.1, 
            'random_seed': 42,
            'hypercube': True
        },
        {
            'n_features': 50, 
            'n_informative': 10, 
            'n_redundant': 30, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 3, 
            'class_sep': 0.9, 
            'flip_y': 0.1, 
            'random_seed': 42,
            'hypercube': True
        },
        {
            'n_features': 100, 
            'n_informative': 50, 
            'n_redundant': 30, 
            'n_repeated': 0, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 3, 
            'class_sep': 0.9, 
            'flip_y': 0.1, 
            'random_seed': 42,
            'hypercube': True
        },
        {
            'n_features': 55, 
            'n_informative': 30, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 10, 
            'class_sep': 0.5, 
            'flip_y': 0.1, 
            'random_seed': 42,
            'hypercube': False
        },
        {
            'n_features': 55, 
            'n_informative': 30, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 5, 
            'class_sep': 0.5, 
            'flip_y': 0.1, 
            'random_seed': 42,
            'hypercube': False
        },
        {
            'n_features': 55, 
            'n_informative': 30, 
            'n_redundant': 5, 
            'n_repeated': 5, 
            'n_classes': 2, 
            'n_samples': 100000, 
            'n_clusters_per_class': 5, 
            'class_sep': 0.7, 
            'flip_y': 0.1, 
            'random_seed': 42,
            'hypercube': False
        }
    ]
    
    # Generate setting names from configurations
    synthetic_settings = []
    for config in synthetic_configs:
        setting = (f"n_feat{config['n_features']}_"
                   f"n_informative{config['n_informative']}_"
                   f"n_redundant{config['n_redundant']}_"
                   f"n_repeated{config['n_repeated']}_"
                   f"n_classes{config['n_classes']}_"
                   f"n_samples{config['n_samples']}_"
                   f"n_clusters_per_class{config['n_clusters_per_class']}_"
                   f"class_sep{config['class_sep']}_"
                   f"flip_y{config['flip_y']}_"
                   f"random_state{config['random_seed']}")
        if not config['hypercube']:
            setting += f"_hypercube{config['hypercube']}"
        synthetic_settings.append((setting, config))
    
    models = ["LightGBM", "MLP", "TabNet", "FTTransformer", "ResNet", "LogisticRegression", "TabTransformer"]
    standard_settings = ["higgs", "jannis", "MiniBooNE"]
    methods = ["lime", "gradient_methods"]
    distance_measures = ["euclidean", "manhattan", "cosine"]
    
    # Generate all command files
    created_files = []
    
    # First, handle standard benchmark datasets
    for model in models:
        for setting in standard_settings:
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
                        dataset_dir = os.path.join(output_dir, method, gradient_dir, model, "synthetic_data", dataset)
                        dataset_files = [f for f in created_files if f"{method}/{gradient_dir}/{model}/synthetic_data/{dataset}/" in f]
                        
                        if dataset_files:
                            dataset_run_all = os.path.join(dataset_dir, "run_all.sh")
                            with open(dataset_run_all, "w") as f:
                                f.write("#!/bin/bash\n\n")
                                f.write(f"# Run all experiments for {method}/{gradient_dir}/{model}/synthetic_data/{dataset}\n\n")
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
                    dataset_dir = os.path.join(output_dir, method, model, "synthetic_data", dataset)
                    dataset_files = [f for f in created_files if f"{method}/{model}/synthetic_data/{dataset}/" in f]
                    
                    if dataset_files:
                        dataset_run_all = os.path.join(dataset_dir, "run_all.sh")
                        with open(dataset_run_all, "w") as f:
                            f.write("#!/bin/bash\n\n")
                            f.write(f"# Run all experiments for {method}/{model}/synthetic_data/{dataset}\n\n")
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
