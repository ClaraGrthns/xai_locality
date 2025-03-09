import os
import yaml

DATASETS = {
    'standard': ['higgs', 'jannis', 'MiniBooNE'],
    'synthetic': [
        'n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42',
        'n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42',
        'n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42'
    ]
}

MODELS = {
    'gbt': ['XGBoost', 'CatBoost', 'LightGBM'],
    'deep': ['TabNet', 'FTTransformer', 'ResNet', 'MLP', 'TabTransformer',
             'Trompt', 'ExcelFormer', 'FTTransformerBucket']
}

# Model type mapping for GBT models
GBT_MODEL_TYPES = {
    'LightGBM': 'pt_frame_lgm',
    "XGBoost": 'pt_frame_xgboost',
}

def get_gbt_paths_synthetic_data(model, dataset):
    model_path = f'/home/grotehans/xai_locality/pretrained_models/{model}/synthetic_data/{model}_{dataset}.pt'
    return model_path

def create_lime_config(model, dataset, is_synthetic=False):
    # Determine if it's a GBT model
    is_gbt = model in MODELS['gbt']
    
    # Get correct model type for GBT models
    if model in GBT_MODEL_TYPES:
        model_type = GBT_MODEL_TYPES[model]#['synthetic' if is_synthetic else 'standard']
    else:
        model_type = model

    # Get correct paths based on model type
    if is_synthetic:
        model_path = get_gbt_paths_synthetic_data(model, dataset) if is_gbt else f'/home/grotehans/xai_locality/pretrained_models/{model}/synthetic_data/{model}_{dataset}_results.pt'
        data_path = f'/home/grotehans/xai_locality/data/synthetic_data/{dataset}_normalized_tensor_frame.pt'
    else:
        model_path = f'/home/grotehans/xai_locality/pretrained_models/{model}/{dataset}/{model}_normalized_binary_{dataset}_results.pt'
        data_path =  f'/home/grotehans/xai_locality/data/{model}_{dataset}_normalized_data.pt'
    
    config = {
        'explanation_method': {
            'method': 'lime'
        },
        'paths': {
            'results_path': f'/home/grotehans/xai_locality/results/lime/{model}/{"synthetic_data/" if is_synthetic else ""}{dataset}',
            'data_path': data_path,
            'model_path': model_path
        },
        'model': {
            'model_type': model_type
        },
        'analysis': {
            'num_features': 50,
            'num_samples': 1000,
            'random_seed': 42,
            'chunk_size': 20
        },
        'other': {
            'max_test_points': 200
        }
    }
    return config

def main():
    base_path = '/home/grotehans/xai_locality/configs'
    
    # Generate LIME configs for all models
    for model_type, models in MODELS.items():
        for model in models:
            # Standard datasets
            for dataset in DATASETS['standard']:
                path = f'{base_path}/lime/{model}/{dataset}'
                os.makedirs(path, exist_ok=True)
                
                config = create_lime_config(model, dataset)
                with open(f'{path}/config.yaml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            # Synthetic datasets
            for dataset in DATASETS['synthetic']:
                path = f'{base_path}/lime/{model}/synthetic_data/{dataset}'
                os.makedirs(path, exist_ok=True)
                
                config = create_lime_config(model, dataset, is_synthetic=True)
                with open(f'{path}/config.yaml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

if __name__ == '__main__':
    main()