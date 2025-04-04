import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression


def apply_nonlinearity(X, y, regression_mode, n_informative):
    """Apply non-linearities to ALL informative features."""
    X_informative = X[:, :n_informative]  # First n_informative features are the true signals
    
    if regression_mode == "linear":
        return y
    
    elif regression_mode == "polynomial":
        # Quadratic and cubic terms for all informative features
        y_nonlinear = y + 0.3 * np.sum(X_informative**2, axis=1) - 0.1 * np.sum(X_informative**3, axis=1)
    
    elif regression_mode == "interaction":
        # All pairwise interactions between informative features
        interaction_terms = 0
        for i in range(n_informative):
            for j in range(i + 1, n_informative):
                interaction_terms += X_informative[:, i] * X_informative[:, j]
        y_nonlinear = y + 0.5 * interaction_terms
    
    elif regression_mode == "periodic":
        # Sinusoidal effects for all informative features (different frequencies)
        periodic_terms = 0
        for i in range(n_informative):
            periodic_terms += (i + 1) * np.sin((i + 2) * X_informative[:, i])
        y_nonlinear = y + periodic_terms
    
    elif regression_mode == "hierarchical":
        # Multiplicative hierarchy: x0 * (x1 + x2) + x3^2, etc.
        hierarchical_term = (X_informative[:, 0] * (X_informative[:, 1] + X_informative[:, 2])) + 0.5 * X_informative[:, 3]**2
        y_nonlinear = y + hierarchical_term
    
    elif regression_mode == "neural_like":
        # Simulate a neuron-like activation: tanh weighted sum
        weights = np.linspace(0.5, 1.5, n_informative)
        activated = np.tanh(np.dot(X_informative, weights))
        y_nonlinear = y + 2 * activated
    
    else:
        raise ValueError(f"Unknown regression_mode: {regression_mode}")
    
    return y_nonlinear

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

def get_setting_name_classification(n_features,
                                    n_informative,
                                    n_redundant,
                                    n_repeated,
                                    n_classes,
                                    n_samples,
                                    n_clusters_per_class,
                                    class_sep,
                                    flip_y,
                                    random_seed,
                                    hypercube):
    setting_name = (f'n_feat{n_features}_n_informative{n_informative}_n_redundant{n_redundant}'
                    f'_n_repeated{n_repeated}_n_classes{n_classes}_n_samples{n_samples}'
                    f'_n_clusters_per_class{n_clusters_per_class}_class_sep{class_sep}'
                    f'_flip_y{flip_y}_random_state{random_seed}')
    if not hypercube:
        setting_name += f'_hypercube{hypercube}'
    return setting_name

def create_synthetic_classification_data_sklearn(n_features, 
                                   n_informative, 
                                   n_redundant, 
                                   n_repeated, 
                                   n_classes, 
                                   n_samples, 
                                   n_clusters_per_class, 
                                   class_sep, 
                                   flip_y, 
                                   hypercube,
                                   random_seed, 
                                   data_folder, 
                                   test_size=0.4, 
                                   val_size=0.1):
    
    setting_name = get_setting_name_classification(
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_samples=n_samples,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        random_seed=random_seed,
        hypercube=hypercube
    )
    file_path = os.path.join(data_folder, f'{setting_name}.npz')
    
    if os.path.exists(file_path):
        data = np.load(file_path)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
    else:
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=n_informative, 
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class, # Controls feature dependence by grouping samples into clusters
            class_sep=class_sep, # Higher values = features more separated between classes
            flip_y=flip_y, # Add noise by randomly flipping labels
            hypercube=hypercube,
            random_state=random_seed
        )
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        np.savez(file_path, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
    return setting_name, X_train, X_val, X_test, y_train, y_val, y_test


def create_synthetic_regression_data_sklearn(regression_mode,
                                             n_features, 
                                            n_informative, 
                                            n_samples, 
                                            noise, 
                                            bias, 
                                            random_seed, 
                                            data_folder, 
                                            test_size=0.4, 
                                            val_size=0.1,
                                            tail_strength=0.5,  # Only used if `effective_rank` is specified
                                            coef=False,         # Return true coefficients
                                            effective_rank=None  # Approximate rank of the data
                                           ):
    setting_name = get_setting_name_regression(
        regression_mode, 
        n_features=n_features,
        n_informative=n_informative,
        n_samples=n_samples,
        noise=noise,
        bias=bias,
        tail_strength=tail_strength,
        coef=coef,
        effective_rank=effective_rank,
        random_seed=random_seed
    )
    
    file_path = os.path.join(data_folder, f'{setting_name}.npz')
    
    if os.path.exists(file_path):
        data = np.load(file_path)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        if coef:
            coef_true = data['coef_true']
    else:
        out = make_regression(
                n_samples=n_samples, 
                n_features=n_features,
                n_informative=n_informative, 
                noise=noise,
                bias=bias,
                tail_strength=tail_strength,
                shuffle=False,
                coef=coef,
                effective_rank=effective_rank,
                random_state=random_seed
            )
    
        X, y, coef_true = out if coef else out[0], out[1], None
        
        y = apply_nonlinearity(X, y, regression_mode, n_informative)
        column_rng = np.random.RandomState(random_seed)
        col_indices = np.arange(X.shape[1])
        column_rng.shuffle(col_indices)
        X = X[:, col_indices]
        
        if coef and coef_true is not None:
            coef_true = coef_true[col_indices]
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        if coef:
            np.savez(file_path, 
                     X_train=X_train, X_val=X_val, X_test=X_test, 
                     y_train=y_train, y_val=y_val, y_test=y_test,
                     coef_true=coef_true)
        else:
            np.savez(file_path, 
                     X_train=X_train, X_val=X_val, X_test=X_test, 
                     y_train=y_train, y_val=y_val, y_test=y_test)
    return setting_name, X_train, X_val, X_test, y_train, y_val, y_test