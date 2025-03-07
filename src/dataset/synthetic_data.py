import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def create_synthetic_data_sklearn(n_features, 
                                   n_informative, 
                                   n_redundant, 
                                   n_repeated, 
                                   n_classes, 
                                   n_samples, 
                                   n_clusters_per_class, 
                                   class_sep, 
                                   flip_y, 
                                   random_seed, 
                                   data_folder, 
                                   test_size=0.4, 
                                   val_size=0.1):
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
        random_state=random_seed
    )
    X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_seed)

    setting_name = f'n_feat{n_features}_n_informative{n_informative}_n_redundant{n_redundant}_n_repeated{n_repeated}_n_classes{n_classes}_n_samples{n_samples}_n_clusters_per_class{n_clusters_per_class}_class_sep{class_sep}_flip_y{flip_y}_random_state{random_seed}'
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
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        np.savez(file_path, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
    
    return setting_name, X_train, X_val, X_test, y_train, y_val, y_test
