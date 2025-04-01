import sys

import os
import os.path as osp
sys.path.append(osp.join(os.getcwd(), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(sys.path)


import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import datetime


from src.model.pytorch_models_handler import LogisticRegression
from src.utils.pytorch_frame_utils import tensorframe_to_tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Train a logistic regression model')
    parser.add_argument('--dataset', type=str,default = "higgs")# default='synthetic_data/n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42', help="Can be 'higgs', 'jannis,")# default='synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42', help="Can be 'higgs', 'jannis, 'synthetic_1', 'synthetic_2', 'synthetic_3'")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print progress during training')
    parser.add_argument('--optimize', action='store_true', help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna optimization trials')
    parser.add_argument('--data_path', type=str, default='/home/grotehans/xai_locality/data/LightGBM_higgs_normalized_data.pt', help='Path to the dataset')
    parser.add_argument('--model_path', type=str, default='/home/grotehans/xai_locality/pretrained_models/LogisticRegression/higgs/LogisticRegression_higgs_results.pt', help='Path to save the trained model')
    return parser.parse_args()


def train_model(X, y, X_val, y_val, model, optimizer, criterion, epochs, weight_decay=0.0, verbose=False):
    # Training loop with validation
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X).flatten()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).flatten()
            val_loss = criterion(val_outputs, y_val).item()
            
            # Save best model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
    
    # Restore best model based on validation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Return best validation loss
    return model, best_val_loss

def objective(trial, X, y, X_val, y_val, input_size, epochs, verbose):
    # Define the hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-1, log=True)
    
    # Initialize model
    model = LogisticRegression(input_size=input_size, output_size=1)
    criterion = nn.BCELoss()
    
    # Set up optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train the model and get validation loss
    _, val_loss = train_model(X, y, X_val, y_val, model, optimizer, criterion, epochs, verbose=verbose)
    
    return val_loss


def main(args=None):
    if args is None:
        args = parse_args()
    print(args)
    
    # Create TensorBoard logger
    log_dir = f"runs/logistic_regression_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    data_path = args.data_path
    model_path = args.model_path
    if not osp.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    data = torch.load(data_path)
    train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
    X = tensorframe_to_tensor(train_tensor_frame)  # All training features
    y = train_tensor_frame.y.float()  # All training labels as float
    X_val = tensorframe_to_tensor(val_tensor_frame)  # Validation features
    y_val = val_tensor_frame.y.float()  # Validation labels as float
    X_test = tensorframe_to_tensor(test_tensor_frame)
    y_test = test_tensor_frame.y.float()  # Test labels as float

    input_size = X.shape[1]
    
    if args.optimize:
        print("Starting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X, y, X_val, y_val, input_size, args.epochs, args.verbose), 
            n_trials=args.n_trials
        )
        # Get the best parameters
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
        writer.add_hparams(best_params, {'hparam/best_loss': study.best_value})
        
        # Train model with the best parameters
        model = LogisticRegression(input_size=input_size, output_size=1)
        criterion = nn.BCELoss()
        
        if best_params["optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=best_params["lr"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        
        # Training with TensorBoard logging and validation
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X).flatten()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Loss/train', loss.item(), epoch)
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).flatten()
                val_loss = criterion(val_outputs, y_val)
                val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                val_auroc = roc_auc_score(y_val.cpu().numpy(), val_probs)
                val_accuracy = ((val_probs > 0.5) == y_val.cpu().numpy()).mean()
                writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalar('Metrics/val_auroc', val_auroc, epoch)
                writer.add_scalar('Metrics/val_accuracy', val_accuracy, epoch)
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
            
            if args.verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}')
    else:
        # Use default parameters
        model = LogisticRegression(input_size=input_size, output_size=1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # Training with TensorBoard logging and validation
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X).flatten()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            accuracy = ((outputs > 0.5) == y).float().mean()
            
            writer.add_scalar('Loss/train', loss.item(), epoch)
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).flatten()
                val_loss = criterion(val_outputs, y_val).item()
                val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                val_auroc = roc_auc_score(y_val.cpu().numpy(), val_probs)
                
                writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalar('Metrics/val_auroc', val_auroc, epoch)
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
            
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.4f} Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}')
    
    # Load the best model for testing
    best_model = LogisticRegression(input_size=input_size, output_size=1)
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()
    
    # Testing
    with torch.no_grad():
        probs = best_model(X_test).cpu().numpy().flatten()
        label_preds = (probs > 0.5).astype(int)
        auroc = roc_auc_score(y_test, probs)
        acc = (label_preds == y_test.cpu().numpy()).mean()
        
        # Log test metrics
        writer.add_scalar('Metrics/test_auroc', auroc)
        writer.add_scalar('Metrics/test_accuracy', acc)
        
        print(f"Test AUROC: {auroc:.4f}")
        print(f"Test Accuracy: {acc:.4f}")

        

    
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
