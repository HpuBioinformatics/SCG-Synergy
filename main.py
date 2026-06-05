# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import Data_Process
import model as Synergy_Models 
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import mlflow.pytorch
import Config
import os
import sys
import random

# ==================== [Set Seed] ====================
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== [Config & Paths] ====================
OUTPUT_DIR = ''
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class Logger(object):
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

sys.stdout = Logger(os.path.join(OUTPUT_DIR, 'run_log.txt'))
print(f"Results and text logs will be saved to: {OUTPUT_DIR}")

# ==================== [Metrics] ====================
def metrics(labels, predictions, epoch, type):
    auc = roc_auc_score(labels, predictions) 
    aupr = average_precision_score(labels, predictions)
    binary_predictions = (np.array(predictions) >= 0.5).astype(int)
    binary_labels = np.array(labels).astype(int)
    f1 = f1_score(binary_labels, binary_predictions)
    accuracy = accuracy_score(binary_labels, binary_predictions)
    
    mlflow.log_metric(key=f"auc-{type}", value=float(auc), step=epoch)
    mlflow.log_metric(key=f"aupr-{type}", value=float(aupr), step=epoch)
    mlflow.log_metric(key=f"f1-{type}", value=float(f1), step=epoch)
    mlflow.log_metric(key=f"accuracy-{type}", value=float(accuracy), step=epoch)
    return auc, aupr, f1, accuracy

# ==================== [Test Function] ====================
def test(edges, labels):
    Model.eval()
    with torch.no_grad():
        preds, _ = Model(Drug_Features, Cell_Line_Feature, edges)
        loss = 0
        pred = []
        real = []
        for idx in range(Data.numDrug, Data.numNode):
            if idx not in preds: continue
            
            pred.extend(torch.sigmoid(preds[idx]).cpu().detach().numpy())
            real.extend(labels[idx].cpu().detach().numpy())
            
            each_preds = preds[idx]
            each_labels = labels[idx]

            criterion = nn.BCEWithLogitsLoss()
            each_loss = criterion(each_preds, each_labels)
            loss += each_loss
        loss = loss / Data.CellsCount
    return loss, pred, real 
 
# ==================== [Train Function] ====================
def train(Drug_Features, Data, epochs):
    Model.train()
    
    best_val_metric = 0 # Using Accuracy or AUC to track best model
    best_model_state = None
    best_epoch = 0
    patience = 50    
    counter = 0        

    for epoch in range(epochs):
        preds, cl_loss = Model(Drug_Features, Cell_Line_Feature, Data.train_edges)
        loss_train = 0
        train_pred = []
        train_real = []
        
        # 2. Compute Supervised Loss (Synergy Prediction)
        for idx in range(Data.numDrug, Data.numNode):
            if idx not in preds: continue 
            
            train_real.extend(Data.train_labels[idx].cpu().detach().numpy())
            each_labels = Data.train_labels[idx]
            
            # Handle bidirectional edges if necessary
            half_len = len(each_labels)
            if preds[idx].shape[0] >= 2 * half_len:
                 each_preds = (torch.add(preds[idx][:half_len], preds[idx][half_len:2*half_len])) / 2
            else:
                 each_preds = preds[idx] 

            train_pred.extend(torch.sigmoid(each_preds).cpu().detach().numpy())

            each_pos_weight = Data.pos_weights[idx]
            criterion = nn.BCEWithLogitsLoss(pos_weight=each_pos_weight)
            each_loss = criterion(each_preds, each_labels)

            loss_train += each_loss

        # 3. Combine Losses
        # args.alpha controls the weight of the Contrastive Learning task
        avg_train_loss = loss_train / Data.CellsCount
        total_loss = avg_train_loss + args.alpha * cl_loss
        
        # 4. Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient Clipping (Optional but recommended for Transformers/GNNs)
        torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # 5. Logging
        train_results = metrics(train_real, train_pred, epoch, 'train')
        mlflow.log_metric(key="Total Loss", value=float(total_loss), step=epoch)
        mlflow.log_metric(key="CL Loss", value=float(cl_loss), step=epoch)

        # 6. Validation
        loss_valid, valid_pred, valid_real = test(Data.valid_edges, Data.valid_labels)
        valid_results = metrics(valid_real, valid_pred, epoch, 'valid')     
        mlflow.log_metric(key="valid loss", value=float(loss_valid), step=epoch)
        
        # 7. Early Stopping (Monitoring AUC here, can change to ACC)
        current_metric = valid_results[0] # 0=AUC, 3=ACC
        
        if current_metric > best_val_metric: 
            best_val_metric = current_metric
            best_model_state = Model.state_dict()
            best_epoch = epoch
            counter = 0 
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Metric: {best_val_metric:.6f}")
            break 

        if epoch % 10== 0:
            print(f'Epoch: {epoch} | Total Loss: {total_loss:.6f} | CL Loss: {cl_loss:.6f}')
            print('   Train -> AUC: {:.4f}, AUPR: {:.4f}, F1: {:.4f}, ACC: {:.4f}'.format(*train_results))
            print('   Valid -> AUC: {:.4f}, AUPR: {:.4f}, F1: {:.4f}, ACC: {:.4f}'.format(*valid_results))
    return best_model_state, best_epoch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)   

# ==================== [Main Execution] ====================
if __name__ == '__main__':
    args = Config.parse()
    set_seed(args.seed)
    
    # Ensure temperature is in args, otherwise default to 0.5
    if not hasattr(args, 'temperature'):
        args.temperature = 0.5
        print("Warning: 'temperature' not found in Config, defaulting to 0.5")

    # MLflow Setup
    mlflow_tracking_uri = "file://" + os.path.join(OUTPUT_DIR, "mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow data will be saved to: {mlflow_tracking_uri}")
    
    experiment = mlflow.set_experiment("Experiment-" + args.dataset)
    mlflow.log_param("Dataset_Name", args.dataset)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("temperature", args.temperature)

    # Load Data
    realFolds, Protein_Adjacency_Matrix, Drug_Adjacency_Matrix, Drug_Target_Matrix, numNode1, Drug_Features, Cell_Line_Feature, _, _, _, V1, E1, edge_num1, edge_length1, degV1 = Data_Process.process_data(args.dataset)
    
    Drug_Features = torch.tensor(Drug_Features).float().to(args.device)
    Cell_Line_Feature = torch.tensor(Cell_Line_Feature).float().to(args.device)
    V1 = torch.tensor(V1).long().to(args.device)
    E1 = torch.tensor(E1).long().to(args.device)

    for i in range(len(degV1)):
        degV1[i] = torch.tensor(degV1[i]).float().to(args.device)
    
    results = []
    
    for fold in range(args.k_fold):
        mlflow.start_run(nested=True)
        set_seed(args.seed)
        
        Data = realFolds[fold]
        Data = Data_Process.torch_from_numpy(Data, args.device)
        
        # ================= [Initialize Contrastive Loss Module] =================
        cl_module = Synergy_Models.ContrastiveLoss(
            feature_dim=256,    
            hidden_dim=128, 
            temperature=args.temperature
        ).to(args.device)
        
        # ================= [Initialize Model] =================
        Model = Synergy_Models.Synergy(
            numDrug = Data.numDrug, 
            BioEncoder = Synergy_Models.BioEncoder(Drug_Features.shape[1], Cell_Line_Feature.shape[1], Data.CellsCount, numNode1 - Data.numDrug, 512, device = args.device), 
            encoder1 = Synergy_Models.RHGNN(V1, E1, edge_num1, edge_length1, degV1, 512, 256, 256, num_edge_types = 4, dropout = 0.2),
            encoder2 = Synergy_Models.RHGNN(Data.V, Data.E, Data.hypergraph_edge_num, Data.edge_length, Data.degV_dict, 512, 128, 256, num_edge_types = 2, dropout = 0),
            attention = Synergy_Models.ChannelAttention(emb_size = 256),
            decoder = Synergy_Models.ContextGatedSelfAttentionPredictor(feature_dim = 256, nhead = 4, dim_feedforward = 512, dropout = 0.2),
            contrastive_loss_module = cl_module  # Inject CL module
        ).to(args.device)
        
        # ================= [Optimizer] =================
        # Since cl_module is registered inside Synergy, Model.parameters() includes it.
        # No need to add SSL_agent_protein parameters anymore.
        optimizer = torch.optim.Adam(Model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
        
        mlflow.log_param("num_params", count_parameters(Model))
        print(f"Fold {fold+1} Training...")
        
        best_model_state, best_epoch = train(Drug_Features, Data, args.epochs)
        
        Model.load_state_dict(best_model_state)
        print(f"Saving model for Fold {fold+1}...")
        mlflow.pytorch.log_model(Model, "model")
        
        loss_test, test_pred, test_real = test(Data.test_edges, Data.test_labels)
        test_results = metrics(test_real, test_pred, best_epoch, 'test')

        results.append(list(test_results))
        
        print(f'Fold {fold+1} Test Results -> AUC: {test_results[0]:.6f}, AUPR: {test_results[1]:.6f}, F1: {test_results[2]:.6f}, ACC: {test_results[3]:.6f}')
        mlflow.log_metric(key="results", value=float(test_results[1]), step=fold)
        mlflow.end_run()
    
    results = pd.DataFrame(results).to_numpy()
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    print('\n' + '='*60)
    print('Final Summary (Mean ± Std):')
    print(f'AUC : {means[0]:.6f} ± {stds[0]:.6f}')
    print(f'AUPR: {means[1]:.6f} ± {stds[1]:.6f}')
    print(f'F1  : {means[2]:.6f} ± {stds[2]:.6f}')
    print(f'ACC : {means[3]:.6f} ± {stds[3]:.6f}')
    print('='*60)
    output_file_path = os.path.join(OUTPUT_DIR, "result-" + args.dataset + ".txt")                                  
    with open(output_file_path, "w") as file:
        file.write("Fold\tAUC\t\tAUPR\t\tF1\t\tACC\n")
        file.write("="*60 + "\n")
        for i, item in enumerate(results):
            file.write(f"Fold {i+1}\t{item[0]:.6f}\t{item[1]:.6f}\t{item[2]:.6f}\t{item[3]:.6f}\n")
        file.write("="*60 + "\n")
        file.write(f"Mean:\t{means[0]:.6f}\t{means[1]:.6f}\t{means[2]:.6f}\t{means[3]:.6f}\n")
        file.write(f"Std:\t{stds[0]:.6f}\t{stds[1]:.6f}\t{stds[2]:.6f}\t{stds[3]:.6f}\n")
    print(f"All results saved to {output_file_path}")