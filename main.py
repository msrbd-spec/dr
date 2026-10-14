# File: main.py
import os
import argparse
import copy
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.data.dataset import DRDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.baselines import get_baseline_model
from src.models.coda_net import CODANet
from src.training.trainer import StandardTrainer, CODANetTrainer
from src.training.confident_learning import ConfidentLearningFilter
from src.utils.metrics import compute_metrics, save_classification_report, aggregate_metrics
from src.utils.visualizer import plot_curves, plot_confusion_matrix, plot_auroc, plot_tsne

def setup_experiment_dir(exp_name: str) -> dict:
    base_dir = os.path.join('results', exp_name)
    dirs = {
        'base': base_dir,
        'curves': os.path.join(base_dir, 'curves'),
        'matrices': os.path.join(base_dir, 'matrices'),
        'tables': os.path.join(base_dir, 'tables')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def extract_features_and_predict(model, dataloader, device, is_coda_net=False):
    model.eval()
    all_features, all_probs, all_preds, all_labels = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs, labels = batch[0].to(device), batch[1].numpy()
            if is_coda_net:
                logits, projections = model(imgs, None)
                features = projections 
            else:
                features = model.extract_features(imgs)
                logits = model.classifier(features)
                
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_features.append(features.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels)
            
    return (np.vstack(all_features), np.vstack(all_probs), 
            np.concatenate(all_preds), np.concatenate(all_labels))

def get_class_counts(df):
    counts = np.bincount(df['diagnosis'].values, minlength=5)
    return counts.tolist()

def run_baseline(args, data_paths: dict, device: torch.device):
    print(f"\n{'='*50}\nRunning Baseline (5-Fold CV): {args.model_name}\n{'='*50}")
    dirs = setup_experiment_dir(args.exp_name)
    
    full_train_df = pd.read_csv(data_paths['train_csv'])
    test_dataset = DRDataset(csv_path=data_paths['val_csv'], img_dir=data_paths['val_img'], transform=get_val_transforms(), is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    internal_metrics_list, external_metrics_list = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, full_train_df['diagnosis'])):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        train_df = full_train_df.iloc[train_idx]
        val_df = full_train_df.iloc[val_idx]
        
        train_dataset = DRDataset(df=train_df, img_dir=data_paths['train_img'], transform=get_train_transforms(), is_training=True)
        val_dataset = DRDataset(df=val_df, img_dir=data_paths['train_img'], transform=get_val_transforms(), is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        
        model = get_baseline_model(args.model_name).to(device)
        trainer = StandardTrainer(model, device, epochs=25, loss_type=args.loss_type, 
                                  cls_num_list=get_class_counts(train_df), use_drw=args.use_drw)
        history, best_model = trainer.train_and_evaluate(train_loader, val_loader)
        
        # Evaluate on Internal Held-out Test Set
        _, _, y_pred, y_true = extract_features_and_predict(best_model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        internal_metrics_list.append(metrics)
        print(f"Fold {fold+1} Internal -> 5-Class Acc: {metrics['Accuracy']}% | Bin Acc: {metrics['Bin_Acc']}% | QWK: {metrics['QWK']}")
        
        # Evaluate on External Messidor-2
        ext_dataset = DRDataset(csv_path=data_paths['ext_csv'], img_dir=data_paths['ext_img'], transform=get_val_transforms(), is_training=False)
        ext_loader = DataLoader(ext_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        _, _, ext_y_pred, ext_y_true = extract_features_and_predict(best_model, ext_loader, device)
        ext_metrics = compute_metrics(ext_y_true, ext_y_pred)
        external_metrics_list.append(ext_metrics)
        
        # Save plots for the last fold as representative
        if fold == 4:
            features, y_prob, _, _ = extract_features_and_predict(best_model, test_loader, device)
            plot_curves(history['train_acc'], history['val_acc'], history['train_loss'], history['val_loss'], dirs['curves'])
            plot_confusion_matrix(y_true, y_pred, dirs['matrices'])
            plot_auroc(y_true, y_prob, dirs['curves'])
            plot_tsne(features, y_true, dirs['curves'])
            torch.save(best_model.state_dict(), os.path.join(dirs['base'], f"{args.model_name}_best_fold5.pth"))

    # Aggregate and Save
    final_internal = aggregate_metrics(internal_metrics_list)
    final_external = aggregate_metrics(external_metrics_list)
    
    print(f"\n[FINAL 5-FOLD INTERNAL] 5-Class Acc: {final_internal['Accuracy']}% | Bin Acc: {final_internal['Bin_Acc']}% | QWK: {final_internal['QWK']}")
    print(f"[FINAL 5-FOLD EXTERNAL] 5-Class Acc: {final_external['Accuracy']}% | Bin Acc: {final_external['Bin_Acc']}% | QWK: {final_external['QWK']}")
    
    pd.DataFrame([final_internal]).to_csv(os.path.join(dirs['tables'], 'metrics_summary.csv'), index=False)
    pd.DataFrame([final_external]).to_csv(os.path.join(dirs['tables'], 'external_metrics_summary.csv'), index=False)
    print(f"Results saved to {dirs['base']}")

def run_coda_net(args, data_paths: dict, device: torch.device):
    print(f"\n{'='*50}\nRunning CODA-Net (5-Fold CV): {args.exp_name}\n{'='*50}")
    dirs = setup_experiment_dir(args.exp_name)
    
    valid_models = {
        'swinv2_l': 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
        'convnextv2_l': 'convnextv2_large.fcmae_ft_in22k_in1k_384',
        'resnet50': 'resnet50.tv_in1k',
        'efficientnetv2_m': 'efficientnetv2_rw_m.agc_in1k',
        'mobilenetv3_l': 'mobilenetv3_large_100',
        'maxvit_l': 'maxvit_large_tf_512.in1k',
        'densenet121': 'densenet121',
        'regnety_120': 'regnety_120',
        'vgg16_bn': 'vgg16_bn',
        'squeezenet1_1': 'squeezenet1_1'
    }
    full_backbone_name = valid_models.get(args.backbone, valid_models['convnextv2_l'])
    
    full_train_df = pd.read_csv(data_paths['train_csv'])
    
    # 1. Confident Learning Phase (Done once on the whole training set)
    if args.use_cl:
        print("\n--- Phase 1: Confident Learning (Label Noise Cleaning) ---")
        full_train_dataset = DRDataset(df=full_train_df, img_dir=data_paths['train_img'], transform=get_val_transforms(), is_training=False)
        cl_filter = ConfidentLearningFilter(num_classes=5)
        builder_fn = lambda: get_baseline_model('mobilenetv3_l', pretrained=True)
        oof_probs, noisy_labels = cl_filter.get_out_of_fold_probabilities(builder_fn, full_train_dataset, device, epochs=5)
        error_indices = cl_filter.find_label_errors(oof_probs, noisy_labels)
        
        # Drop noisy labels from the dataframe
        full_train_df = full_train_df.drop(error_indices).reset_index(drop=True)
        print(f"Cleaned Dataset Size: {len(full_train_df)} images.")

    # 2. CODA-Net 5-Fold Training Phase
    print("\n--- Phase 2: CODA-Net 5-Fold Training ---")
    use_mixstyle = (args.domain_adapt == 'mixstyle')
    use_fda = (args.domain_adapt == 'fda')
    
    test_dataset = DRDataset(csv_path=data_paths['val_csv'], img_dir=data_paths['val_img'], transform=get_val_transforms(), is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    internal_metrics_list, external_metrics_list = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, full_train_df['diagnosis'])):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        train_df = full_train_df.iloc[train_idx]
        val_df = full_train_df.iloc[val_idx]
        
        train_dataset = DRDataset(
            df=train_df, img_dir=data_paths['train_img'], transform=get_train_transforms(), is_training=True, 
            style_pool_dir=data_paths['style_pool_img'] if use_mixstyle else None,
            fda_target_pool=glob.glob(os.path.join(data_paths['style_pool_img'], "*.*")) if use_fda else None,
            fda_prob=0.3 if use_fda else 0.0
        )
        val_dataset = DRDataset(df=val_df, img_dir=data_paths['train_img'], transform=get_val_transforms(), is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        
        model = CODANet(backbone_name=full_backbone_name, pretrained=True, 
                        use_mixstyle=use_mixstyle, mixstyle_p=0.5, 
                        ssl_weights_path=args.ssl_weights).to(device)
        
        trainer = CODANetTrainer(model, device, epochs=25, use_ordinal=args.use_ordinal, alpha=0.5)
        history, best_model = trainer.train_and_evaluate(train_loader, val_loader)
        
        # Internal Eval
        _, _, y_pred, y_true = extract_features_and_predict(best_model, test_loader, device, is_coda_net=True)
        metrics = compute_metrics(y_true, y_pred)
        internal_metrics_list.append(metrics)
        print(f"Fold {fold+1} Internal -> 5-Class Acc: {metrics['Accuracy']}% | Bin Acc: {metrics['Bin_Acc']}% | QWK: {metrics['QWK']}")
        
        # External Eval
        ext_dataset = DRDataset(csv_path=data_paths['ext_csv'], img_dir=data_paths['ext_img'], transform=get_val_transforms(), is_training=False)
        ext_loader = DataLoader(ext_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        _, _, ext_y_pred, ext_y_true = extract_features_and_predict(best_model, ext_loader, device, is_coda_net=True)
        ext_metrics = compute_metrics(ext_y_true, ext_y_pred)
        external_metrics_list.append(ext_metrics)
        
        if fold == 4:
            features, y_prob, _, _ = extract_features_and_predict(best_model, test_loader, device, is_coda_net=True)
            plot_curves(history['train_acc'], history['val_acc'], history['train_loss'], history['val_loss'], dirs['curves'])
            plot_confusion_matrix(y_true, y_pred, dirs['matrices'])
            plot_auroc(y_true, y_prob, dirs['curves'])
            plot_tsne(features, y_true, dirs['curves'])
            torch.save(best_model.state_dict(), os.path.join(dirs['base'], "coda_net_best_fold5.pth"))

    # Aggregate and Save
    final_internal = aggregate_metrics(internal_metrics_list)
    final_external = aggregate_metrics(external_metrics_list)
    
    print(f"\n[FINAL 5-FOLD INTERNAL] 5-Class Acc: {final_internal['Accuracy']}% | Bin Acc: {final_internal['Bin_Acc']}% | QWK: {final_internal['QWK']}")
    print(f"[FINAL 5-FOLD EXTERNAL] 5-Class Acc: {final_external['Accuracy']}% | Bin Acc: {final_external['Bin_Acc']}% | QWK: {final_external['QWK']}")
    
    pd.DataFrame([final_internal]).to_csv(os.path.join(dirs['tables'], 'metrics_summary.csv'), index=False)
    pd.DataFrame([final_external]).to_csv(os.path.join(dirs['tables'], 'external_metrics_summary.csv'), index=False)
    print(f"Results saved to {dirs['base']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'coda_net'])
    parser.add_argument('--exp_name', type=str, required=True, help='Folder name to save results')
    
    parser.add_argument('--model_name', type=str, default='convnextv2_l', help='Timm model name for baseline')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'ldam'], help='Loss function for baseline')
    parser.add_argument('--use_drw', action='store_true', help='Enable Deferred Re-Weighting for baseline')
    
    parser.add_argument('--backbone', type=str, default='convnextv2_l', help='Winning baseline to use in CODA-Net')
    parser.add_argument('--ssl_weights', type=str, default=None, help='Path to EyePACS SSL weights')
    parser.add_argument('--use_cl', action='store_true', help='Enable Confident Learning')
    parser.add_argument('--use_ordinal', action='store_true', help='Enable Ordinal Contrastive Loss')
    parser.add_argument('--domain_adapt', type=str, default='none', choices=['none', 'fda', 'mixstyle'], help='Domain adaptation method')
    parser.add_argument('--style_source', type=str, default='messidor', choices=['eyepacs', 'messidor', 'both'], help='Which dataset to use for style pool')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Update these paths to match your local dataset structure
    data_paths = {
        'train_csv': 'data/aptos/train.csv',
        'train_img': 'data/aptos/train_images/',
        'val_csv': 'data/aptos/val.csv',
        'val_img': 'data/aptos/val_images/',
        'ext_csv': 'data/messidor/messidor_data.csv',
        'ext_img': 'data/messidor/images/',
        'style_pool_img': '' 
    }
    
    if args.mode == 'baseline':
        run_baseline(args, data_paths, device)
        
    elif args.mode == 'coda_net':
        if args.style_source == 'both':
            sources = ['eyepacs', 'messidor']
            for src in sources:
                print(f"\n\n{'*'*60}\nRunning CODA-Net with Style Source: {src.upper()}\n{'*'*60}")
                data_paths['style_pool_img'] = 'data/messidor/images/' if src == 'messidor' else 'data/eyepacs/images/'
                suffix = 'uda' if src == 'messidor' else 'dg'
                
                current_args = copy.deepcopy(args)
                current_args.exp_name = f"{args.exp_name}_{suffix}"
                current_args.style_source = src
                
                run_coda_net(current_args, data_paths, device)
        else:
            data_paths['style_pool_img'] = 'data/messidor/images/' if args.style_source == 'messidor' else 'data/eyepacs/images/'
            run_coda_net(args, data_paths, device)