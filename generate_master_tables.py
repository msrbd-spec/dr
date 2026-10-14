# File: generate_master_tables.py
import os
import pandas as pd

def get_metrics(run_dir):
    csv_path = os.path.join('results', run_dir, 'tables', 'metrics_summary.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.iloc[0].to_dict()
    return {'Accuracy': '-', 'QWK': '-', 'Precision': '-', 'Recall': '-', 'F1_Score': '-', 
            'Bin_Acc': '-', 'Bin_Prec': '-', 'Bin_Rec': '-', 'Bin_F1': '-'}

def get_ext_metrics(run_dir):
    csv_path = os.path.join('results', run_dir, 'tables', 'external_metrics_summary.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.iloc[0].to_dict()
    return {'Accuracy': '-', 'QWK': '-', 'Precision': '-', 'Recall': '-', 'F1_Score': '-', 
            'Bin_Acc': '-', 'Bin_Prec': '-', 'Bin_Rec': '-', 'Bin_F1': '-'}

def generate_tables():
    os.makedirs('results/master_tables', exist_ok=True)
    
    # Table 1: 10-Model Baseline Plateau
    baselines = [
        'swinv2_l', 'convnextv2_l', 'resnet50', 'efficientnetv2_m', 
        'mobilenetv3_l', 'maxvit_l', 'densenet121', 'regnety_120', 
        'vgg16_bn', 'squeezenet1_1'
    ]
    t1_data = []
    for b in baselines:
        metrics = get_metrics(f'baseline_{b}')
        t1_data.append({'Model': b.upper(), **metrics})
    pd.DataFrame(t1_data).to_csv('results/master_tables/Table_1_Baselines.csv', index=False)
    print("Generated Table 1: Baselines")

    # Table 2: SSL Pretraining Impact
    t2_data = [
        {'Configuration': 'ImageNet Weights', **get_metrics('ablation_ssl_imagenet')},
        {'Configuration': 'EyePACS Supervised', **get_metrics('ablation_ssl_supervised')},
        {'Configuration': 'Contrastive Only', **get_metrics('ablation_ssl_contrastive')},
        {'Configuration': 'Multitask Only', **get_metrics('ablation_ssl_multitask')},
        {'Configuration': 'Full SSL (Lesion-Aware)', **get_metrics('ablation_ssl_full')}
    ]
    pd.DataFrame(t2_data).to_csv('results/master_tables/Table_2_SSL_Impact.csv', index=False)
    print("Generated Table 2: SSL Impact")

    # Table 3: Label Noise & Imbalance Resolution
    t3_data = [
        {'Configuration': 'Standard Cross-Entropy', **get_metrics('ablation_ssl_full')},
        {'Configuration': 'LDAM + DRW (Baseline)', **get_metrics('ablation_ldam_drw')},
        {'Configuration': 'Confident Learning (Proposed)', **get_metrics('ablation_noise_cleaned')}
    ]
    pd.DataFrame(t3_data).to_csv('results/master_tables/Table_3_Label_Noise.csv', index=False)
    print("Generated Table 3: Label Noise")

    # Table 4: Grade Boundary Resolution
    t4_data = [
        {'Configuration': 'Cross-Entropy Loss', **get_metrics('ablation_noise_cleaned')},
        {'Configuration': 'Ordinal Contrastive Loss', **get_metrics('ablation_loss_ordinal')}
    ]
    pd.DataFrame(t4_data).to_csv('results/master_tables/Table_4_Grade_Boundary.csv', index=False)
    print("Generated Table 4: Grade Boundary")

    # Table 5: Domain Generalization (External Metrics)
    t5_data = [
        {'Configuration': 'No Domain Adaptation', **get_ext_metrics('ablation_domain_none')},
        {'Configuration': 'Pixel-Space FDA (Baseline Bug)', **get_ext_metrics('ablation_domain_fda')},
        {'Configuration': 'Feature-Space MixStyle (Zero-Shot DG via EyePACS)', **get_ext_metrics('ablation_domain_mixstyle_dg')},
        {'Configuration': 'Feature-Space MixStyle (UDA via Messidor)', **get_ext_metrics('ablation_domain_mixstyle_uda')}
    ]
    pd.DataFrame(t5_data).to_csv('results/master_tables/Table_5_Domain_Gen.csv', index=False)
    print("Generated Table 5: Domain Generalization")

    # Table 6: Final SOTA Comparison
    t6_data = [
        {'Method': 'EffNet-SVM (Naveen et al., 2025)', 'Accuracy': '-', 'Bin_Acc': '97.26', 'QWK': '-', 'F1_Score': '-', 'Bin_F1': '97.46'},
        {'Method': 'KD-MobileNet (Dejene et al., 2025)', 'Accuracy': '93.03 (Ternary)', 'Bin_Acc': '98.38', 'QWK': '-', 'F1_Score': '93.07', 'Bin_F1': '98.36'},
        {'Method': 'Q-SqueezeNet (Bilal et al., 2025)', 'Accuracy': '99.80 (IDRiD only)', 'Bin_Acc': '-', 'QWK': '-', 'F1_Score': '100.0', 'Bin_F1': '-'},
        {'Method': 'CODA-Net (Proposed)', **get_metrics('ablation_domain_mixstyle_dg')}
    ]
    pd.DataFrame(t6_data).to_csv('results/master_tables/Table_6_SOTA.csv', index=False)
    print("Generated Table 6: SOTA Comparison")
    print("\nAll tables saved to results/master_tables/")

if __name__ == '__main__':
    generate_tables()