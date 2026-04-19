"""
Jenny Nguyen
April 17, 2026

CS5330 - Final Project: Dental X-Ray Cavity Detection
Load best saved models and evaluate on the test set.
Prints classification report, plots confusion matrices and ROC curves.
"""

import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)

from dataset import build_labels, DentalDataset, val_transforms, TEST_DIR, BATCH_SIZE
from train import build_model


# Evaluate one model on the test set 
def evaluate(model, loader, model_name, device):
    """Run the model on the test set and print classification metrics. 
    Returns the true labels, predicted probabilities, and AUC for ROC curve plotting.
    """

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)[:, 1]  # probability of cavity class
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f'\n=== {model_name} test results ===')
    print(classification_report(all_labels, all_preds, target_names=['no cavity', 'cavity']))

    auc = roc_auc_score(all_labels, all_probs)
    print(f'AUC-ROC: {auc:.4f}')

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['no cavity', 'cavity'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} - confusion matrix')
    plt.savefig(f'results/{model_name.replace("-", "")}_confusion.png')
    plt.show()

    return all_labels, all_probs, auc


# ROC curve comparison 
def plot_roc(rn_labels, rn_probs, rn_auc, dn_labels, dn_probs, dn_auc):
    """Plot both models on the same ROC axes so we can compare them directly."""

    fpr_rn, tpr_rn, _ = roc_curve(rn_labels, rn_probs)
    fpr_dn, tpr_dn, _ = roc_curve(dn_labels, dn_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_rn, tpr_rn, color='blue',   label=f'resnet-18 (AUC = {rn_auc:.3f})')
    plt.plot(fpr_dn, tpr_dn, color='orange', label=f'densenet-121 (AUC = {dn_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='random')  # diagonal = random classifier baseline
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/roc_curves.png')
    plt.show()


def main(argv):
    """Load best saved models and evaluate on the test set."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')

    # load test set
    test_labels = build_labels(TEST_DIR)
    test_set = DentalDataset(test_labels, TEST_DIR, val_transforms)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # load resnet-18 best checkpoint
    resnet = build_model('resnet18').to(device)
    resnet.load_state_dict(torch.load('results/best_resnet18.pth', weights_only=True))
    rn_labels, rn_probs, rn_auc = evaluate(resnet, test_loader, 'resnet-18', device)

    # load densenet-121 best checkpoint
    densenet = build_model('densenet121').to(device)
    densenet.load_state_dict(torch.load('results/best_densenet121.pth', weights_only=True))
    dn_labels, dn_probs, dn_auc = evaluate(densenet, test_loader, 'densenet-121', device)

    # plot ROC curves side by side
    plot_roc(rn_labels, rn_probs, rn_auc, dn_labels, dn_probs, dn_auc)


if __name__ == '__main__':
    main(sys.argv)