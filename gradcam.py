"""
Jenny Nguyen
April 17,2026

CS5330 - Final Project: Dental X-Ray Cavity Detection
Grad-CAM visualizations showing which regions of the X-ray each model
focuses on when making its cavity/no-cavity prediction.

"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from dataset import build_labels, DentalDataset, val_transforms, TEST_DIR, IMG_SIZE
from train import build_model


# Generate Grad-CAM heatmap for a single image 
def get_gradcam_heatmap(model, img_tensor, target_layer, device):
    """Generate a Grad-CAM heatmap for a single image.

    Hooks into the target conv layer and computes the gradient of the
    predicted class score with respect to the feature maps.
    High gradient = that region matters for the prediction.
    """
    gradients  = []
    activations = []

    # hooks to capture gradients and activations from the target layer
    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook = target_layer.register_forward_hook(save_activation)

    # forward pass
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    out = model(img_tensor)
    pred_class = out.argmax(dim=1).item()

    # backward pass on the predicted class score
    model.zero_grad()
    out[0, pred_class].backward()

    hook.remove()

    # compute weighted combination of feature maps
    grad = gradients[0].squeeze().cpu().detach().numpy()
    activ = activations[0].squeeze().cpu().detach().numpy()
    weights = grad.mean(axis=(1, 2))  # global average pooling over spatial dims

    heatmap = np.zeros(activ.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * activ[i]

    # relu to keep only positive influences, then normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap, pred_class


# Show Grad-CAM overlays on sample images 
def show_gradcam(model, model_name, target_layer, dataset, device, n=6):
    """Show Grad-CAM heatmaps overlaid on original X-ray images.

    Picks a mix of cavity and no-cavity samples so we can see what
    the model focuses on for each class. Green title = correct, red = wrong.
    """
    # grab some cavity and no-cavity samples from the test set
    cavity_idxs = dataset.labels[dataset.labels['label'] == 1].index.tolist()[:n//2]
    no_cavity_idxs = dataset.labels[dataset.labels['label'] == 0].index.tolist()[:n//2]
    idxs = cavity_idxs + no_cavity_idxs

    fig, axes = plt.subplots(2, n, figsize=(18, 6))

    for col, idx in enumerate(idxs):
        img_tensor, true_label = dataset[idx]
        heatmap, pred = get_gradcam_heatmap(model, img_tensor, target_layer, device)

        # original image (unnormalized for display)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # resize heatmap to match image and overlay with jet colormap
        heatmap_resized = np.array(
            Image.fromarray(np.uint8(255 * heatmap)).resize((IMG_SIZE, IMG_SIZE))
        ) / 255.0
        colored = cm.jet(heatmap_resized)[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * colored  # blend original + heatmap

        true_str = 'cavity' if true_label == 1 else 'no cavity'
        pred_str = 'cavity' if pred == 1 else 'no cavity'
        color = 'green' if pred == true_label else 'red'

        axes[0, col].imshow(img_np, cmap='gray')
        axes[0, col].set_title(f'true: {true_str}', fontsize=8)
        axes[0, col].axis('off')

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f'pred: {pred_str}', fontsize=8, color=color)
        axes[1, col].axis('off')

    plt.suptitle(f'Grad-CAM: {model_name} (green = correct, red = wrong)')
    plt.tight_layout()
    plt.savefig(f'results/gradcam_{model_name.replace("-", "")}.png')
    plt.show()


def main(argv):
    """Load best saved models and generate Grad-CAM visualizations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')

    # load test set
    test_labels = build_labels(TEST_DIR)
    test_set = DentalDataset(test_labels, TEST_DIR, val_transforms)

    # resnet-18 grad-cam
    resnet = build_model('resnet18').to(device)
    resnet.load_state_dict(torch.load('results/best_resnet18.pth', weights_only=True))
    print('\ngenerating grad-cam for resnet-18...')
    show_gradcam(resnet, 'resnet-18', resnet.layer4[-1].conv2, test_set, device)

    # densenet-121 grad-cam
    densenet = build_model('densenet121').to(device)
    densenet.load_state_dict(torch.load('results/best_densenet121.pth', weights_only=True))
    print('\ngenerating grad-cam for densenet-121...')
    show_gradcam(densenet, 'densenet-121',
                 densenet.features.denseblock4.denselayer16.conv2, test_set, device)


if __name__ == '__main__':
    main(sys.argv)