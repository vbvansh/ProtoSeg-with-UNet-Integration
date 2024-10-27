import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import CityscapesDataset, get_data_loaders
from helpers import list_of_distances, make_one_hot
from model import PPNet
from Pytorch_UNet_master.unet.unet_model import UNet

# Define root directory for dataset
root_dir = "D:/Research Internship IISER Bhopal/proto-segmentation-master sacha/proto-segmentation-master/CityScapes Dataset"

# Define model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize model
unet_model = UNet(n_channels=3, n_classes=19)
proto_layer_rf_info = {'rf_size': 1, 'stride': 1, 'padding': 0}
model = PPNet(features=unet_model, img_size=256, prototype_shape=(1995, 512, 1, 1), 
              proto_layer_rf_info=proto_layer_rf_info, num_classes=19).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def calculate_mIoU(pred, target, num_classes=19):
    iou = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union == 0:
            iou.append(np.nan)
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        input = image.to(device)
        # Reshape target to be [B, H, W] instead of [B, 1, H, W]
        target = label.squeeze(1).to(device)  # Remove channel dimension
        
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # Get model output
            output, min_distances = model(input)
            
            # Print shapes for debugging
            if i == 0:
                print(f"Input shape: {input.shape}")
                print(f"Output shape: {output.shape}")
                print(f"Target shape: {target.shape}")
            
            # Handle the case where output is not in the expected format
            if len(output.shape) == 1:
                # If output is a 1D tensor of size 19 (num_classes)
                B = input.size(0)  # Batch size
                H = W = input.size(2)  # Assuming square input
                output = output.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, C, 1, 1]
                output = output.expand(B, -1, H, W)  # [B, C, H, W]
            
            # Now reshape for loss calculation
            B, C, H, W = output.shape
            output_flat = output.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            target_flat = target.reshape(-1)  # [B*H*W]
            
            cross_entropy = torch.nn.functional.cross_entropy(output_flat, target_flat)

            if class_specific:
                max_dist = (model.prototype_shape[1]
                           * model.prototype_shape[2]
                           * model.prototype_shape[3])

                # Handle label shape for prototype matching
                batch_labels = target_flat.unique()
                prototypes_of_correct_class = torch.zeros(len(batch_labels), model.prototype_shape[0]).to(device)
                for idx, lbl in enumerate(batch_labels):
                    prototypes_of_correct_class[idx] = model.prototype_class_identity[:, lbl]
                
                # Ensure min_distances has the right shape
                if len(min_distances.shape) == 2:  # If [B, num_prototypes]
                    min_distances = min_distances.unsqueeze(2).unsqueeze(3)  # [B, num_prototypes, 1, 1]
                    min_distances = min_distances.expand(-1, -1, H, W)  # [B, num_prototypes, H, W]
                
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class.unsqueeze(2).unsqueeze(3), dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class.unsqueeze(2).unsqueeze(3), dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class.unsqueeze(2).unsqueeze(3), dim=1) / torch.sum(prototypes_of_wrong_class, dim=1).unsqueeze(1).unsqueeze(2)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.prototype_class_identity).to(device)
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1)

            # For accuracy calculation
            _, predicted = torch.max(output, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        if is_train:
            if class_specific:
                loss = (coefs['crs_ent'] * cross_entropy + coefs['clst'] * cluster_cost
                       + coefs['sep'] * separation_cost + coefs['l1'] * l1) if coefs else (cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1)
            else:
                loss = (coefs['crs_ent'] * cross_entropy + coefs['clst'] * cluster_cost + coefs['l1'] * l1) if coefs else (cross_entropy + 0.8 * cluster_cost + 1e-4 * l1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input, target, output, predicted, min_distances

    end = time.time()
    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.last_layer.weight.norm(p=1).item()))

    return n_correct / n_examples

def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                         class_specific=class_specific, coefs=coefs, log=log)

def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                         class_specific=class_specific, log=log)

# Main training loop
if __name__ == '__main__':
    train_loader, val_loader = get_data_loaders(root_dir, batch_size=4)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, optimizer, class_specific=True, log=print)
        test(model, val_loader, class_specific=True, log=print)