import torch
from tqdm import tqdm


def dice_coeff(input, target):
    """Dice coeff for individual examples"""
    eps = 0.0001
    intersection = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * intersection.float() + eps) / union.float()
    return t


def batch_dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + dice_coeff(c[0], c[1])

    return s / (i + 1)

def eval_dice(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    mask_tot = 0
    boundary_tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks_and_boundaries = batch['transformed_image'], batch['transformed_mask_and_boundary']
            true_masks = true_masks_and_boundaries[:,0,:,:]
            true_boundaries = true_masks_and_boundaries[:,1,:,:]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            true_boundaries = true_boundaries.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                preds_mask_and_boundary = net(imgs)


            mask_preds = (torch.sigmoid(preds_mask_and_boundary[:, 0, :, :]) > 0.5).float()
            boundary_preds = (torch.sigmoid(preds_mask_and_boundary[:, 1, :, :]) > 0.5).float()

            mask_tot += batch_dice_coeff(mask_preds, true_masks).item()
            boundary_tot += batch_dice_coeff(boundary_preds, true_boundaries).item()

        pbar.update()

    net.train()
    return mask_tot / n_val, boundary_tot / n_val
