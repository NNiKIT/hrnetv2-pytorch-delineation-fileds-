import argparse
import logging
import os

import torch
from PIL import Image
import tifffile
import numpy as np

from core.dataset import CustomDataset, InferenceAugmentor
from core.hrnetv2 import HRNetv2


results_dir = 'results/'
test_imgs_dir = 'test_imgs/'

def predict_img(net,
                img,
                sizes_list,
                device,
                out_threshold=0.5):
    net.eval()
    
    augmentations_list = ['resize', 'v_flip', 'h_flip']

    # original image
    img_tensor = torch.from_numpy(CustomDataset.preprocess_to_inference(img))
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    # input tensors for [N*resized_imgs, v_flipped_img, h_flipped_img]
    all_inputs_dict = InferenceAugmentor.return_augmented(img_tensor, augmentations_list, sizes_list)
    
    # output tensors for [img, N*resized_imgs, v_flipped_img, h_flipped_img]
    all_outputs_dict = {}
    for augmentation_key, augmented_img in all_inputs_dict.items():
        augmented_img = augmented_img.unsqueeze(0)
        augmented_img = augmented_img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            preds_mask_and_boundary = net(augmented_img)

            preds_mask_and_boundary = preds_mask_and_boundary.squeeze(0)

            mask = torch.sigmoid(preds_mask_and_boundary[0,:,:])
            boundary = torch.sigmoid(preds_mask_and_boundary[1,:,:])

            mask_minus_borders = torch.subtract(mask, boundary)

            all_outputs_dict[augmentation_key] = mask_minus_borders
    
    reverse_augmented_outputs_list = InferenceAugmentor.return_reverse_augmented(all_outputs_dict, sizes_list)
    
    normal_ouputs = []
    # result of inference on original img
    normal_ouputs.append(img_tensor)
    for normal_ouput in reverse_augmented_outputs_list:     
        normal_ouput = normal_ouput.unsqueeze(0).to(device=device, dtype=torch.float32)
        normal_ouputs.append(normal_ouput)

    concat_result = torch.cat(normal_ouputs, dim=0)
    final_mask = torch.mean(concat_result, dim=0) > out_threshold
    final_mask = final_mask.squeeze(0)
    final_mask = final_mask.squeeze().cpu().numpy()
    return final_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--predict_type', '-type', default='single',
                        metavar='predict type',
                        help="Specify predict type: single or batch, if batch: wole test_imgs dir is processed")
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def np_mask_to_image(np_mask):
    return Image.fromarray((np_mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(results_dir):
            try:
                os.mkdir(results_dir)
                logging.info('Created results directory')
            except OSError:
                pass

    net = HRNetv2(img_channels=4)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device)['model_state_dict'])

    logging.info("Model loaded !")

    sizes_list = []
    i = 224
    sizes_list.append(str(i))
    while i < 448:
        i += 32
        sizes_list.append(str(i))

    # python predict.py -type single -m checkpoints/pytorch_hrnetv2-1div2-size_vnir+borders+new_aug+dicebce-224x224_epoch-50.pth -i test_imgs/test1.tiff -t 0.2
    if args.predict_type == 'single':
        in_file = args.input

        img = tifffile.imread(in_file[0])

        mask = predict_img(net=net,
                        img=img,
                        sizes_list=sizes_list,
                        out_threshold=args.mask_threshold,
                        device=device)

        mask_result = np_mask_to_image(mask)
        mask_result.save(results_dir + 'mask_' + in_file[0].split('/')[-1].split('.tiff')[0] + '.png')

        logging.info("Mask predictions saved to {} diectory".format(results_dir))
    # python predict.py -type batch -m checkpoints/pytorch_hrnetv2-1div2-size_vnir+borders+new_aug+dicebce-224x224_epoch-50.pth -t 0.2
    else:
        if args.predict_type == 'batch':
            for tiff_img in os.listdir(test_imgs_dir):
                if tiff_img.endswith('.tiff'):
                    in_file = test_imgs_dir + tiff_img
                    img = tifffile.imread(in_file)

                    mask = predict_img(net=net,
                                    img=img,
                                    sizes_list=sizes_list,
                                    out_threshold=args.mask_threshold,
                                    device=device)

                    mask_result = np_mask_to_image(mask)
                    mask_result.save(results_dir + 'mask_' + in_file.split('/')[-1].split('.tiff')[0] + '.png')

                    logging.info("Mask predictions saved to {} diectory".format(results_dir))
