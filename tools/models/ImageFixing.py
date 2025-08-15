import os
from skimage import img_as_ubyte
import torch
from utils.utils import imgpath2vec
import matplotlib.pyplot as plt
from runpy import run_path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def image_deblurring(input_path: str, output_path: str) -> str:
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
    weights = os.path.join('tools', 'models', 'Restormer','Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    load_arch = run_path(os.path.join('tools', 'models', 'Restormer','basicsr', 'models', 'archs', 'restormer_arch.py'))
    image_deblurring_model = load_arch['Restormer'](**parameters)
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    checkpoint = torch.load(weights)
    image_deblurring_model.load_state_dict(checkpoint['params'])
    image_deblurring_model.eval()

    cur = imgpath2vec(input_path)
    with torch.no_grad():

        h = cur.shape[1]
        w = cur.shape[2]
        img = cur.contiguous().view(h, w, 3)

        restored = self.image_deblurring_model(input_)

        restored = torch.clamp(restored, 0, 1)

        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()[:,:w//3,:h//3, ]
        restored = img_as_ubyte(restored[0])
        plt.imsave(output_path, restored)

        #print(restored)
        return restoreds


def image_denoising(input_path: str, output_path: str) -> str:
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
    weights = os.path.join('tools', 'models', 'Restormer','Denoising', 'pretrained_models', 'real_denoising.pth')
    parameters['LayerNorm_type'] =  'BiasFree'
    load_arch = run_path(os.path.join('tools', 'models', 'Restormer','basicsr', 'models', 'archs', 'restormer_arch.py'))
    image_denoising_model = load_arch['Restormer'](**parameters)
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    checkpoint = torch.load(weights)
    image_denoising_model.load_state_dict(checkpoint['params'])
    image_denoising_model.eval()


    cur = imgpath2vec(input_path)
    with torch.no_grad():

        h = cur.shape[1]
        w = cur.shape[2]
        img = cur.contiguous().view(h, w, 3)

        input_ = img.float().div(255.).permute(2,0,1).unsqueeze(0)


        restored = image_denoising_model(input_)

        restored = torch.clamp(restored, 0, 1)

        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()[:,:w//3,:h//3, ]
        restored = img_as_ubyte(restored[0])
        plt.imsave(output_path, restored)


        return output_path

if __name__ == '__main__':
    image_denoising("pic1.jpg", "denoising.jpg")
