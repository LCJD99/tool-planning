import tools.models.colorization.colorizers as colorizers
from tools.models.colorization.colorizers import *
from utils.utils import imgpath2vec
import cv2
import matplotlib.pyplot as plt


def image_colorization(image_path: str, output_path: str ) -> str:
    """colorization image

    Args:
        image_path: Path to the image file
        output_path: Optional path to save the enhanced image directly

    Returns:
        Enhanced image path
    """
    # Get from registry or create and register if not exists
    model_instance = get_tool('image_colorizaion')
    # if model_instance is None:
    #     model_instance = ImageSuperResolutionModel()
    #     register_tool('image_super_resolution', model_instance)

    colorizer = colorizers.siggraph17().eval()
    img = load_img(image_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    output_image = postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())
    plt.imsave(output_path, output_image)

    return output_path


if __name__ == "__main__":
    image_colorization("./pic1.jpg", "colord1.jpg")
