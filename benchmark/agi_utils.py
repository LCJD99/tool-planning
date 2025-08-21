from evaluate import load
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import re
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

def text2picpath(text: str) -> str:
    regex = r'([\'"]?)(/tmp/[^\'"]+\.(?:jpg|png))\1'
    match = re.search(regex , text)
    if match:
        return match.group(2)
    else:
        return ""

def txt_eval(predictions, references, bertscore, device="cuda"):
    score = bertscore.compute(
                    predictions=predictions,
                    references=references,
                    lang="en",
                    model_type="microsoft/deberta-xlarge-mnli",
                    device=device)["f1"]

    return score


def txt_loader(path):
    text = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            text.append(line)
    f.close()
    return text

def path2PIL(paths):
    imgs = []
    img = Image.open(paths[0])
    img = img.convert("RGB")
    imgs.append(img)
    return imgs

def img2vec(img):
    img_transform = transforms.Compose([ transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.PILToTensor(),
                                        ])
    emb = img_transform(img)
    return emb



def image_similarity(im_path1, im_path2, model, extractor):
    im1 = path2PIL(im_path1)
    im2 = path2PIL(im_path2)

    batch_size = len(im1)
    # Load two images
    img1 = extractor(im1, return_tensors="pt")
    img2 = extractor(im2, return_tensors="pt")

    # Preprocess the images and get their embeddings
    with torch.no_grad():
        emb1 = model(img1.pixel_values)[0].squeeze().numpy()
        emb2 = model(img2.pixel_values)[0].squeeze().numpy()

    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)

    sims = np.diag(cosine_similarity(emb1, emb2))

    avg_similarity = np.mean(sims)

    return avg_similarity

def module_seq_filter(module_seq, task_id):
    io_dict = {
                "Colorization":['image','image'],
                "Image Denoising":['image','image'],
                "Image Deblurring":['image','image'],
                "Image Super Resolution":['image','image'],
                "Image Classification":['image','text'],
                "Image Captioning":['image','text'],
                "Object Detection":['image','text'],
                "Text Summarization":['text','text'],
                "Text Generation":['text','text'],
                "Machine Translation":['text','text'],
                "Fill Mask":['text','text'],
                "Sentiment Analysis":['text','text'],
                "Text to Image Generation":['text','image'],
                "Question Answering":['text-text','text'],
                "Visual Question Answering":['image-text','text']
        }
    module_seq_list = module_seq.split(", ")
    input_type = io_dict[module_seq_list[0]][0]
    output_type = io_dict[module_seq_list[-1]][1]
    if input_type == "image" and output_type == "image" and 0<=task_id<=14:
        return True
    elif input_type == "image" and output_type == "text" and 15<=task_id<=104:
        return True
    elif input_type == "text" and output_type == "image" and 105<=task_id<=107:
        return True
    elif input_type == "text" and output_type == "text" and 108<=task_id<=125:
        return True
    elif input_type == "image-text" and output_type == "text" and 126<=task_id<=170:
        return True
    elif input_type == "text-text" and output_type == "text" and 171<=task_id<=188:
        return True
    else:
        return False



def whole_module_seq_filter(module_seq, task_id):
    io_dict = {
                "Colorization":['image','image'],
                "Image Denoising":['image','image'],
                "Image Deblurring":['image','image'],
                "Image Super Resolution":['image','image'],
                "Image Classification":['image','text'],
                "Image Captioning":['image','text'],
                "Object Detection":['image','text'],
                "Text Summarization":['text','text'],
                "Text Generation":['text','text'],
                "Machine Translation":['text','text'],
                "Fill Mask":['text','text'],
                "Sentiment Analysis":['text','text'],
                "Text to Image Generation":['text','image'],
                "Question Answering":['text-text','text'],
                "Visual Question Answering":['image-text','text']
        }
    module_seq_list = module_seq.split(", ")
    condition_1 = None
    for i, m in enumerate(module_seq_list):
        if i < len(module_seq_list)-1 and io_dict[m][1] != io_dict[module_seq_list[i+1]][0]:
            condition_1 = False
            break
        else:
            condition_1 = True


    condition_2 = None
    input_type = io_dict[module_seq_list[0]][0]
    output_type = io_dict[module_seq_list[-1]][1]
    if input_type == "image" and output_type == "image" and 0<=task_id<=14:
        condition_2 = True
    elif input_type == "image" and output_type == "text" and 15<=task_id<=104:
        condition_2 = True
    elif input_type == "text" and output_type == "image" and 105<=task_id<=107:
        condition_2 = True
    elif input_type == "text" and output_type == "text" and 108<=task_id<=125:
        condition_2 = True
    elif input_type == "image-text" and output_type == "text" and 126<=task_id<=170:
        condition_2 = True
    elif input_type == "text-text" and output_type == "text" and 171<=task_id<=188:
        condition_2 = True
    else:
        condition_2 = False

    return condition_1 and condition_2



def match_module_seq(model_steps, sentence_model):
    module_seq = ""

    for i in range(len(model_steps)):

        sentences1 = [model_steps[i]]*15

        sentences2 = ["Image Classification","Colorization","Object Detection",\
                  "Image Super Resolution","Image Captioning","Image Deblurring",\
                  "Image Denoising","Text to Image Generation","Visual Question Answering",\
                  "Sentiment Analysis","Question Answering","Text Summarization",\
                  "Text Generation","Machine Translation","Fill Mask"]

        #Compute embedding for both lists
        embeddings1 = sentence_model.encode(sentences1, convert_to_tensor=True)#.to(device_)
        embeddings2 = sentence_model.encode(sentences2, convert_to_tensor=True)#.to(device_)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        similarities = torch.stack([cosine_scores[i][i] for i in range(15)])

        module_index = torch.argmax(similarities).item()
        module_seq += sentences2[module_index] + ", "
        # print(similarities[module_index])
        # print(sentences2[module_index])

    #Output the pairs with their score
    # for i in range(len(sentences1)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    module_seq = module_seq.strip()[:-1]
    return module_seq

if __name__ == '__main__':
    from transformers import AutoModel, AutoFeatureExtractor
    vit_ckpt = "nateraw/vit-base-beans"
    vit = AutoModel.from_pretrained(vit_ckpt)
    vit.eval()
    vit_extractor = AutoFeatureExtractor.from_pretrained(vit_ckpt)
    dist = image_similarity(['pic1.jpg'], ['enhanced_image.jpg'], vit, vit_extractor)
    print(dist)

