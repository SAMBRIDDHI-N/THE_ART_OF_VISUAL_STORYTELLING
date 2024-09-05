import os
import torch
import openai
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from openai import OpenAI
from transformers import DetrImageProcessor, DetrForObjectDetection
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def getImages(path):
    return [Image.open(os.path.join(image_folder_path, path)) for path in os.listdir(image_folder_path)]

def getPreprocessor(preprocessor):
    return DetrImageProcessor.from_pretrained(preprocessor, revision="no_timm")

def getModel(model):
    return DetrForObjectDetection.from_pretrained(model, revision="no_timm")

def getModelOutputs(processor, model, images):
    inputs = [processor(images=image, return_tensors="pt") for image in images]
    return [model(**pre_input) for pre_input in inputs]

def getResults(processor, images, outputs):
    target_sizes = [torch.tensor([image.size[::-1]]) for image in images]
    return [processor.post_process_object_detection(output, target_sizes=target_size, threshold=0.9)[0] for output, target_size in zip(outputs, target_sizes)]

def story_for_each_sentence(string):
    print(string)
    responce = client.chat.completions.create(
		                model = 'gpt-3.5-turbo',
		                messages = [
		                                {
                                            'role' : 'user',
                                            'content' : f"Write a short story of a scene depicted in an image having {string}"
                                        }
		                            ],
		                max_tokens=100,
		                temperature = 0.2
	            )
    print(f"Generated Story:", {responce.choices[0].message.content}, sep = "\n")
 
def story_for_all_images(string):
    prompt = ""
    for idx, s in enumerate(string):
        prompt += f"Image {idx + 1} contains {s},"
    responce = client.chat.completions.create(
                                                model = 'gpt-3.5-turbo',
	                                            messages = [
		                                                        {
                                                                    'role' : 'user', 'content' : f"Write a short story of a scene depicted in the images having {prompt}"
                                                                }
		                                                    ],
		                                                    max_tokens=300,
		                                                    temperature = 0.2
	            )
    print(f"Generated Story:", {responce.choices[0].message.content}, sep = "\n")
 

def plot_boxes(idx, image, boxes, scores, labels, class_map):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        class_name = class_map[label.item()]

        box = [round(i, 2) for i in box.tolist()]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax.text(box[0], box[1], f"{class_name} {round(score.item(), 3)}", color='r')

    plt.axis('off')
    plt.savefig(f"result{idx}.jpg")
    
def generate_story(results_list):
    all_sentences = []
    for idx, results in enumerate(results_list):
        print("\nImage", idx + 1, end = "\n\n")
        counts = {}
        for value in list(results["labels"].numpy()):
            if model.config.id2label[value] not in counts:
                counts[model.config.id2label[value]] = list(results["labels"].numpy()).count(value)
        string = f"Detected "
        for key, value in counts.items():
            string += f"{value} {key} "
        # print(string, end = "\n\n")
        all_sentences.append(string)
        story_for_each_sentence(string)
    return all_sentences

# Enter name of the image folder. e.g., images
image_folder_path = input("Enter images folder name:")

# Getting images
images = getImages(image_folder_path)

# Loading preprocessor and model
processor = getPreprocessor("facebook/detr-resnet-50")
model = getModel("facebook/detr-resnet-50")

# Extracting model outputs
outputs = getModelOutputs(processor, model, images)

# Extracting results form model output
results_list = getResults(processor, images, outputs)

# Generating stories for each image
all_sentences = generate_story(results_list)

# Generating story by combining all images
print("\nStory Combining all images\n")
story_for_all_images(all_sentences)

# Drawing bounding box for all images
for idx, image, results in zip([i for i in range(len(images))], images, results_list):
    plot_boxes(idx, image, results["boxes"], results["scores"], results["labels"], model.config.id2label)
