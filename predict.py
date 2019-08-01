import torchvision
import torch
from PIL import Image
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to predict the class of an image using a pretrained neuronal network.')

    parser.add_argument(
        'input',
        type=str,
        help='Path to the image.')

    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to the pretrained model.')

    parser.add_argument(
        '--top_k',
        required=False,
        type=int,
        default=5,
        help='Return predictions for top k classes.')

    parser.add_argument(
        '--category_names',
        required=False,
        type=str,
        default="cat_to_name.json",
        help='Map predictions to class names.')

    parser.add_argument(
        '--gpu',
        required=False,
        type=str,
        default="gpu",
        help='Use gpu for inference.')

    args = parser.parse_args()

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255),
                                                      torchvision.transforms.CenterCrop(224),
                                                      torchvision.transforms.ToTensor(),
                                                      normalize])

    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # use test transformer for image preprocessing of pillow image
        # color channel is set to first dimension automatically
        image = test_transforms(image)

        return image


    def load_model(path):
        return torch.load(path)


    model_loaded = load_model(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu == "gpu" else "cpu")
    model_loaded.to(device)


    img_path = args.input
    img = Image.open(img_path)
    image_processed = process_image(img)


    def predict(image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        image = Image.open(image_path)
        image = process_image(image)

        image = image.to(device)
        image = image.unsqueeze(0)
        logps = model.forward(image)

        ps = torch.exp(logps)
        top_probs, top_classes = ps.topk(topk, dim=1)

        idx_to_class = {v: k for k, v in model.class_to_idx.items()}

        top_classes_string = []
        top_probs_list = top_probs.tolist()[0]

        for i, c in enumerate(top_classes.tolist()[0]):
            top_classes_string.append(idx_to_class[c])

        return top_classes_string, top_probs_list


    top_classes, top_probs = predict(img_path, model_loaded, args.top_k)
    real_image_label = img_path.split("/")[-2]

    if args.category_names is not None:

        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

        real_image_label = cat_to_name[real_image_label]
        top_classes_names = [cat_to_name[c] for c in top_classes]
        print("True label: {}".format(real_image_label))
        print("Predicted label: {}".format(top_classes_names[0]))

        if args.top_k == 1:
            print("Class Probability: {}".format(top_probs[0]))

        if args.top_k > 1:
            print("Class Probabilities for top {} classes:".format(args.top_k))
            for i,name in enumerate(top_classes_names):
                print("{} - label: {}, class probability {}".format(i+1, name, top_probs[i]))

    else:
        print("True label: {}".format(real_image_label))
        print("Predicted label: {}".format(top_classes[0]))

        if args.top_k == 1:
            print("Class Probability: {}".format(top_probs[0]))

        if args.top_k > 1:
            print("Class Probabilities for top {} classes:".format(args.top_k))
            for i, name in enumerate(top_classes):
                print("{} - label: {}, class probability {}".format(i + 1, name, top_probs[i]))



