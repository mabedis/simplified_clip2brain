"""The class of extraction the ConvNet features."""

import copy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image
import torchextractor as tx
from sklearn.decomposition import PCA
from torchvision import transforms, models

from src.utils import utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConvNetFeatureExtractor:
    """ConvNet Feature Extractor."""

    def __init__(self):
        self.stimuli_path = utils.GetNSD(section='DATA', entry='StimuliDir')
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def extract_resnet_pre_pca_feature(self) -> None:
        """Extract features from the ResNet prePCA."""
        layers = ["layer1", "layer2", "layer3", "layer4", "layer4.2.relu"]
        # model, self.preprocess = clip.load("RN50", device=device)
        model = models.resnet50(pretrained=True)
        model = tx.Extractor(model, layers)
        compressed_features = [copy.copy(e)
                               for _ in range(len(layers)) for e in [[]]]
        subsampling_size = 5000

        print("Extracting ResNet features")
        for cid in tqdm(all_coco_ids):
            with torch.no_grad():
                image_path = f"{self.stimuli_path}{cid}.jpg"
                image = self.preprocess(Image.open(
                    image_path)).unsqueeze(0).to(DEVICE)

                _, features = model(image)

                for i, f in enumerate(features.values()):
                    # print(f.size())
                    if len(f.size()) > 3:
                        c = f.data.shape[1]  # number of channels
                        k = int(np.floor(np.sqrt(subsampling_size / c)))
                        tmp = nn.functional.adaptive_avg_pool2d(f.data, (k, k))
                        # print(tmp.size())
                        compressed_features[i].append(
                            tmp.squeeze().cpu().numpy().flatten())
                    else:
                        compressed_features[i].append(
                            f.squeeze().data.cpu().numpy().flatten()
                        )

        for l, f in enumerate(compressed_features):
            np.save(f"{feature_output_dir}/convnet_resnet_prePCA_{l:02}.npy", f)

    def extract_visual_resnet_feature(self) -> None:
        """Extract features from the visual ResNet."""
        for l in range(7):
            try:
                f = np.load(
                    f"{feature_output_dir}/convnet_resnet_prePCA_{l:02}.npy")
            except FileNotFoundError:
                self.extract_resnet_pre_pca_feature()
                f = np.load(
                    f"{feature_output_dir}/convnet_resnet_prePCA_{l:02}.npy")

            print("Running PCA")
            print("feature shape: ")
            print(f.shape)
            pca = PCA(n_components=min(f.shape[0], 64), svd_solver="auto")

            fp = pca.fit_transform(f)
            print(f"Feature {l:02} has shape of: ")
            print(fp.shape)

            np.save(f"{feature_output_dir}/resnet_{l:02}.npy", fp)

    def extract_resnet_last_layer_feature(self, cid=None, saving=True) -> np.ndarray:
        """Extract features from the ResNet's last layer.

        Args:
            cid (_type_, optional): _description_. Defaults to None.
            saving (bool, optional): Save the results. Defaults to True.

        Returns:
            np.ndarray: _description_
        """
        model = models.resnet50(pretrained=True).to(DEVICE)
        model = tx.Extractor(model, "avgpool")

        if cid is None:  # Extracting ResNet features
            output = []
            for cid in tqdm(all_coco_ids):
                with torch.no_grad():
                    image_path = f"{self.stimuli_path}{cid}.jpg"
                    image = self.preprocess(
                        Image.open(image_path)
                    ).unsqueeze(0).to(DEVICE)

                    _, features = model(image)
                    output.append(
                        features["avgpool"].squeeze(
                        ).data.cpu().numpy().flatten()
                    )
        else:
            with torch.no_grad():
                image_path = f"{self.stimuli_path}{cid}.jpg"
                image = self.preprocess(Image.open(
                    image_path)).unsqueeze(0).to(DEVICE)
                _, features = model(image)
                output = features["avgpool"].data.squeeze().cpu()
        if saving:
            np.save(f"{feature_output_dir}/convnet_resnet_avgpool.npy", output)

        return output


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--subj", default=1, type=int)
#     parser.add_argument(
#         "--feature_dir",
#         type=str,
#         default="features",
#     )
#     parser.add_argument(
#         "--project_output_dir",
#         type=str,
#         default="output",
#     )
#     args = parser.parse_args()
#     feature_output_dir = "%s/subj%01d" % (args.feature_dir, args.subj)
#     all_coco_ids = np.load(
#         "%s/coco_ID_of_repeats_subj%02d.npy" % (
#             args.project_output_dir, args.subj)
#     )

#     extract_resnet_last_layer_feature()
