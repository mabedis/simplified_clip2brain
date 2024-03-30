'''Extract image features using CLIP.'''

import os
import copy

import clip
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchextractor as tx
from sklearn.decomposition import PCA
from transformers import BertTokenizer, VisualBertModel

from src.utils import utils
from src.utils.coco_utils import COCOUtils
from src.utils.data_util import (
    load_top1_objects_in_coco, load_objects_in_coco)
from src.features.extract_convnet_features import (
    ConvNetFeatureExtractor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ExtractFeatures:
    """Extract Clip Features."""

    def __init__(self, subject: int, output_dir: str, feature_dir: str):
        self.subject = subject
        self.output_dir = output_dir
        self.feature_dir = os.path.join(feature_dir, 'output/features')

        utils.Directory(path=os.path.join(
            feature_dir, 'output/features')).check_dir_existence()
        self.stimuli_path = utils.GetNSD(
            section='DATA', entry='StimuliDir').get_dataset_path()
        self.coco = COCOUtils()
        self.expand_dict = {
            'person': ["man", "men", "women", "woman", "people", "guys"]
        }

        if self.subject == 0:
            for subj in range(8):
                print(f"Extracting subj{subj+1:02}")
                self.feature_output_dir = f"{self.feature_dir}/subj{subj+1:02}"
                utils.Directory(
                    path=self.feature_output_dir).check_dir_existence()
                self.all_coco_ids = np.load(
                    os.path.join(
                        self.output_dir,
                        f"coco_ID_of_repeats_subj{subj+1:02}.npy",
                    )
                )
                try:
                    np.load(
                        os.path.join(
                            self.feature_dir,
                            "clip_text.npy",
                        )
                    )
                except FileNotFoundError:
                    self.text_feat = self.extract_last_layer_feature(
                        modality="text"
                    )
                    np.save(
                        os.path.join(
                            self.feature_dir,
                            "clip_text.npy",
                        ),
                        self.text_feat
                    )

                try:
                    np.load(
                        os.path.join(
                            self.feature_dir,
                            "clip_visual_resnet.npy",
                        )
                    )
                except FileNotFoundError:
                    self.visual_res_feat = self.extract_last_layer_feature(
                        model_name="RN50",
                        modality="vision",
                    )
                    np.save(
                        os.path.join(
                            self.feature_dir,
                            "clip_visual_resnet.npy",
                        ),
                        self.visual_res_feat,
                    )
        else:
            # self.all_coco_ids = np.load(os.path.join(
            #     self.output_dir,
            #     f"output/coco_ID_of_repeats_subj{self.subject:02}.npy"
            # ))
            self.all_coco_ids = np.array([391895, 522418, 550221, 554625])
            self.feature_output_dir = os.path.join(
                self.feature_dir,
                f"subj{self.subject:02}",
            )
            utils.Directory(path=self.feature_output_dir).check_dir_existence()
            self.visual_res_feat = self.extract_last_layer_feature(
                model_name="RN50",
                modality="vision",
            )
            np.save(
                os.path.join(
                    self.feature_dir,
                    "clip_visual_resnet.npy",
                ),
                self.visual_res_feat
            )

            text_feat = self.extract_last_layer_feature(modality="text")
            np.save(
                os.path.join(
                    self.feature_dir,
                    "clip_text.npy",
                ),
                text_feat
            )

            text_feat = self.extract_last_layer_feature(modality="vision")
            np.save(
                os.path.join(
                    self.feature_dir,
                    "clip.npy",
                ),
                text_feat
            )

            self.extract_visual_resnet_feature()

    def load_object_caption_overlap(self, cid: int) -> list:
        """Load object's caption overlap.

        Args:
            cid (int): Caption ID.

        Returns:
            list: _description_
        """
        caption = self.coco.load_captions(cid)
        objs = load_objects_in_coco(cid)
        for k, v in self.expand_dict.items():
            if k in objs:
                objs += v
        all_caps: str = ""
        for c in caption:
            all_caps += c
        obj_intersect = [o for o in objs if o in all_caps]

        return obj_intersect

    def extract_object_base_text_feature(self) -> None:
        """Extract object base text feature."""
        model, _ = clip.load("ViT-B/32", device=DEVICE)
        all_features = []
        for cid in tqdm(self.all_coco_ids):
            with torch.no_grad():
                objects = load_objects_in_coco(cid)
                expression = "a photo of " + " ".join(objects)
                text = clip.tokenize(expression).to(DEVICE)
                cap_emb = model.encode_text(text).cpu().data.numpy()
                all_features.append(cap_emb)

        all_features = np.array(all_features).squeeze()
        print("Feature shape is: " + str(all_features.shape))
        np.save(
            os.path.join(
                self.feature_output_dir,
                "clip_object.npy",
            ),
            all_features,
        )

    def extract_top1_obejct_base_text_feature(self) -> None:
        """Extract top1 obejct base text feature."""
        model, _ = clip.load("ViT-B/32", device=DEVICE)
        all_features = []
        for cid in tqdm(self.all_coco_ids):
            with torch.no_grad():
                obj = load_top1_objects_in_coco(cid)
                text = clip.tokenize("a photo of a " + obj).to(DEVICE)
                cap_emb = model.encode_text(text).cpu().data.numpy()
                all_features.append(cap_emb)

        all_features = np.array(all_features).squeeze()
        print("Feature shape is: " + str(all_features.shape))
        np.save(
            os.path.join(
                self.feature_output_dir,
                "clip_top1_object.npy",
            ),
            all_features,
        )

    def extract_obj_cap_intersect_text_feature(self) -> None:
        """Extract object's caption intersection with text feature."""
        model, _ = clip.load("ViT-B/32", device=DEVICE)
        all_features = []
        for cid in tqdm(self.all_coco_ids):
            with torch.no_grad():
                overlaps = self.load_object_caption_overlap(cid)
                text = clip.tokenize(overlaps).to(DEVICE)
                cap_emb = model.encode_text(text).cpu().data.numpy()
                all_features.append(cap_emb)

        all_features = np.array(all_features)
        print("Feature shape is: " + str(all_features.shape))
        np.save(
            os.path.join(
                self.feature_output_dir,
                "clip_object_caption_overlap.npy",
            ),
            all_features,
        )

    def extract_visual_resnet_pre_pca_feature(self) -> None:
        """Extract visual ResNet prePCA feature."""
        LOI_ResNet_vision = [
            "visual.bn1",
            "visual.avgpool",
            "visual.layer1.2.bn3",
            "visual.layer2.3.bn3",
            "visual.layer3.5.bn3",
            "visual.layer4.2.bn3",
            "visual.attnpool",
        ]
        model, preprocess = clip.load("RN50", device=DEVICE)
        model_visual = tx.Extractor(model, LOI_ResNet_vision)
        compressed_features = [copy.copy(e) for _ in range(8) for e in [[]]]
        subsampling_size = 5000

        print("Extracting ResNet features")
        for cid in tqdm(self.all_coco_ids):
            with torch.no_grad():
                image_path = f"{self.stimuli_path}/{cid}.jpg"
                image = preprocess(
                    Image.open(image_path)
                ).unsqueeze(0).to(DEVICE)
                captions = self.coco.load_captions(cid)
                text = clip.tokenize(captions).to(DEVICE)

                _, features = model_visual(image, text)

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
            np.save(
                os.path.join(
                    self.feature_output_dir,
                    f"visual_layer_resnet_prePCA_{l:02}.npy",
                ),
                f
            )

    def extract_visual_resnet_feature(self) -> None:
        """Extract visual ResNet feature."""
        for l in range(7):
            try:
                f = np.load(
                    os.path.join(
                        self.feature_output_dir,
                        f"visual_layer_resnet_prePCA_{l:02}.npy",
                    )
                )
            except FileNotFoundError:
                self.extract_visual_resnet_pre_pca_feature()
                f = np.load(
                    os.path.join(
                        self.feature_output_dir,
                        f"visual_layer_resnet_prePCA_{l:02}.npy",
                    )
                )

            print("Running PCA")
            print("Feature Shape: ")
            print(f.shape)
            pca = PCA(n_components=min(f.shape[0], 64), svd_solver="auto")

            fp = pca.fit_transform(f)
            print(f"Feature {l:02} has shape of:")
            print(fp.shape)

            np.save(
                os.path.join(
                    self.feature_output_dir,
                    f"visual_layer_resnet_{l:02}.npy",
                ),
                fp
            )

    def extract_visual_transformer_feature(self) -> None:
        """Extract visual transformer feature."""

        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        LOI_transformer_vision = [
            f"visual.transformer.resblocks.{i:02}.ln_2" for i in range(12)
        ]
        model_visual = tx.Extractor(model, LOI_transformer_vision)
        compressed_features = [copy.copy(e) for _ in range(12) for e in [[]]]

        for cid in tqdm(self.all_coco_ids):
            with torch.no_grad():
                image_path = f"{self.stimuli_path}/{cid}.jpg"
                image = preprocess(Image.open(image_path)
                                   ).unsqueeze(0).to(DEVICE)
                captions = self.coco.load_captions(cid)
                text = clip.tokenize(captions).to(DEVICE)

                _, features = model_visual(image, text)

                for i, f in enumerate(features.values()):
                    compressed_features[i].append(
                        f.squeeze().cpu().data.numpy().flatten())

        compressed_features = np.array(compressed_features)

        for l, f in enumerate(compressed_features):
            pca = PCA(n_components=min(
                f.shape[0], 64), whiten=True, svd_solver="full")
            try:
                fp = pca.fit_transform(f)
            except ValueError:
                print(fp.shape)

            print(f"Feature {l:02} has shape of: ")
            print(fp.shape)

            np.save(
                os.path.join(
                    self.feature_output_dir,
                    f"visual_layer_{l:02}.npy",
                ),
                fp
            )

    def extract_text_layer_feature(self) -> None:
        """Extract text layer feature."""
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        LOI_text = [f"transformer.resblocks.{i:02}.ln_2" for i in range(12)]
        text_features = [copy.copy(e) for _ in range(12) for e in [[]]]
        model_text = tx.Extractor(model, LOI_text)
        for cid in tqdm(self.all_coco_ids):
            with torch.no_grad():
                image_path = f"{self.stimuli_path}/{cid}.jpg"
                image = preprocess(Image.open(image_path)
                                   ).unsqueeze(0).to(DEVICE)

                captions = self.coco.load_captions(cid)

                layer_features = [
                    copy.copy(e) for _ in range(12) for e in [[]]
                ]  # layer_features is 12 x 5 x m
                for caption in captions:
                    text = clip.tokenize(caption).to(DEVICE)
                    _, features = model_text(image, text)
                    # features is a feature dictionary for all layers, each image, each caption
                    for i, layer in enumerate(LOI_text):
                        layer_features[i].append(
                            features[layer].cpu().data.numpy(
                            ).squeeze().flatten()
                        )

                    # print(np.array(layer_features).shape)
                avg_features = np.mean(
                    np.array(layer_features), axis=1)  # 12 x m

            for i in range(len(LOI_text)):
                text_features[i].append(avg_features[i])

        text_features = np.array(text_features)
        # print(text_features.shape) # 12 x 10000 x m

        for l, f in enumerate(text_features):
            pca = PCA(n_components=min(
                f.shape[0], 64), whiten=True, svd_solver="full")
            try:
                fp = pca.fit_transform(f)
            except ValueError:
                print(fp.shape)

            print(f"Feature {l:02} has shape of:")
            print(fp.shape)

            np.save(
                os.path.join(
                    self.feature_output_dir,
                    "text_layer_{l:02}.npy",
                ),
                fp
            )

    def extract_last_layer_feature(
        self,
        model_name: str = "ViT-B/32",
        modality: str = "vision",
    ) -> np.array:
        """Extract last layer feature.

        Args:
            model_name (str, optional): Model to do pre-process. Defaults to "ViT-B/32".
            modality (str, optional): Type of modal. Defaults to "vision".

        Returns:
            np.array: All the features in np.array form.
        """
        all_images_paths = [
            f"{self.stimuli_path}/{_id}.jpg" for _id in self.all_coco_ids
        ]
        print(f"Number of Images: {len(all_images_paths)}")
        model, preprocess = clip.load(model_name, device=DEVICE)

        if modality == "vision":
            all_features = []
            for p in tqdm(all_images_paths):
                image = preprocess(Image.open(p)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                all_features.append(image_features.cpu().data.numpy())

            return np.array(all_features)

        if modality == "text":  # this is subject specific
            # extract text feature of image titles
            all_text_features = []
            for cid in tqdm(self.all_coco_ids):
                with torch.no_grad():
                    captions = self.coco.load_captions(cid)
                    # print(captions)
                    embs = []
                    for caption in captions:
                        text = clip.tokenize(caption).to(DEVICE)
                        embs.append(model.encode_text(text).cpu().data.numpy())

                    mean_emb = np.mean(np.array(embs), axis=0).squeeze()
                    all_text_features.append(mean_emb)

            all_text_features = np.array(all_text_features)
            print(all_text_features.shape)
            return all_text_features

    def extract_vibert_feature(self) -> list:
        """Extract vibert feature.

        Returns:
            list: List of all the features read in np.array form.
        """
        all_images_paths = []
        print(f"Number of Images: {len(all_images_paths)}")
        # model, preprocess = clip.load(model_name, device=DEVICE)
        model = VisualBertModel.from_pretrained(
            "uclanlp/visualbert-vqa-coco-pre")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # this is a custom function that returns the visual embeddings given the image path

        all_features = []
        for cid in tqdm(self.all_coco_ids):
            with torch.no_grad():
                captions = self.coco.load_captions(cid)
                inputs = tokenizer(
                    captions[0], return_tensors="pt"
                )  # take in the first caption in COCO
                feature_extractor = ConvNetFeatureExtractor()
                visual_embeds = feature_extractor.extract_resnet_last_layer_feature(
                    cid=cid, saving=False
                ).unsqueeze(0)
                # print("shape -1")
                # print(visual_embeds.shape)

                # pylint: disable=E1101
                visual_token_type_ids = torch.ones(
                    visual_embeds.shape, dtype=torch.long)
                visual_attention_mask = torch.ones(
                    visual_embeds.shape, dtype=torch.float)
                # print(visual_attention_mask.shape)
                # pylint: enable=E1101

                inputs.update(
                    {
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                    }
                )
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                # print(last_hidden_state.shape)

                all_features.append(
                    last_hidden_state.squeeze().data.cpu().numpy())

            np.save(f"{self.feature_dir}/vibert.npy", all_features)

        return np.array(all_features)
