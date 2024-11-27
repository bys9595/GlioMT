import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from typing import List


 
def minmax_norm(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img


class GenerateViTClinicalAttMap(nn.Module):
    def __init__(self):
        super(GenerateViTClinicalAttMap, self).__init__()
        
    def plot_and_save(self, net, original_image, clinical_feats, seg_slice, name, cfg, class_index=None, correct=True):
        self.attribution_generator = LRP_multi(net)

        transformer_attribution, clinic_cams = self.attribution_generator.generate_LRP(original_image, clinical_feats, index=class_index, start_layer=0)
        transformer_attribution = transformer_attribution.detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        
        
        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        fig = plt.figure(frameon=False, dpi=600)
        for i in range(5):
            ax = fig.add_subplot(2,5,i+1)
            ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
            ax.axis('off')
        
            ax = fig.add_subplot(2,5,i+6)
            ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
            ax.imshow(transformer_attribution, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        
        if correct:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'correct_AttMap')
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'wrong_AttMap')
            os.makedirs(save_path, exist_ok=True)
            
        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        plt.clf()
        

        age = clinical_feats[0]
        sex = clinical_feats[1]
        
        age = ['an old' if int(i)>=net.age_cutoff else 'a young' for i in age]

        age_prompt = ['a magnetic resonance image of ' + age[i] + ' patient' for i in range(len(age))]
        sex_prompt = ['a magnetic resonance image of a ' + sex[i] + ' patient' for i in range(len(sex))]
            
        age_token = net.tokenizer(age_prompt, return_tensors="pt", padding=True, truncation=True)
        sex_token = net.tokenizer(sex_prompt, return_tensors="pt")

        clini_tokens = [age_token, sex_token]
        
        clinic_dict = {'tokens' : [], 'clinic_cam' : []}
        for clinic_cam, clinic in zip(clinic_cams, clini_tokens):
            clinic_cam = clinic_cam.detach().cpu().squeeze()
            
            # normalize scores
            clinic_cam = (clinic_cam - clinic_cam.min()) / (clinic_cam.max() - clinic_cam.min())

            tokens = net.tokenizer.convert_ids_to_tokens(clinic['input_ids'].flatten())
            
            clinic_dict['tokens'].append(tokens)
            clinic_dict['clinic_cam'].append(clinic_cam)
            
        self.clinic_dict = clinic_dict
        
        # self.relevance_score = net.get_relevance_score()
        
    def return_clinic_dict(self):
        return self.clinic_dict
    
    def return_relevance_score(self, net):
        return net.get_relevance_score()
    
    def return_precision_score(self):
        return self.precision
   
    def generate_attribution(self, net, original_image, clinical_feats, class_index=None):
        self.attribution_generator = LRP_multi(net)

        transformer_attribution, clinic_cams = self.attribution_generator.generate_LRP(original_image, clinical_feats, index=class_index, start_layer=0)
        transformer_attribution = transformer_attribution.detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        
        return transformer_attribution
        
        
        

class LRP_multi:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, img, clinic, index=None, start_layer=0):
        output = self.model(img, clinic)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(img.device), start_layer=start_layer, **kwargs)