import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from models.vit_utils.ViT_explanation_generator import LRP

 
def minmax_norm(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img


class GenerateViTAttMap(nn.Module):
    def __init__(self):
        super(GenerateViTAttMap, self).__init__()
        
    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None, correct=True):
        self.attribution_generator = LRP(net)

        transformer_attribution = self.attribution_generator.generate_LRP(original_image, method="transformer_attribution", index=class_index, start_layer=0).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        # transformer_attribution = attribution_generator.generate_LRP(original_image, method="full", index=class_index, start_layer=0).detach()[0].data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        
        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(transformer_attribution, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
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


class GenerateSwinTAttMap(nn.Module):
    def __init__(self):
        super(GenerateSwinTAttMap, self).__init__()
        
    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None, correct=True):
        self.attribution_generator = LRP(net)

        transformer_attributions = self.attribution_generator.generate_LRP(original_image, method="transformer_attribution", index=class_index, start_layer=0)
        
        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        new_atts = []
        for transformer_attribution in transformer_attributions:
            
            transformer_attribution = transformer_attribution.detach().unsqueeze(0).unsqueeze(0)
            transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, size=(224, 224), mode='bilinear')
            transformer_attribution = transformer_attribution.reshape(224, 224)
            
            # transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
            
            new_atts.append(transformer_attribution)
        
        # new_atts = torch.stack(new_atts, 0)
        # transformer_attribution = torch.sum(new_atts, dim=0).data.cpu().numpy()
        transformer_attribution = new_atts[-1].data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        
        
        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(transformer_attribution, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
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


from captum.attr import visualization
from selenium import webdriver

class GenerateViTClinicalAttMap(nn.Module):
    def __init__(self):
        super(GenerateViTClinicalAttMap, self).__init__()
        
    def plot_and_save(self, net, original_image, clinical_feats, seg_slice, name, cfg, class_index=None, correct=True):
        self.attribution_generator = LRP_multi(net)

        transformer_attribution, clinic_cams = self.attribution_generator.generate_LRP(original_image, clinical_feats, method="transformer_attribution", index=class_index, start_layer=0)
        transformer_attribution = transformer_attribution.detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        # transformer_attribution = attribution_generator.generate_LRP(original_image, method="full", index=class_index, start_layer=0).detach()[0].data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        
        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(transformer_attribution, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
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
        
        # -------------------------------------------------------------------------------------------------------
        age = clinical_feats[0]
        sex = clinical_feats[1]
        
        age = ['an old' if int(i)>=net.age_cutoff else 'a young' for i in age]

        # age_prompt = ['an mri of ' + age[i] + ' patient' for i in range(len(age))]
        # sex_prompt = ['an mri of a ' + sex[i] + ' patient' for i in range(len(sex))]
        age_prompt = ['a magnetic resonance image of ' + age[i] + ' patient' for i in range(len(age))]
        sex_prompt = ['a magnetic resonance image of a ' + sex[i] + ' patient' for i in range(len(sex))]
            
        age_token = net.tokenizer(age_prompt, return_tensors="pt", padding=True, truncation=True)
        sex_token = net.tokenizer(sex_prompt, return_tensors="pt")

        clini_tokens = [age_token, sex_token]
        
        clinic_dict = {'tokens' : [], 'clinic_cam' : []}
        for clinic_cam, clinic in zip(clinic_cams, clini_tokens):
            clinic_cam = clinic_cam.detach().cpu().squeeze()
            
            # # normalize scores
            # clinic_cam = (clinic_cam - clinic_cam.min()) / (clinic_cam.max() - clinic_cam.min())

            tokens = net.tokenizer.convert_ids_to_tokens(clinic['input_ids'].flatten())
            # print([(tokens[i], clinic_cam[i].item()) for i in range(len(tokens))])
            # vis_data_records = [visualization.VisualizationDataRecord(
            #                                 clinic_cam, 1, 1, 1, 1, 1, tokens, 1)]
            # viz = visualization.visualize_text(vis_data_records, legend=False)

            # # HTML 파일로 저장
            # html_file = os.path.join(save_path, name + '.html')
            # with open(html_file, "w") as file:
            #     file.write(viz.data)

            # # Selenium 설정
            # driver_path = 'path/to/chromedriver'  # ChromeDriver의 경로
            # driver = webdriver.Chrome(driver_path)

            # # HTML 파일 로드
            # driver.get("file://" + html_file)

            # # 스크린샷 저장
            # screenshot_file = "visualization.png"
            # driver.save_screenshot(screenshot_file)

            # # 브라우저 닫기
            # driver.quit()
            
            
            clinic_dict['tokens'].append(tokens)
            clinic_dict['clinic_cam'].append(clinic_cam)
            
        self.clinic_dict = clinic_dict
        
        self.relevance_score = net.get_relevance_score()
        
    def return_clinic_dict(self):
        return self.clinic_dict
    
    def return_relevance_score(self):
        return self.relevance_score
   

class LRP_multi:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, img, clinic, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
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

        return self.model.relprop(torch.tensor(one_hot_vector).to(img.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)



class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
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

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)



class GenerateCNNGradCAM(nn.Module):
    def __init__(self):
        super(GenerateCNNGradCAM, self).__init__()

    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None, correct=True):
        target_layers = [net.layer4[-1]]
        self.attribution_generator = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
        
        targets = [ClassifierOutputTarget(class_index)]
        grayscale_cam = self.attribution_generator(input_tensor=original_image, targets=targets)
        # grayscale_cam = np.expand_dims(grayscale_cam, 3)[0]
        grayscale_cam = grayscale_cam.transpose(1, 2, 0)
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())


        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(grayscale_cam, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(grayscale_cam, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)


        if correct:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'correct_GradCAM')
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'wrong_GradCAM')
            os.makedirs(save_path, exist_ok=True)
            

        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        plt.clf()



class GenerateCNNLRP(nn.Module):
    def __init__(self):
        super(GenerateCNNLRP, self).__init__()
        
    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None, correct=True):
        outputs = net(original_image)

        R = torch.ones(outputs.shape).cuda()
        R = net.fc.relprop(R, 1)
        R = R.reshape_as(net.avgpool.Y)
        R4 = net.avgpool.relprop(R, 1)
        R3 = net.layer4.relprop(R4, 1)
        R2 = net.layer3.relprop(R3, 1)
        R1 = net.layer2.relprop(R2, 1)
        R0 = net.layer1.relprop(R1, 1)
        R0 = net.maxpool.relprop(R0, 1)
        R0 = net.relu.relprop(R0, 1)
        R0 = net.bn1.relprop(R0, 1)
        R_final = net.conv1.relprop(R0, 1)
        R_final = R_final[0]
            
        R_t1, R_t1c, R_t2, R_flair = R_final
    
        
        R_t1 = minmax_norm(R_t1)
        R_t1c = minmax_norm(R_t1c)
        R_t2 = minmax_norm(R_t2)
        R_flair = minmax_norm(R_flair)
        zero_array = torch.zeros_like(R_flair).cuda()
        
        R_final = torch.stack([R_t1, R_t1c, R_t2, R_flair, zero_array], 0)
        R_final = R_final.detach().cpu().numpy()
        
        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(R_final[i], cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(R_final[i], cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        
        if correct:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'correct_LRP')
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'wrong_LRP')
            os.makedirs(save_path, exist_ok=True)
            

        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        plt.clf()        



class GenerateCNNRCAM(nn.Module):
    def __init__(self):
        super(GenerateCNNRCAM, self).__init__()
        self.value = dict()

    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None, correct=True):
        
        mode='layer2'
        target_layer = net.layer2
        
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)
        
        layer1, layer2, layer3, layer4, out = net(original_image, True)
        
        
        R = net.CLRP(out, [class_index])
        R = net.fc.relprop(R, 1)
        R = R.reshape_as(net.global_pool.Y)
        R4 = net.global_pool.relprop(R, 1)

        if mode == 'layer4':
            r_weight4 = torch.mean(R4, dim=(2, 3), keepdim=True)
            r_cam4 = layer4 * r_weight4
            r_cam = torch.sum(r_cam4, dim=(1), keepdim=True)
        elif mode == 'layer3':
            R3 = net.layer4.relprop(R4, 1)
            r_weight3 = torch.mean(R3, dim=(2, 3), keepdim=True)
            r_cam3 = layer3 * r_weight3
            r_cam = torch.sum(r_cam3, dim=(1), keepdim=True)
        elif mode == 'layer2':
            R3 = net.layer4.relprop(R4, 1)
            R2 = net.layer3.relprop(R3, 1)
            r_weight2 = torch.mean(R2, dim=(2, 3), keepdim=True)
            r_cam2 = layer2 * r_weight2
            r_cam = torch.sum(r_cam2, dim=(1), keepdim=True)
            
            
        r_cam = torch.nn.functional.interpolate(r_cam, size=(224, 224), mode='bilinear')
        r_cam = r_cam.reshape(224, 224).data.cpu().numpy()
        r_cam = (r_cam - r_cam.min()) / (r_cam.max() - r_cam.min())


        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(r_cam, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(r_cam, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

        if correct:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'correct_RCAM')
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = os.path.join(cfg.paths.output_dir, 'images', 'wrong_RCAM')
            os.makedirs(save_path, exist_ok=True)
            

        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        plt.clf()
    
    def forward_hook(self, module, input, output):
        self.value['activations'] = output
        
    def backward_hook(self, module, input, output):
        self.value['gradients'] = output[0]
