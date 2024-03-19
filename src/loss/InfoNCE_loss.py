import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def NCE_loss(features, targets, memory_bank, memory_targets, args, features_dict=None):
    
    previous_feature = features_dict['logits']

    if len(memory_bank) != args.batch_size:
        memory_bank = torch.cat((memory_bank, previous_feature.detach()), dim=0)
        memory_targets = torch.cat((memory_targets, targets), dim=0)
        nce = 0
    else:
        query = F.normalize(features, dim=-1)
        # if not args.selfkey:
        key = F.normalize(memory_bank, dim=-1)
        # else:
            # key = F.normalize(features, dim=-1)
        
        query_key = torch.cat((query, key), 0)
        # if not args.selfkey:
        query_key_targets = torch.cat((targets, memory_targets), 0)
        # else:
        #     query_key_targets = torch.cat((targets, targets), 0)

        sim_matrix = torch.mm(query_key, query_key.t().contiguous()) / args.temperature
        expand_targets = torch.unsqueeze(query_key_targets, dim=1)
        
        pos_mask = torch.eq(expand_targets, expand_targets.T).float().to(args.device)
        pos_mask = pos_mask - torch.eye(pos_mask.size(0)).to(args.device)
        
        # neg_mask = torch.ones_like(sim_matrix) - pos_mask
        
        exp_sim_matrix = torch.exp(sim_matrix)
        exp_sim_matrix_sum = exp_sim_matrix.sum(dim=-1, keepdim=True)
        
        log_exp_logits = - torch.log(exp_sim_matrix / exp_sim_matrix_sum)
        
        log_exp_pos_logits = log_exp_logits * pos_mask
        pos_pair_count = pos_mask.sum(dim=-1)
        pos_pair_count[pos_pair_count==0] = 1
        nce = (log_exp_pos_logits.sum(dim=-1) / pos_pair_count).mean()
        # nce = (log_exp_pos_logits.sum(dim=-1) / pos_mask.sum(dim=-1)).mean()
        
        # for i in range(query.shape[0]):
        #     pos_key = query_key[torch.where(query_key_targets == targets[i].cpu())]
        #     neg_key = query_key[torch.where(query_key_targets != targets[i].cpu())]
            
        #     if len(pos_key) == 0 or len(neg_key) == 0:
        #         continue
            
        #     # pos_key = torch.mean(pos_key, keepdim=True, dim=0)
            
        #     if init:
        #         pos_sim = torch.unsqueeze(torch.exp(torch.sum(query[i] * pos_key, dim=-1) / args.temperature).sum(), dim=0)
        #         neg_sim = torch.unsqueeze(torch.exp(torch.sum(query[i] * neg_key, dim=-1) / args.temperature).sum(), dim=0)
        #         init = False
        #     else:
        #         pos_sim = torch.cat((pos_sim, torch.unsqueeze(torch.exp(torch.sum(query[i] * pos_key, dim=-1) / args.temperature).sum(), dim=0)), dim=0)
        #         neg_sim = torch.cat((neg_sim, torch.unsqueeze(torch.exp(torch.sum(query[i] * neg_key, dim=-1) / args.temperature).sum(), dim=0)), dim=0)
                    
        # nce = (- torch.log(pos_sim / neg_sim)).mean()
        
        
        memory_bank = torch.cat((memory_bank[len(previous_feature):], previous_feature.detach()), dim=0)
        memory_targets = torch.cat((memory_targets[len(targets):], targets), dim=0)

        
    return memory_bank, memory_targets, nce