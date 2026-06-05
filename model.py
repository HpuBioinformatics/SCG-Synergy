# -*- coding: utf-8 -*-
import setuptools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch.utils.checkpoint as checkpoint


class GatedBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedBlock, self).__init__()
        self.value = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.value.bias, 0)
        nn.init.constant_(self.gate.bias, 1)

    def forward(self, x):
        val = self.value(x)
        gate = torch.sigmoid(self.gate(x))
        out = val * gate
        return self.norm(out)

class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, num_cellline, num_protein, embeddingSize, device):
        super(BioEncoder, self).__init__()
        self.device = device
        self.drug_encoder = GatedBlock(dim_drug, embeddingSize)
        self.cell_encoder = GatedBlock(dim_cellline, embeddingSize)
        self.protein = nn.Embedding(num_protein, embeddingSize)
        nn.init.kaiming_uniform_(self.protein.weight.data, nonlinearity='relu')
        self.proteinIndices = torch.arange(num_protein).long().to(self.device)

    def forward(self, Drug_Features, Cell_Line_Feature):
        x_drug = self.drug_encoder(Drug_Features)
        x_cell = self.cell_encoder(Cell_Line_Feature)
        x_protein = self.protein(self.proteinIndices)
        return x_drug, x_cell, x_protein

def normalize_l2(X):
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class rhgcnConv(nn.Module):
    def __init__(self, edge_num, Xe_class_length, degV_dict, in_channels, out_channels, num_edge_types=3, negative_slope=0.2, use_norm = True):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=True) for _ in range(num_edge_types + 1)])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.num_edge_types = num_edge_types
        self.reset_parameters()
        self.edge_num = edge_num
        self.Xe_class_length = Xe_class_length
        self.use_norm = use_norm
        self.degV_dict = degV_dict

    def reset_parameters(self):
        for layer in self.W:
            nn.init.kaiming_uniform_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)

    def forward(self, X, vertex, edges):
        X0 = self.W[0](X)
        Xve = []
        for i in range(self.num_edge_types):
            Xve.append((self.W[i + 1](X)*self.degV_dict[i])[vertex])

        Xe1 = []
        for i in range(self.num_edge_types):
            Xe1.append(scatter(Xve[i], edges, dim = 0, reduce = 'mean', dim_size = self.edge_num))

        Xe = Xe1[0][:self.Xe_class_length[0],:]
        for i in range(self.num_edge_types - 1):
            Xe = torch.cat((Xe,Xe1[i+1][self.Xe_class_length[i]:self.Xe_class_length[i+1],:]),0)

        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim = 0, reduce = 'sum', dim_size = X.shape[0])
        Xv = Xv + X0
        if self.use_norm:
            Xv = normalize_l2(Xv)
        Xv = self.leaky_relu(Xv)
        return Xv

class RHGNN(nn.Module):
    def __init__(self, V, E, edge_num, Xe_class_length, degV_dict, nfeat, nhid, out_dim, num_edge_types, dropout):
        super().__init__()
        self.conv_in = rhgcnConv(edge_num, Xe_class_length, degV_dict, nfeat, nhid, num_edge_types)
        self.conv_out1 = rhgcnConv(edge_num, Xe_class_length, degV_dict, nhid, out_dim, num_edge_types)
        self.V = V
        self.E = E
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.conv_in(X, self.V, self.E)
        X = self.dropout(X)
        X = self.conv_out1(X, self.V, self.E)
        return X

class ChannelAttention(nn.Module):
    def __init__(self, emb_size):
        super(ChannelAttention, self).__init__()
        self.weights = nn.ParameterDict({
            'attention': nn.Parameter(torch.randn(1, emb_size)),
        })
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights['attention'])

    def forward(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(torch.matmul(self.weights['attention'], embedding.t())) 
        score = F.softmax(torch.stack(weights, dim=1), dim=1)
        mixed_embeddings = torch.zeros_like(channel_embeddings[0])
        for i in range(len(weights)):
            mixed_embeddings += (score[:, i]*channel_embeddings[i].t()).t()
        return mixed_embeddings, score

class ContextGatedSelfAttentionPredictor(nn.Module):
    def __init__(self, feature_dim, nhead=4, dim_feedforward=512, dropout=0.5):
        super(ContextGatedSelfAttentionPredictor, self).__init__()
        self.feature_dim = feature_dim
        self.context_gate = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_uniform_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, 1.0) 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        for p in self.predictor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, full_embedding, index):
        outputs = {}
        BATCH_CHUNK_SIZE = 512 

        for cellidx, indices in index.items():
            num_samples = indices.size(0)
            cell_scores_list = []
            
            for start_i in range(0, num_samples, BATCH_CHUNK_SIZE):
                end_i = min(start_i + BATCH_CHUNK_SIZE, num_samples)
                batch_indices = indices[start_i:end_i]
                drugA_feat = full_embedding[batch_indices[:, 0]]
                drugB_feat = full_embedding[batch_indices[:, 1]]
                cell_feat = full_embedding[cellidx].unsqueeze(0).expand(drugA_feat.size(0), -1)
                
                gate_signal = torch.sigmoid(self.context_gate(cell_feat))
                drugA_gated = drugA_feat * gate_signal
                drugB_gated = drugB_feat * gate_signal
                seq = torch.stack([drugA_gated, drugB_gated, cell_feat], dim=1)
                transformed = checkpoint.checkpoint(self.transformer_encoder, seq, use_reentrant=False)
                pooled = transformed.mean(dim=1)
                score = self.predictor(pooled).squeeze(-1)
                cell_scores_list.append(score)
            
            outputs[cellidx] = torch.cat(cell_scores_list, dim=0)
            
        return outputs

class ContrastiveLoss(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, temperature=0.5):
        """
        Contrastive Loss Module
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Dimension of projection head
            temperature: Softmax temperature scaling
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, view1, view2):
        """
        Args:
            view1: Drug Embeddings from Hypergraph 1 [Num_Drugs, Dim]
            view2: Drug Embeddings from Hypergraph 2 [Num_Drugs, Dim]
        """
        z1 = self.projection_head(view1)
        z2 = self.projection_head(view2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        batch_size = z1.size(0)
        labels = torch.arange(batch_size).to(z1.device)
        loss_v1_to_v2 = F.cross_entropy(sim_matrix, labels)
        loss_v2_to_v1 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_v1_to_v2 + loss_v2_to_v1) / 2

class Synergy(nn.Module):
    def __init__(self, numDrug, BioEncoder, encoder1, encoder2, attention, decoder, contrastive_loss_module):
        super(Synergy, self).__init__()
        self.BioEncoder = BioEncoder
        self.hgnn_encoder1 = encoder1
        self.hgnn_encoder2 = encoder2
        self.attention = attention
        self.decoder = decoder
        self.numDrug = numDrug
        self.contrastive_loss_module = contrastive_loss_module
    def forward(self, Drug_Features, Cell_Line_Feature, combination_index):
        x_drug, x_cell, x_protein = self.BioEncoder(Drug_Features, Cell_Line_Feature)
        hypergraph1 = torch.cat((x_drug, x_protein), 0) 
        hypergraph2 = torch.cat((x_drug, x_cell), 0)    
        embedding1 = self.hgnn_encoder1(hypergraph1)
        embedding2 = self.hgnn_encoder2(hypergraph2)
        drug_emb1 = embedding1[:self.numDrug,:]
        drug_emb2 = embedding2[:self.numDrug,:]
        cl_loss = self.contrastive_loss_module(drug_emb1, drug_emb2)
        drug_embedding_fused, _ = self.attention(*[drug_emb1, drug_emb2])
        cell_features = embedding2[self.numDrug:, :]
        full_embedding = torch.cat([drug_embedding_fused, cell_features], dim=0)
        result = self.decoder(full_embedding, combination_index)
        return result, cl_loss