import torch
from torch import nn
from .components.mlp import MultiLayerPerceptron
from .db.queryProcessor import Retriever
from .components.xp import transformxp
from .AdjCreation.loadAdj import AdjacencyLoader
from .AdjCreation.fetchAdj import extractTODadj
from .components.MIA import MemoryInfusedAttention
from .components.gcnStage1 import GC1L
from .components.gcnStage2 import GC2L
from .components.gcnStage3 import GC3L
from .components.attention import SelfAttentionLayer
from .components.TAdNet import AdjNet, GLU
from .components.gcnFusionModule import ABFusion
from .components.STfusion import stfusion


class archPro(nn.Module):
    def __init__(self, **model_args):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_len = model_args["output_len"]
        self.input_len = model_args["input_len"]
        self.db_path = model_args["db_path"]
        self.adj_path = model_args["adj_path"]
        self.window_size = model_args["window_size"]
        self.rx = model_args["rx"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.dropout = model_args["dropout"]
        self.adj = model_args["adj_mx"]
        self.k = model_args["k"]
        self.adjnet = AdjNet(self.num_nodes,self.input_len,self.adj, self.k)
        self.glu = GLU(input_channel=self.node_dim*2, output_channel=self.node_dim)
        self.gcFusion = ABFusion(feature_dim = self.node_dim)
        self.STfusion = stfusion(input_dim=self.node_dim, time_steps=self.window_size, out_dim=self.node_dim*4)
        self.retriever = Retriever(self.db_path, self.window_size)
        self.daloader = AdjacencyLoader(self.adj_path, device=self.device)
        self.dAdj = self.daloader.get()
        self.mia = MemoryInfusedAttention( embed_dim=self.num_nodes, use_batchnorm=False, use_layernorm=True)
        self.gc1 = GC1L(self.node_dim, self.node_dim*2, self.node_dim)
        self.gc2 = GC2L(self.node_dim, self.node_dim*2, self.node_dim)
        self.gc3 = GC3L(self.node_dim, self.node_dim*2, self.node_dim)
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)


        self.short_emb_layer = nn.Conv2d(
            in_channels=self.window_size, out_channels=self.node_dim, kernel_size=(1, 1), bias=True)
        self.long_emb_layer = nn.Conv2d(
            in_channels=self.rx+self.window_size, out_channels=self.node_dim, kernel_size=(1, 1), bias=True)
        self.embed_dimx = self.node_dim

        self.convEmbedder = nn.Linear(self.window_size * 2, self.embed_dimx).to(self.device)

        nn.init.xavier_uniform_(self.convEmbedder.weight)
        nn.init.zeros_(self.convEmbedder.bias)

        
        self.ppv_emb_layer = nn.Conv2d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * (self.input_len+self.rx), out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        
        self.decoder = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.OuputEncoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim+self.node_dim+(self.node_dim*4), self.hidden_dim+self.node_dim+(self.node_dim*4)) for _ in range(self.num_layer)])
        self.conv2d = nn.Conv2d(
            in_channels=self.hidden_dim+self.node_dim+(self.node_dim*4), out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        self.regression = transformxp

        self.project = nn.Linear(1, self.node_dim)

        self.temporal_attn = SelfAttentionLayer(model_dim=self.node_dim, num_heads=4)
        self.spatial_attn = SelfAttentionLayer(model_dim=self.node_dim, num_heads=4)





    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:

    #<-------------------------------INPUT PREPRATION/SELECTION/RETRIVAL--------------------------------------------------------------------------------------------->
        adj, fadj = extractTODadj(history_data, self.dAdj)
        x = history_data
        B, W, F, C = x.shape
        x_np = x.detach().cpu().numpy()

        # retrieved_np, indices = self.retriever.retrieve_batch(x_np, X=self.rx)
        retrieved_np, _ = self.retriever.retrieve_batch(x_np, X=self.rx)
        x_ex = torch.from_numpy(retrieved_np).float()
        x_ex = x_ex.to(self.device)
        xx = x_ex
        xs = history_data[:,:,:,0]
        dadj,radj,padj = self.adjnet(xs,adj)




    #<---------------------------------Memory Infused Attention Layer :Long X Short attention procedure ------------------------------------------------------------------------------------------------------->
        #"for now used a non learning attention funtion but can improve it by using a learnable attention layer class"
        xl = x_ex[:,0:self.rx,:,0]
        xls = self.mia(xs, xl)
        xls = xls.unsqueeze(-1)
        x = x[:,:,:,0:1]
        xls = torch.concat([x,xls], dim = -1)
        xls = xls.reshape(xls.shape[0], xls.shape[1] * xls.shape[3], xls.shape[2], 1)
        xls = xls.squeeze(-1)           # [B, 24, N]
        xls = xls.permute(0, 2, 1)      # [B, N, 24]


        xls = self.convEmbedder(xls)   # Linear(24, 32) --> [B, N, 32]
        xls = xls.permute(0, 2, 1).unsqueeze(-1)  # Back to [B, 32, N, 1]
        xls = xls.contiguous()  # [B, C, N, L]
        xls = xls.squeeze(-1)
        xls = xls.permute(0,2,1)

#<--------------------------------Three - Step GCN BLOCk ------------------------------------------------------------------------------------------------------->
        #step1 for dynamic
        g1c = self.gc1(xls,dadj)
        g2c = self.gc2(xls, radj)
     
        g12c = torch.cat([g1c,g2c], dim = -1)
        gfc = self.glu(g12c)
        
        #step3 for Pivot
        g3c = self.gc3(xls, padj)
        
        gcfused = self.gcFusion(gfc,g3c)

        gcfused = gcfused.permute(0,2,1)
    
#<---------------------------------INPUT EMBEDDING PREPRATION------------------------------------------------------------------------------------------------------->

        #1.#TOD Embedding
        x_exe = x_ex
        if self.if_time_in_day:
            t_i_d_data = x_exe[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None

        #2.#DOW Embedding
        if self.if_day_in_week:
            d_i_w_data = x_exe[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        #3.#Time Series Embedding
        batch_size, _, num_nodes, _ = x_exe.shape
        x_exe = x_exe.transpose(1, 2).contiguous()
        x_exe = x_exe.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(x_exe)

        #4.#Spatial/Node Embedding
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        #5.#Temporal Embeddings {basically derived from TOD embedding and DOW embedding}
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        
        ## Combined Embedding
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        hidden = self.encoder(hidden)
        hidden_saved = hidden.squeeze(-1)
        hidden = self.decoder(hidden)
        hidden = self.project(hidden)


        hidd_t = self.temporal_attn(hidden, dim = -3)
       
        hidd_s = self.spatial_attn(hidden, dim = -2)
        
        
        stf = self.STfusion(hidd_t,hidd_s)

#<---------------------------------Context Fusion ------------------------------------------------------------------------------------------------------->
       
        gcemb = torch.cat([stf,gcfused,hidden_saved], dim =1)

        gcemb = gcemb.unsqueeze(-1)



#<-------------------------------------output layer ------------------------------------------------------->
        pred = self.OuputEncoder(gcemb)
        prediction = self.conv2d(pred)
        prediction = self.regression(prediction)
        return prediction
