import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from src.spodernet.spodernet.utils.global_config import Config
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from torch.nn.init import xavier_normal_, xavier_uniform_
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# timer = CUDATimer()


class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred



class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, word_embedding, max_len):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
       
        self.conv2 = torch.nn.Conv2d(32, 32, (3, 3), 1, 2, bias=Config.use_bias, dilation=2)
        self.conv3 = torch.nn.Conv2d(32, 32, (3, 3), 1, 2, bias=Config.use_bias, dilation=2)
        self.conv4 = torch.nn.Conv2d(32, 32, (3, 3), 1, 2, bias=Config.use_bias, dilation=2)
        self.conv_dilation = torch.nn.Conv2d(32, 32, (3, 3), 1, 3, bias=Config.use_bias, dilation=2)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.b1 = torch.nn.BatchNorm2d(32)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(16000,Config.embedding_dim)
        print(num_entities, num_relations)

        self.max_len = max_len
        self.max_len1 = 100
        self.max_len2 = 15
        self.num_entities = num_entities
        ######################description-CNN########################
        self.filter_size = 3
        self.num_filters = 100
        self.desp_word_dim = 100
        self.b_cnn = torch.nn.BatchNorm2d(100)

        self.word_embedding = torch.nn.Embedding(len(word_embedding), self.desp_word_dim, padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.Tensor(word_embedding))

        self.conv_desp = torch.nn.Conv2d(1, self.num_filters, (self.filter_size, self.desp_word_dim), 1, 0, bias=False)
        self.conv_desp_dbp = torch.nn.Conv2d(1, self.num_filters, (self.filter_size, self.desp_word_dim), 1, 0, bias=False)

        ########Inside-attention#######
        self.vc = torch.nn.Linear(100, 1)
        self.mc = torch.nn.Linear(300, 100)
        self.uc = torch.nn.Linear(200, 100)
        self.active = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

        """-----Outside Attention-------"""
        self.W1 = torch.nn.Linear(self.max_len2, self.num_filters)
        self.W2 = torch.nn.Linear(self.desp_word_dim, 1)
        self.We = torch.nn.Linear(self.num_filters, 1)
        self.U1 = torch.nn.Linear(200, 1)
        self.Wa = torch.nn.Linear(100, 200)
        self.Wb = torch.nn.Linear(200, 100)
        self.W_gate_e = torch.nn.Linear(200, 200)
        self.W_gate_hd = torch.nn.Linear(100, 200)
        self.linear_trans = torch.nn.Linear(100, 200)
        #############################################################

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        # self.e_all = torch.cat(self.emb_e,self.emb_e)

    def forward(self, e1, rel, entity_wiki, entity_dbp):
        ######################description-CNN########################
        wiki_words = torch.autograd.Variable(torch.Tensor(entity_wiki).long())
        dbp_words = torch.autograd.Variable(torch.Tensor(entity_dbp).long())
        if Config.cuda:
            wiki_words = wiki_words.cuda()
            dbp_words = dbp_words.cuda()
        wiki_words = wiki_words.view(Config.batch_size * (self.max_len1 + 2), 1)
        dbp_words = dbp_words.view(Config.batch_size * (self.max_len2 + 2), 1)
        wiki_e1_desp = self.word_embedding(wiki_words)
        #########Inside-attention#########
        wiki_e1_desp = wiki_e1_desp.view(Config.batch_size, self.max_len1 + 2, self.desp_word_dim)
        wiki_e1_context_left = wiki_e1_desp.narrow(1, 0, self.max_len1)
        wiki_e1_context_right = wiki_e1_desp.narrow(1, 2, self.max_len1)
        wiki_e1_desp = wiki_e1_desp.narrow(1, 1, self.max_len1)
        wiki_e1_context = (wiki_e1_context_left + wiki_e1_context_right + wiki_e1_desp) / 3

        dbp_e1_desp = self.word_embedding(dbp_words)
        dbp_e1_desp = dbp_e1_desp.view(Config.batch_size, self.max_len2 + 2, self.desp_word_dim)
        dbp_e1_context_left = dbp_e1_desp.narrow(1, 0, self.max_len2)
        dbp_e1_context_right = dbp_e1_desp.narrow(1, 2, self.max_len2)
        dbp_e1_desp = dbp_e1_desp.narrow(1, 1, self.max_len2)
        dbp_e1_context = (dbp_e1_context_left + dbp_e1_context_right + dbp_e1_desp) / 3

        entity_emb = self.emb_e(e1).view(Config.batch_size, 200)
        relation_emb = self.emb_rel(rel).view(Config.batch_size, 1, 200)

        entity_expand_wiki = entity_emb.view(Config.batch_size, 1, 200).expand(Config.batch_size, self.max_len1, 200)
        entity_expand_dbp = entity_emb.view(Config.batch_size, 1, 200).expand(Config.batch_size, self.max_len2, 200)
        #        print(entity_expand.size())



        wiki_entity_context = torch.cat([wiki_e1_context, entity_expand_wiki], 2)
        dbp_entity_context = torch.cat([dbp_e1_context, entity_expand_dbp], 2)
        #        print(entity_context.size())

        wiki_entity_context = wiki_entity_context.view(Config.batch_size, self.max_len1, 1, 300)
        dbp_entity_context = dbp_entity_context.view(Config.batch_size, self.max_len2, 1, 300)

        wiki_relation_emb = relation_emb.expand(Config.batch_size, self.max_len1, 200).contiguous().view(
            Config.batch_size, self.max_len1, 1, 200)
        dbp_relation_emb = relation_emb.expand(Config.batch_size, self.max_len2, 200).contiguous().view(
            Config.batch_size, self.max_len2, 1, 200)

        wiki_fl = self.active(self.mc(wiki_entity_context) + self.uc(wiki_relation_emb))
        wiki_fl = self.vc(wiki_fl)
        dbp_fl = self.active(self.mc(dbp_entity_context) + self.uc(dbp_relation_emb))
        dbp_fl = self.vc(dbp_fl)

        wiki_fl = wiki_fl.view(Config.batch_size, self.max_len1)
        wiki_fl = self.softmax(wiki_fl)
        wiki_fl = wiki_fl.view(Config.batch_size, self.max_len1, 1)

        dbp_fl = dbp_fl.view(Config.batch_size, self.max_len2)
        dbp_fl = self.softmax(dbp_fl)
        dbp_fl = dbp_fl.view(Config.batch_size, self.max_len2, 1)
        ####################################
        wiki_e1_desp = wiki_fl * wiki_e1_desp
        wiki_e1_desp = wiki_e1_desp.view(Config.batch_size, 1, self.max_len1, self.desp_word_dim)
        wiki_e1_desp = self.inp_drop(wiki_e1_desp)
        wiki_e1_desp_out = self.conv_desp(wiki_e1_desp)
        wiki_e1_desp_out = self.feature_map_drop(wiki_e1_desp_out)
        d1 = F.max_pool2d(wiki_e1_desp_out, (self.max_len1 - 2, 1)).view(Config.batch_size, self.num_filters)
       # d1 = F.relu(d1)

        dbp_e1_desp = dbp_fl * dbp_e1_desp
        dbp_e1_desp = dbp_e1_desp.view(Config.batch_size, 1, self.max_len2, self.desp_word_dim)
        dbp_e1_desp = self.inp_drop(dbp_e1_desp)
        dbp_e1_desp_out = self.conv_desp_dbp(dbp_e1_desp)
        dbp_e1_desp_out = self.feature_map_drop(dbp_e1_desp_out)
        d2 = F.max_pool2d(dbp_e1_desp_out, (self.max_len2 - 2, 1)).view(Config.batch_size, self.num_filters)
       # d2 = F.relu(d2)

        We_product_d1 = self.Wa(d1) * entity_emb
        We_product_d1 = torch.sum(We_product_d1,1)
        We_product_d2 = self.Wa(d2) * entity_emb
        We_product_d2 = torch.sum(We_product_d2, 1)
         ####out_att_2
        U1_product_r1 = self.Wa(d1) * self.emb_rel(rel).view(Config.batch_size, 200)
        U1_product_r1 = torch.sum(U1_product_r1,1)
        U1_product_r2 = self.Wa(d2) * self.emb_rel(rel).view(Config.batch_size, 200)
        U1_product_r2 = torch.sum(U1_product_r2, 1)

        g1 = F.tanh(We_product_d1 + U1_product_r1).view(Config.batch_size, 1)
        g2 = F.tanh(We_product_d2 + U1_product_r2).view(Config.batch_size, 1)

        ## out_att_U1#####
       ## U1_product_r = self.U1(self.emb_rel(rel)).view(Config.batch_size)

       ## g1 = F.tanh(We_product_d1 + U1_product_r).view(Config.batch_size, 1)
       ## g2 = F.tanh(We_product_d2 + U1_product_r).view(Config.batch_size, 1)
        g = torch.cat([g1, g2], -1)
        alpha = F.softmax(g, -1)
        hd = (alpha[:,0].view(Config.batch_size, 1) * d1 + alpha[:,1].view(Config.batch_size,1) * d2)
        
        e1_embedded = torch.cat([entity_emb, hd], 1)
        e1_embedded = e1_embedded.view(Config.batch_size, 1, 15,20)
        #################
        #e1_embedded= self.emb_e(e1).view(-1, 1, 10, 20) #[128,1,10,20])
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)#[128,1,20,20]
        # print(">>>> ",stacked_inputs.size())
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        #####add resnet and dilation convolution##########
        res = x

        x = self.conv1(x)
        # x = self.b1(x)
        # x = F.relu(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_dilation(x)
        x = x + res
        ################################################
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred


# Add your own model here

class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = F.sigmoid(output)

        return prediction
