import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from models.transformer import TorchGeneratorModel,_build_encoder,_build_decoder,_build_encoder_mask, _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings,_create_entity_embeddings
from models.graph import SelfAttentionLayer,SelfAttentionLayer_batch
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import json

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

def concept_edge_list4GCN():
    node2index=json.load(open('data/word/key2index_3rd.json', encoding='utf-8'))
    f=open('data/word/conceptnet_edges2nd.txt', encoding='utf-8')
    edges=set()
    stopwords=set([word.strip() for word in open('data/word/stopwords.txt', encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        entity0=node2index[lines[1].split('/')[0]]
        entity1=node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).cuda()

class CrossModel(nn.Module):
    def __init__(self, opt, dictionary, is_finetune=False, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']

        # ---------------
        self.genre_len = opt['n_genre']
        with open('data/genre/entityId2genre_dict.pkl', 'rb') as f:
            dict_entityId2genre = pkl.load(f)
        self.genre = dict_entityId2genre

        with open('data/genre/genreId2entityId.pkl', 'rb') as f:
            genreId2entityId = pkl.load(f)
        self.genreId2entityId = genreId2entityId
        # -----------------------------

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings=_create_entity_embeddings(
            opt['n_concept']+1, opt['dim'], 0)
        self.concept_padding=0

        self.kg = pkl.load(
            open("data/dbmg/DBMG_subkg.pkl", "rb")
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.db_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])
        ##################
        self.self_attn_rv = SelfAttentionLayer_batch(opt['embedding_size'], opt['embedding_size'])
        self.self_attn_intro = SelfAttentionLayer_batch(opt['embedding_size'], opt['embedding_size'])
        ##################

        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        # -----------
        self.user_norm_genre = nn.Linear(opt['dim'] * 3, opt['dim'])
        self.gate_norm_genre = nn.Linear(opt['dim'], 3)
        # -----------
        self.copy_norm = nn.Linear(opt['embedding_size']*2+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)
        ##################
        self.vocab_size = len(dictionary) + 4
        self.copy_norm_rev = nn.Linear(opt['embedding_size']+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias_rev = nn.Linear(opt['embedding_size'], len(dictionary) + 4)
        self.copy_norm_intro = nn.Linear(opt['embedding_size'] + opt['embedding_size'], opt['embedding_size'])
        self.representation_bias_intro = nn.Linear(opt['embedding_size'], len(dictionary) + 4)
        ##################

        self.info_con_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_db_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1)
        self.info_con_loss = nn.MSELoss(size_average=False,reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False,reduce=False)
        # -------
        self.info_genre_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_genre = nn.Linear(opt['dim'], opt['n_entity'])
        # -------

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)

        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2)
        edge_list = list(set(edge_list))
        # print(len(edge_list), self.n_relation)
        self.dbpedia_edge_sets=torch.LongTensor(edge_list).cuda()
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation, num_bases=opt['num_bases'])
        #self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.n_con_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)

        #self.concept_GCN4gen=GCNConv(self.dim, opt['embedding_size'])

        w2i=json.load(open('data/word/word2index_redial_intro.json', encoding='utf-8'))
        self.i2w={w2i[word]:word for word in w2i}

        self.mask4key=torch.Tensor(np.load('data/introduction/mask4key20intro.npy')).cuda()
        self.mask4movie=torch.Tensor(np.load('data/introduction/mask4movie20intro.npy')).cuda()
        self.mask4=self.mask4key+self.mask4movie
        if is_finetune:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.concept_embeddings.parameters(),
                      self.self_attn.parameters(), self.self_attn_db.parameters(),self.user_norm.parameters(),self.gate_norm.parameters(),
                      self.output_en.parameters(),self.gate_norm_genre.parameters(), self.user_norm_genre.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def decode_greedy(self, encoder_states, xs_rev, xs_intro, encoder_states_kg, encoder_states_db, attention_kg, attention_db, attention_rv, attention_intro, bsz, maxlen):
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            scores = scores[:, -1:, :]
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            db_attn_norm = self.db_attn_norm(attention_db)

            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))
            copy_latent_rev = self.copy_norm_rev(torch.cat([attention_rv.unsqueeze(1), scores], -1))
            copy_latent_intro = self.copy_norm_intro(torch.cat([attention_intro.unsqueeze(1), scores], -1))

            con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)
            voc_logits = F.linear(scores, self.embeddings.weight)
            
            #############################
            mask4_rev = torch.zeros(bsz, self.vocab_size)
            index_list_mask_rev = xs_rev.tolist()
            mask4_rev = mask4_rev.scatter_(1, torch.LongTensor(index_list_mask_rev), torch.ones(bsz, self.vocab_size)).cuda()
            con_logits_rev = self.representation_bias_rev(copy_latent_rev)*mask4_rev.unsqueeze(1)

            mask4_intro = torch.zeros(bsz, self.vocab_size)
            index_list_mask_intro = xs_intro.tolist()
            mask4_intro = mask4_intro.scatter_(1, torch.LongTensor(index_list_mask_intro),torch.ones(bsz, self.vocab_size)).cuda()
            con_logits_intro = self.representation_bias_intro(copy_latent_intro)*mask4_intro.unsqueeze(1)
            #############################

            sum_logits = voc_logits + con_logits + con_logits_rev + con_logits_intro
            _, preds = sum_logits.max(dim=-1)

            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, xs_rev, xs_intro, encoder_states_kg, encoder_states_db, attention_kg, attention_db, attention_rv, attention_intro, ys):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db)

        kg_attention_latent=self.kg_attn_norm(attention_kg)
        db_attention_latent=self.db_attn_norm(attention_db)
        copy_latent = self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), db_attention_latent.unsqueeze(1).repeat(1,seqlen,1), latent], -1))

        copy_latent_rev = self.copy_norm_rev(torch.cat([attention_rv.unsqueeze(1).repeat(1,seqlen,1), latent], -1))
        copy_latent_intro = self.copy_norm_intro(torch.cat([attention_intro.unsqueeze(1).repeat(1,seqlen,1), latent], -1))

        logits=F.linear(latent, self.embeddings.weight)
        con_logits=self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)
        
        #############################
        mask4_rev = torch.zeros(bsz, self.vocab_size)
        index_list_mask_rev = xs_rev.tolist()
        mask4_rev = mask4_rev.scatter_(1, torch.LongTensor(index_list_mask_rev), torch.ones(bsz, self.vocab_size)).cuda()
        con_logits_rev = self.representation_bias_rev(copy_latent_rev)*mask4_rev.unsqueeze(1)

        mask4_intro = torch.zeros(bsz, self.vocab_size)
        index_list_mask_intro = xs_intro.tolist()
        mask4_intro = mask4_intro.scatter_(1, torch.LongTensor(index_list_mask_intro),torch.ones(bsz, self.vocab_size)).cuda()
        con_logits_intro = self.representation_bias_intro(copy_latent_intro) * mask4_intro.unsqueeze(1)
        #############################
        sum_logits = logits+con_logits+con_logits_rev + con_logits_intro
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    def infomax_loss(self, con_nodes_features, db_nodes_features, con_user_emb, db_user_emb, genre_user_emb, con_label, db_label, mask):
        #batch*dim
        #node_count*dim
        con_emb=self.info_con_norm(con_user_emb)
        db_emb=self.info_db_norm(db_user_emb)
        genre_emb = self.info_genre_norm(genre_user_emb)

        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)
        genre_scores = F.linear(genre_emb, db_nodes_features, self.info_output_genre.bias)

        info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.cuda().float()),dim=-1)*mask.cuda()
        info_con_loss=torch.sum(self.info_con_loss(con_scores,con_label.cuda().float()),dim=-1)*mask.cuda()
        info_genre_loss = torch.sum(self.info_con_loss(genre_scores, db_label.cuda().float()), dim=-1) * mask.cuda()

        return torch.mean(info_db_loss), torch.mean(info_con_loss), torch.mean(info_genre_loss)

    def forward(self, xs, ys, mask_ys, concept_mask, db_mask, reviews_mask, introduction_mask, seed_sets, entities_sets_altitude, labels, movie_sets_altitude,con_label, db_label, entity_vector, rec, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        if test == False:
            self.longest_label = max(self.longest_label, ys.size(1))

        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)
        
        xs_rev = xs.mul(reviews_mask.cuda())
        reviews_states, _ = self.encoder(xs_rev)

        xs_intro = xs.mul(introduction_mask.cuda())
        introduction_states, _ = self.encoder(xs_intro)

        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets)

        user_representation_list = []
        db_con_mask=[]

        genre_representation_list = []
        for i, seed_set in enumerate(seed_sets):
            if int(movie_sets_altitude[i]) == 1 and int(labels[i]) in self.genre.keys():
                genre_representation_list.append(self.self_attn_db(db_nodes_features[[self.genreId2entityId[i.item()] for i in torch.tensor(self.genre[int(labels[i])])]]))
            else:
                genre_representation_list.append(torch.zeros(self.dim).cuda())
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb=torch.stack(user_representation_list)
        db_con_mask=torch.stack(db_con_mask)
        db_movie_genre_emb = torch.stack(genre_representation_list)

        graph_con_emb=con_nodes_features[concept_mask.to(torch.int64)]
        con_emb_mask=concept_mask==self.concept_padding

        con_user_emb=graph_con_emb
        con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.cuda())

        # ----------
        genre_emb_count = 0
        user_emb_original = self.user_norm(torch.cat([con_user_emb, db_user_emb], dim=-1))
        uc_gate_original = F.sigmoid(self.gate_norm(user_emb_original))
        user_emb = uc_gate_original * db_user_emb + (1 - uc_gate_original) * con_user_emb

        user_emb_genre = self.user_norm_genre(torch.cat([con_user_emb, db_user_emb, db_movie_genre_emb], dim=-1))
        uc_gate_genre = F.softmax(self.gate_norm_genre(user_emb_genre))
        for i in range(user_emb.shape[0]):
            if sum(db_movie_genre_emb[i]) == 0:
                continue
            else:
                user_emb[i] = uc_gate_genre[i, 0] * db_user_emb[i] + uc_gate_genre[i, 1] * con_user_emb[i] + uc_gate_genre[i, 2] * db_movie_genre_emb[i]
                genre_emb_count += 1
        # ----------

        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias)

        mask_loss=0

        info_db_loss, info_con_loss, info_genre_loss = self.infomax_loss(con_nodes_features,db_nodes_features,con_user_emb,db_user_emb, db_movie_genre_emb, con_label,db_label,db_con_mask)

        rec_loss = self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda())
        rec_loss = torch.sum(rec_loss*rec.float().cuda())

        self.user_rep=user_emb

        #generation---------------------------------------------------------------------------------------------------
        rev_atten_emb, attention_rev = self.self_attn_rv(reviews_states, reviews_mask.cuda())
        intro_atten_emb, attention_intro = self.self_attn_intro(introduction_states, introduction_mask.cuda())

        con_nodes_features4gen = con_nodes_features#self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
        con_emb4gen = con_nodes_features4gen[concept_mask.to(torch.int64)]
        con_mask4gen = concept_mask != self.concept_padding
        #kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
        kg_encoding = (self.kg_norm(con_emb4gen),con_mask4gen.cuda())

        db_emb4gen = db_nodes_features[entity_vector.to(torch.int64)] #batch*50*dim
        db_mask4gen = entity_vector != 0
        #db_encoding = self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
        db_encoding = (self.db_norm(db_emb4gen),db_mask4gen.cuda())

        if test == False:
            # use teacher forcing
            scores, preds = self.decode_forced(
                encoder_states, xs_rev, xs_intro,
                kg_encoding, db_encoding, con_user_emb, db_user_emb, rev_atten_emb, intro_atten_emb,
                mask_ys
            )
            gen_loss = torch.mean(self.compute_loss(scores, mask_ys))
        else:
            scores, preds = self.decode_greedy(
                encoder_states, xs_rev, xs_intro,
                kg_encoding, db_encoding, con_user_emb, db_user_emb, rev_atten_emb, intro_atten_emb,
                bsz, maxlen or self.longest_label
            )
            gen_loss = None

        return scores, preds, entity_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss, info_genre_loss, genre_emb_count

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.cuda(), score_view.cuda())
        return loss

    def save_model(self):
        torch.save(self.state_dict(), 'saved_model/net_parameter1.pkl')

    def load_model(self):
        pretrained = torch.load('saved_model/net_parameter1.pkl')
        network_dict = self.state_dict()
        pretrained_dict = {k:v for k,v in pretrained.items()}
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in network_dict}
        network_dict.update(pretrained_dict)
        self.load_state_dict(network_dict)
        #self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsqueeze(dim=1)
        output += up_bias
        return output
