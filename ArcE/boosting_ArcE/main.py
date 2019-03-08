import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits
from model import ArcE, DistMult, Complex
from utils import load_pretrain_emb, load_description

from src.spodernet.spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from src.spodernet.spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from src.spodernet.spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from src.spodernet.spodernet.utils.global_config import Config, Backends
from src.spodernet.spodernet.utils.logger import Logger, LogLevel
from src.spodernet.spodernet.preprocessing.batching import StreamBatcher
from src.spodernet.spodernet.preprocessing.pipeline import Pipeline
from src.spodernet.spodernet.preprocessing.processors import TargetIdx2MultiTarget
from src.spodernet.spodernet.hooks import LossHook, ETAHook
from src.spodernet.spodernet.utils.util import Timer
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from src.spodernet.spodernet.preprocessing.processors import TargetIdx2MultiTarget
np.set_printoptions(precision=3)

# timer = CUDATimer()
cudnn.benchmark = True
if not os.path.exists("./saved_models"):
    os.mkdir("./saved_models")
# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)

Config.cuda = False
Config.embedding_dim = 200
#Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG
Config.desp_emb_dim = 100
Config.batch_size = 128


#model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = 2500
load = False
if Config.dataset is None:
    Config.dataset = 'FB15k-237'
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)
desp_path_wiki = 'FB15k/FB15k_description.txt'
desp_path_dbp = 'FB15k/wiki_data.txt'
pre_emb_file = 'FB15k/glove.6B.{0}d.txt'.format(Config.desp_emb_dim)



''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)


def main():
    if Config.process: preprocess(Config.dataset, delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']
    num_entities = vocab['e1'].num_token

    train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)
    dev_rank_batcher = StreamBatcher(Config.dataset, 'dev_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)
    test_rank_batcher = StreamBatcher(Config.dataset, 'test_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)

    wiki_words, entity2wiki, max_len1 = load_description(desp_path_wiki, 100)
    dbp_words, entity2dbp, max_len2 = load_description(desp_path_dbp, 15)
    words = wiki_words | dbp_words
    max_len = max(max_len1, max_len2)
    emb_words, word2id = load_pretrain_emb(pre_emb_file, words)  # emb_words: wordid--embedding

    word_embedding = np.empty([len(emb_words), Config.desp_emb_dim])
    for i, w in enumerate(emb_words):
        word_embedding[i, :] = emb_words[i]

    # words, entity2desp, max_len = load_description(desp_path)
    # emb_words, word2id = load_pretrain_emb(pre_emb_file, words)  # emb_words:
    # word_embedding = np.empty([len(emb_words), Config.desp_emb_dim])
    # for i, w in enumerate(emb_words):
    #     word_embedding[i, :] = emb_words[i]
    print("len ", max_len1, max_len2)
    e_id_desp_wiki = {}
    e_id_desp_dbp = {}
    entitydesp_wiki = {}
    entitydesp_dbp = {}
    for e in entity2wiki:
        e_id_desp_wiki[vocab['e1'].get_idx(e)] = entity2wiki[e]
        e_id_desp_dbp[vocab['e1'].get_idx(e)] = entity2dbp[e]

    for i in range(num_entities):
        if vocab['e1'].get_word(i) in entity2wiki:
            # print(vocab['e1'].get_word(i))
            e_id_desp_wiki[i] = entity2wiki[vocab['e1'].get_word(i)]
            e_id_desp_dbp[i] = entity2dbp[vocab['e1'].get_word(i)]
        else:
            # print(vocab['e1'].get_word(i))
            e_id_desp_wiki[i] = ['<PAD>'] * (max_len1 + 2)
            e_id_desp_dbp[i] = ['<PAD>'] * (max_len2 + 2)
    # e_id_desp:{entity_id:[w1,w2,...,wn]}
    for i in e_id_desp_dbp:
        tmp_sen = e_id_desp_dbp.get(i)
        id_sen = []
        for w in tmp_sen:
            id_sen.append(word2id[w])
        entitydesp_dbp[i] = id_sen  # entityid--wordid
    for i in e_id_desp_wiki:
        tmp_sen = e_id_desp_wiki.get(i)
        id_sen = []
        for w in tmp_sen:
            id_sen.append(word2id[w])
        entitydesp_wiki[i] = id_sen  # entityid--wordid

    #######################################################


    if Config.model_name is None:
        model = ArcE(vocab['e1'].num_token, vocab['rel'].num_token, word_embedding, max_len)
    elif Config.model_name == 'ArcE':
        model = ArcE(vocab['e1'].num_token, vocab['rel'].num_token, word_embedding, max_len)
    elif Config.model_name == 'DistMult':
        model = DistMult(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ComplEx':
        model = Complex(vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        log.info('Unknown model: {0}', Config.model_name)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))


    eta = ETAHook('train', print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))

    if Config.cuda:
        model.cuda()
    if load:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
        ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model.init()

    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))

    opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
    for epoch in range(epochs):
        model.train()
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var['e1']
            rel = str2var['rel']
            e2_multi = str2var['e2_multi1_binary'].float()
            # label smoothing
            e2_multi = ((1.0-Config.label_smoothing_epsilon)*e2_multi) + (1.0/e2_multi.size(1))
            entity_wiki = np.empty([Config.batch_size, max_len1 + 2])
            entity_dbp = np.empty([Config.batch_size, max_len2 + 2])
            for i, e in enumerate(e1):
                idx = int(e[0].item())
                entity_wiki[i, :] = entitydesp_wiki[idx]
                entity_dbp[i, :] = entitydesp_dbp[idx]
            pred = model.forward(e1, rel, entity_wiki,entity_dbp)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

            train_batcher.state.loss = loss.cpu()


        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', entitydesp_wiki,entitydesp_dbp, max_len1, max_len2, epoch)
 
            if epoch % 2 == 0:
                if epoch > 0:
                    ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation', entitydesp_wiki,entitydesp_dbp, max_len1, max_len2, epoch)


if __name__ == '__main__':
    main()
