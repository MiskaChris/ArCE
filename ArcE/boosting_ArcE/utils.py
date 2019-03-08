import numpy as np

def load_pretrain_emb(embedding_path, words):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    emb_words = dict()
    word2id = {}
    for i, w in enumerate(words):
        word2id[w] = i
    word2id['<PAD>'] = len(words)
    words.add('<PAD>')
    word2id['<B>'] = len(words)
    words.add('<B>')
    word2id['<E>'] = len(words)
    words.add('<E>')
    #print(len(word2id), len(emb_words))
    for word in words:
        if word in embedd_dict:
            emb_words[word2id[word]] = embedd_dict[word]
        else:
            emb_words[word2id[word]] = embedd_dict['.']
    print(len(word2id), len(emb_words))
    return emb_words, word2id

def load_description(description_path,len):
    entity2id = {}
    words = set()
    entity2d = []
    entity2desp = {}
    entity = []
    max_len = 0
    with open('FB15k_description/entity2id.txt') as file:
        for line in file:
            tmp = line.split()
            entity2id[tmp[0]] = tmp[1]

    with open(description_path, 'r') as file:
        for line in file:
            count = 0
            tmp_desp = []
            line = line.replace('\\n',' ')
            tmp = line.split()
            e = tmp[0]
            for w in range(1,tmp.__len__()):
                if w == 1:
                    word = tmp[w].replace('"','')
                else:
                    word = tmp[w].replace('\\"','')
                    word = word.replace(',','')
                    word = word.replace('.', '')
                    word = word.replace('"@en', '')
                words.add(word.lower())
                tmp_desp.append(word.lower())
                count += 1
                if count >= len:
                    break
            if count > max_len:
                max_len = count
            entity2d.append(tmp_desp)
            entity.append(e)

    for i, e in enumerate(entity):
        entity2desp[e] = entity2d[i]
    print(max_len)
    pad_sequence(entity2desp, max_len)
    return words, entity2desp, max_len

def pad_sequence(entity2desp, max_len):
    for e in entity2desp:
        tmp_arr = entity2desp[e]
        if len(tmp_arr) < max_len:
            tmp_arr.extend(['<PAD>'] * (max_len - len(tmp_arr)))
            entity2desp[e] = tmp_arr
        tmp_arr.insert(0, '<B>')
        tmp_arr.insert(max_len+1, '<E>')
    return entity2desp
def get_ent_pos():
    file_desp = open('FB15k/FB15k_description.txt', 'r')
    file_name = open('FB15k/FB15k_mid2name.txt', 'r')
    ent_pos = {}
    ent_name = {}
    max_len = 0
    count = 0
    for line in file_name:
        tmp = line.split()
        ent_name[tmp[0]] = tmp[1].lower()
    for line in file_desp:
        sen = line.replace('"', '')
        sen = sen.replace(',', '')
        sen = sen.replace('\\\\','')
        sen_arr = sen.split()
        tmp = ''
        for i in range(1,len(sen_arr)):
            tmp += sen_arr[i] + " "
        tmp = tmp.lower()
        ent = ent_name[sen_arr[0]]
        ent = ent.split('_')
        sen = tmp.split()

        if ent[0] in sen:
            begin = sen.index(ent[0])
        elif ent[-1] in sen:
            begin = sen.index(ent[-1])

        if ent[-1] in sen:
            end = sen.index(ent[-1])
        elif ent[0] in sen:
            end = sen.index(ent[0])
        if end - begin > 18:
            end = begin
            count += 1
        ent_pos[sen_arr[0]] = [begin,end]
        if end - begin + 1 > max_len:
            max_len = end - begin + 1

    file_desp.close()
    file_name.close()

    return ent_pos, max_len
if __name__ == "__main__":
    words, entity2desp, max_len = load_description("FB15k_description/FB15k_mid2description.txt",100);
    emb_words, word2id = load_pretrain_emb('FB15k/glove.6B.100d.txt', words)
    # print(word2id)
    print(len(word2id))
    print(len(emb_words))
    print(emb_words[88511])
    print(word2id['sir'],word2id['reginald'])
