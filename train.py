import sys
import os
import torch
import argparse


from tqdm.std import trange
sys.path.append('./lib')
from bert import BERTLM
from treelstm import TreeLSTM
from kvbert import myModel
from kvbert import TreeArgs
from treelstm import treeVocab
import numpy as np
from google_bert import BasicTokenizer
from treelstm import Tree
from treelstm import Constants
from data_loader import DataLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
# from pytorch_pretrained_bert.optimization import BertAdam
import torch.optim as optim
from sklearn import metrics



def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path)
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    tree_args = model_ckpt['tree_args']
    tree_vocab = model_ckpt['tree_vocab']
    return bert_args, model_args, bert_vocab, model_parameters, tree_args, tree_vocab

def init_empty_bert_model(bert_args, bert_vocab, gpu_id, approx = 'none'):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, approx)
    return bert_model

def init_empty_tree_model(t_args, tree_vocab, gpuid):
    tree_model = TreeLSTM(tree_vocab.size(), t_args.input_dim, t_args.mem_dim, t_args.hidden_dim, t_args.num_classes, t_args.freeze_embed)
    tree_model = tree_model.cuda(gpuid)
    return tree_model


def init_sequence_classification_model(empty_bert_model, args, bert_args, gpu_id, bert_vocab, model_parameters, empty_tree_model, tree_args):
    number_class = args.number_class
    number_category = 3
    embedding_size = bert_args.embed_dim
    batch_size = args.batch_size
    dropout = args.dropout
    tree_hidden_dim = tree_args.hidden_dim
    device = gpu_id
    vocab = bert_vocab
    seq_tagging_model = myModel(empty_bert_model, number_class, number_category, embedding_size, batch_size, dropout, device, vocab, empty_tree_model, tree_hidden_dim)
    return seq_tagging_model

def init_sequence_classification_model_with_dict(empty_bert_model, args, bert_args, gpu_id, bert_vocab, model_parameters, empty_tree_model, tree_args):
    number_class = args.number_class
    number_category = 3
    embedding_size = bert_args.embed_dim
    batch_size = args.batch_size
    dropout = args.dropout
    tree_hidden_dim = tree_args.hidden_dim
    device = gpu_id
    vocab = bert_vocab
    seq_tagging_model = myModel(empty_bert_model, number_class, number_category, embedding_size, batch_size, dropout, device, vocab, empty_tree_model, tree_hidden_dim)
    seq_tagging_model.load_state_dict(model_parameters)
    return seq_tagging_model

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--out_path',type=str)
    parser.add_argument('--gpu_id',type=int, default=0)
    parser.add_argument('--train_epoch',type=int, default=3)
    parser.add_argument('--train_batch_size',type=int, default=32)
    parser.add_argument('--dev_batch_size',type=int, default=32)
    parser.add_argument('--learning_rate',type=float, default=2e-5)
    parser.add_argument('--do_train',action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval',action='store_true', help='Whether to run eval.')
    parser.add_argument('--do_test',action='store_true', help='Whether to run test.')
    parser.add_argument('--no_cuda',action='store_true', default=False, help='Whether to gpu.')
    parser.add_argument('--output_dir',type=str, default='saved_model')

    return parser.parse_args()

def segment(text):
    seg = [1 for _ in range(len(text))]
    idx = text.index("sep")
    seg[:idx] = [0 for _ in range(idx)]
    return seg

def profile(text):
    seg = [3 for _ in range(len(text))]
    loc_idx = text.index("loc") - 1
    gender_idx = text.index("gender") - 1
    sep_idx = text.index("sep")
    seg[:loc_idx] = [0 for _ in range(loc_idx)]
    seg[loc_idx:gender_idx] = [1 for _ in range(gender_idx-loc_idx)]
    seg[gender_idx:sep_idx] = [2 for _ in range(sep_idx-gender_idx)]
    return seg


def read_tree(line):
    parents = list(map(int, line.split()))
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    return root


def seq_cut(seq, max_len):
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq


def read_sentence(line, vocab):
    indices = vocab.convertToIdx(line, Constants.UNK_WORD)
    return torch.LongTensor(indices)


def init_model(ckpt_path):
    bert_args, model_args, bert_vocab, model_parameters, tree_args, tree_vocab = extract_parameters(ckpt_path)
    empty_bert_model = init_empty_bert_model(bert_args, bert_vocab, gpu_id, approx='none')
    empty_tree_model = init_empty_tree_model(tree_args, tree_vocab, gpu_id)
    seq_classification_model = init_sequence_classification_model(empty_bert_model, model_args, bert_args, gpu_id, bert_vocab, model_parameters, empty_tree_model, tree_args)
    return seq_classification_model, tree_vocab

def eval(model, data_loader, device):
    batch_num = int(data_loader.dev_num/args.dev_batch_size)
    batch_num = batch_num if data_loader.dev_num% args.dev_batch_size == 0 else batch_num + 1
    predict_all = np.array([], dtype=int)
    label_all = np.array([], dtype=int)
    loss_mean = 0
    
    with torch.no_grad():
        for step in trange(batch_num, desc=f'valid {step}/{batch_num}'):
            batch_data = data_loader.get_next_batch(args.dev_batch_size, 'dev')
            batch_text_list, batch_label_list, batch_seg_list, batch_type_list, batch_category_list, \
            batch_a_seg_list, batch_a_tree_list, batch_b_seg_list, batch_b_tree_list = batch_data
            batch_label_ids = torch.tensor(batch_label_list, dtype=torch.long).to(device)
            pred_output = model(batch_text_list, batch_seg_list, batch_type_list, batch_a_seg_list, batch_a_tree_list, batch_b_seg_list, batch_b_tree_list, fine_tune=True)
            logits = pred_output[0]
            loss = criterion(logits.view(-1, label_nums), batch_label_ids.view(-1))
            loss_mean += torch.sum(loss)
            predict = torch.max(logits.data, 1)[1].cpu().numpy()
            label_all = np.append(label_all, batch_label_ids.data.cpu().numpy())
            predict_all = np.append(predict_all, predict)
        acc = metrics.accuracy_score(label_all, predict_all)
        loss_mean /= data_loader.dev_num
    return acc, loss_mean


def predict(model, data_loader, device, is_train=False):
    model.eval()
    if is_train:
        model.train() 
     
    batch_data = data_loader.get_next_batch(args.train_batch_size, 'train' if is_train else 'dev')
    batch_text_list, batch_label_list, batch_seg_list, batch_type_list, batch_category_list, \
    batch_a_seg_list, batch_a_tree_list, batch_b_seg_list, batch_b_tree_list = batch_data
    batch_label_ids = torch.tensor(batch_label_list, dtype=torch.long).to(device)
    pred_output = model(batch_text_list, batch_seg_list, batch_type_list, batch_a_seg_list, batch_a_tree_list, batch_b_seg_list, batch_b_tree_list, fine_tune=True)
    logits = pred_output[0]
    loss = criterion(logits.view(-1, label_nums), batch_label_ids.view(-1))
    return logits, loss


if __name__ == '__main__':
    args = parse_config()
    ckpt_path = args.ckpt_path
    test_data = args.test_data
    out_path = args.out_path
    gpu_id = args.gpu_id

    model, tree_vocab = init_model(ckpt_path)
    model.cuda(gpu_id)
    tokenizer = BasicTokenizer()

    if args.do_train:
        train_path = 'data/KvPI_train.txt'
        dev_path = 'data/KvPI_valid.txt'
        data_loader = DataLoader(train_path, dev_path, tree_vocab, args.max_len)
        criterion = CrossEntropyLoss()
        label_nums = model.num_class
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        optimizer = optim.AdamW(model.parameters(), lr=3e-5)

        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        # optimizer = BertAdam(optimizer_grouped_parameters,
        #                         lr=args.learning_rate,
        #                         warmup=args.warmup_proportion,
        #                         t_total=num_train_optimization_steps)

        global_step = 0
        best_dev_acc = 0.0
        for epoch in trange(int(args.train_epoch), desc='Epoch'):
            model.train()
            batch_num = int(data_loader.train_num/args.train_batch_size)
            batch_num = batch_num if data_loader.train_num % args.train_batch_size == 0 else batch_num + 1
            train_loss = 0
            for step in trange(batch_num, desc=f'Training {step}/{batch_num}'):
                logits, loss = predict(model, data_loader, device, is_train=True)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 100 == 0:
                    dev_acc, loss = eval(model, data_loader, device)
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        if best_dev_acc > 0.8:
                            output_model_file = os.path.join(args.output_dir, str(global_step)+"_pytorch_model.bin")
                            torch.save(model.state_dict(), output_model_file)
                    print(f'global step:{global_step}, train loss:{loss}, best dev acc {best_dev_acc},current dev acc {dev_acc}, dev loss {loss}')
            print(f'epoch :{epoch} training done, train mean loss:{train_loss/data_loader.train_num}')


    with torch.no_grad():
        with open(out_path, 'w', encoding='utf8') as o:
            with open(test_data, 'r', encoding='utf8') as i:
                lines = i.readlines()
                for l in tqdm(lines[1:], desc='Predicting'):
                    content_list = l.strip().split('\t')
                    text = content_list[0]
                    text_tokenized_list = tokenizer.tokenize(text)
                    if len(text_tokenized_list) > args.max_len:
                        text_tokenized_list = text_tokenized_list[:args.max_len]
                    seg_list = segment(text_tokenized_list)
                    typ_list = profile(text_tokenized_list)

                    a_seg = read_sentence(seq_cut(content_list[3].split(' '), args.max_len), tree_vocab)
                    a_tree = read_tree(content_list[4])
                    b_seg = read_sentence(seq_cut(content_list[5].split(' '), args.max_len), tree_vocab)
                    b_tree = read_tree(content_list[6])

                    pred_output = model([text_tokenized_list], [seg_list], [typ_list], [a_seg], [a_tree], [b_seg], [b_tree], fine_tune=False)[0].cpu().numpy()
                    pred_probability = pred_output[0]
                    pred_label = np.argmax(pred_probability)
                    out_line = text + '\t' + str(pred_label)
                    o.writelines(out_line + '\n')
    print("done.")
