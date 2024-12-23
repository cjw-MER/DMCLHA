# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import gc
import os
import torch
import logging
import random
import numpy as np
from utils_graph import unique_rows

from model import *
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer,RobertaConfig,RobertaTokenizer
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils_graph import unique_rows
import numpy as np, pickle, argparse

import torch
import torch.nn.functional as F
from rgcn import RGCN
from torch_scatter import scatter_add
from torch_geometric.data import Data
from glue_utils import *
from CLLOSS import batched_contrastive_loss_1
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logger = logging.getLogger(__name__)
try:
    from apex import amp
except ImportError:
    amp = None

ALL_MODELS = (
    "roberta-base",
    "roberta-large",
)

MODEL_CLASSES = {
    'roberta_stance' : (RobertaConfig, RobertaStance , RobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
#参数设置
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='model_output')
    parser.add_argument('--graph_dir', type=str, default='utils')
    parser.add_argument("--data_dir", default='labeled_data.csv', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", nargs='+', required=True, 
                        help="Task for stance detection")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--fix_tfm", default=0, type=int, help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default='/home/user/cjh/StanceRel-main/StanceRel-main/roberta', type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="/home/user/cjh/StanceRel-main/StanceRel-main/roberta", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--do_evaluate", action='store_true',
                        help="Whether to run dev.")
    parser.add_argument("--do_dev", action='store_true',
                        help="Whether to run dev.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")#
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for training.")#
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-6, type=float,
                        help="The initial learning rate for Adam.")#1.5e-5#2e-6最好
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")#0.0
    parser.add_argument('--eval_steps', type=int, default=100,
                        help="Evaluate valid set every X steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")#
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")#
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")#
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "constant", "inv_sqrt"])
    parser.add_argument("--optimizer", default="adam", type=str, choices=["adam", "adafactor"])
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization 56 ")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed learning")
    parser.add_argument('--fp16', default=False, action="store_true")
    parser.add_argument("--gpu_n", default=0, type=int,
                        help="gpu.")
    args = parser.parse_args(["--data_dir", "labeled_data.csv", "--model_type", "roberta_stance",  '--graph_dir','utils',        
                              '--model_name_or_path', 'roberta-base',
                              "--do_lower_case",'--do_test', "--data_dir",'labeled_data.csv',
                              "--max_seq_length", '256', "--task", 'Brexit','climate','BlackLivesMatter','Republican','democrats'
                              '--output_dir','RoBERT_rel_1.5e-05'])
    # args = parser.parse_args()
    if args.fp16 and amp is None:
        print("No apex installed, fp16 not used.")
        args.fp16 = False
    print(args)    
    return args
#对原始图中的边进行归一化处理，反映节点之间的关系强度
def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    """
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    """
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)#每个关系类型有两个方向的边
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)#edge_index[0]是源节点的索引，计算了每个节点对应的关系的数量
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)#计算出每条边对应的索引
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]#包含了归一化后的边权重的张量

    return edge_norm
#生成经过图自编码器后的图
def generate_graph(triplets, num_rels):#传入三元组及关系的数量4种
    """
        Get feature extraction graph without negative sampling.
    """
    edges = triplets
    src, rel, dst = edges.transpose()#将输入的三元组triplets分解为源节点src、关系类型rel和目标节点dst
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)#对源节点和目标节点进行去重
    src, dst = np.reshape(edges, (2, -1))#将重新标记后的边索引edges重新分解为源节点src和目标节点dst
    relabeled_edges = np.stack((src, rel, dst)).transpose()
    
    src = torch.tensor(src, dtype = torch.long).contiguous()#将源节点、目标节点和关系类型转换为张量
    dst = torch.tensor(dst, dtype = torch.long).contiguous()
    rel = torch.tensor(rel, dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))#形成双向边
    rel = torch.cat((rel, rel + num_rels))#更新关系 
    edge_index = torch.stack((src, dst))
    edge_type = rel
    
    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)#tensor([2429, 7389])
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    return data#Data(edge_index=[2, 2], entity=[2], edge_type=[2], edge_norm=[2])

def train(args, train_dataset, model,rgcn, rgcn_c, tokenizer, relation_map,concept_graphs,concept_graphs_c,unique_nodes_mapping):
    """ Train the model """

    tb_writer = SummaryWriter(args.output_dir)#记录训练过程中的日志和指标
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)#判断是否是分布式训练
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #训练步数的设置
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']#定义了不进行权重衰减的参数
    optimizer_grouped_parameters = [#指定不同参数组的优化器设置
        {'params'      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #优化器的选择
    if args.optimizer == "adam":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        optimizer2 = AdamW(rgcn.parameters(), lr=1e-2, eps=args.adam_epsilon)
    elif args.optimizer == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False,
                              relative_step=False)

    if args.fp16:#使用混合精度训练
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    #选择不同的学习率调度器（scheduler）
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
    elif args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    else:
        scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # For reproducibility (even between python 2 and 3)
    best_p=0.0
    should_stop = False
    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader,desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()#设置模型和rgcn为训练模式
            rgcn.train()
            graph_feature = torch.zeros((batch[4].shape[0], 100))#创建一个全零的图特征张量
            for j in range(batch[4].shape[0]):
                sd = [batch[4][j],batch[5][j]]#获取当前样本的图信息，batch[4]：源节点的图信息，batch[5]：目标节点的图信息
                if concept_graphs[int(sd[0])]==[]:#根据sd中的节点索引选择对应的概念图
                    xg = np.array(concept_graphs[int(sd[1])])
                elif concept_graphs[int(sd[1])] == []:
                    xg = np.array(concept_graphs[int(sd[0])])
                else:
                    xg = np.concatenate([concept_graphs[int(item)] for item in sd])
                if (xg.shape[0]>0):#处理图结构
                    xg = unique_rows(xg).astype('int64')
                    if len(xg) > 500:
                        xg = xg[:500, :]

                    sg = generate_graph(xg, len(relation_map)).to(args.device)#使用xg和关系映射（relation_map）生成图（sg）
                    features = rgcn(sg.entity, sg.edge_index, sg.edge_type, sg.edge_norm)#[2,100]将实体、边索引、边类型和边标准化值传递给rgcn模型
                    graph_feature[j] = features.mean(axis=0)#在rgcn模型上计算图的特征    
            graph_feature = graph_feature.cuda()#[8,100]

            rgcn_c.train()
            graph_feature_c = torch.zeros((batch[4].shape[0], 100))#创建一个全零的图特征张量
            for j in range(batch[4].shape[0]):
                sd = [batch[4][j],batch[5][j]]#获取当前样本的图信息，batch[4]：源节点的图信息，batch[5]：目标节点的图信息
                if concept_graphs_c[int(sd[0])]==[]:#根据sd中的节点索引选择对应的概念图
                    xg = np.array(concept_graphs_c[int(sd[1])])
                elif concept_graphs_c[int(sd[1])] == []:
                    xg = np.array(concept_graphs_c[int(sd[0])])
                else:
                    xg = np.concatenate([concept_graphs_c[int(item)] for item in sd])
                if (xg.shape[0]>0):#处理图结构
                    xg = unique_rows(xg).astype('int64')
                    if len(xg) > 500:
                        xg = xg[:500, :]
                    sg_c = generate_graph(xg, len(relation_map)).to(args.device)#使用xg和关系映射（relation_map）生成图（sg）
                    features_c = rgcn_c(sg_c.entity, sg_c.edge_index, sg_c.edge_type, sg_c.edge_norm)#将实体、边索引、边类型和边标准化值传递给rgcn模型
                    graph_feature_c[j] = features_c.mean(axis=0)#在rgcn模型上计算图的特征   
            graph_feature_c = graph_feature_c.cuda()
            
            batch = tuple(t.to(args.device) for t in batch)
            loss = torch.tensor(0, dtype=float).to(args.device)
            #构建输入字典
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'sent_labels': batch[3],
                          'graph_feature':graph_feature,
                          'graph_feature_c':graph_feature_c
                          }
            logits,l  = model(**inputs)
            loss=loss
            if args.n_gpu >1:
                loss += l.mean()
            else :
                loss += l

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward(retain_graph=True)
      
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer2.step()
                scheduler.step()  # Update learning rate schedule
                scheduler2.step()
                model.zero_grad()
                rgcn.zero_grad()
                global_step += 1
                if global_step % args.logging_steps == 0 or global_step == 1:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("epoch: {:d}, step: {:d}, "
                                "loss: {:.4f}, lr: {:g}".format(epoch, global_step,
                                                                (tr_loss - logging_loss) / args.logging_steps,
                                                                scheduler.get_lr()[0]))
                    logging_loss = tr_loss

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model,rgcn, rgcn_c, tokenizer, "dev", relation_map, concept_graphs,concept_graphs_c,unique_nodes_mapping)
                    logger.info("macro-f1 {:4f}".format(results['macro-f1']))
                    tb_writer.add_scalar('dev_best_f1', global_step)
                    if best_p < results['macro-f1']:
                        best_p = results['macro-f1']
                        if not os.path.exists(args.output_dir):
                            os.mkdir(args.output_dir)
                        model.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                        torch.save(rgcn.state_dict(),  os.path.join(args.output_dir, '_rgcn.pt'))
                        logger.info("Saving best model checkpoint.")
                        with open(os.path.join(args.output_dir, 'log.txt'), "w") as writer:
                            writer.write("Steps:" + str(global_step) + '\n')
                            for key in sorted(results.keys()):
                                writer.write("%s = %s\n" % (key, str(results[key])))

                    tb_writer.add_scalar('eval_best_p', best_p, global_step)

                if 0 < args.max_steps < global_step:
                    should_stop = True

            if should_stop:
                break
        if should_stop:
            break
    del train_dataloader
    tb_writer.close()
    return best_p

def load_and_cache_examples(args, tokenizer,unique_nodes_mapping, mode='train'):
    processor = StanceDataloader()#数据处理器，根据传入的mode参数确定加载的数据集类型
    print('__________loading '+mode +' dataset__________')
    if mode == 'train':
        examples = processor.get_train_examples(args.data_dir,topic = args.task)
    elif mode == 'dev':
        examples = processor.get_dev_examples(args.data_dir,topic = args.task)
    elif mode == 'test':
        examples = processor.get_test_examples(args.data_dir,topic = args.task)
    elif mode == 'evaluate':
        examples = processor.get_data_examples(args.data_dir,topic = args.task)

    #将获取的示例转换为特征向量
    features = convert_examples_to_features(examples=examples,unique_nodes_mapping=unique_nodes_mapping, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)#将特征向量转换为PyTorch张量
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    all_parents = torch.tensor([f.p for f in features],dtype=torch.long)
    all_childs = torch.tensor([f.c for f in features],dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_parents,all_childs)#创建TensorDataset对象作为数据集
    
    del features 
    gc.collect()
    return dataset

SMALL_POSITIVE_CONST = 1e-4
def compute_metrics_absa(preds, labels):#计算绝对情感分析，preds:表示预测的情感类别,labels：表示真实的情感类别
    errors = []
    #存储预测错误的样本的索引
    #hit_count:存储每个情感类别的预测正确数量
    #gold_count:存储每个情感类别的真实数量
    #pred_count：存储每个情感类别的预测数量
    #ts_precision：存储每个情感类别的精确率（Precision）。
    #ts_recall：存储每个情感类别的召回率（Recall）。
    #ts_f1：存储每个情感类别的F1值
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)

    for u in range(len(preds)):
        pred_count[preds[u]]+=1
        gold_count[labels[u]] +=1
        if preds[u] == labels[u]:
            hit_count[preds[u]]+=1
        else : errors.append(u)
    for i in range(3):
        n_ts = hit_count[i]
        n_g_ts = gold_count[i]
        n_p_ts = pred_count[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)
    macro_f1 = ts_f1.mean()

    n_tp_total = sum(hit_count)#预测正确的总样本数
    # TP + FN
    n_g_total = sum(gold_count)#真实情感类别的总样本数
    n_p_total = sum(pred_count)#预测情感类别的总样本数
    micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + SMALL_POSITIVE_CONST)

    precision = sum(hit_count)/sum(gold_count)
    
    acc = compute_acc(preds, labels)
    scores = {'disagree-f1': ts_f1[0],'disagree-precision': ts_precision[0],'disagree-recall': ts_recall[0],
              'neutral-f1': ts_f1[1],'neutral-precision': ts_precision[1],'neutral-recall': ts_recall[1],
              'agree-f1': ts_f1[2],'agree-precision': ts_precision[2],'agree-recall': ts_recall[2],
              'macro-f1': macro_f1, "micro-f1": micro_f1,"precision":precision, 'acc':acc}
    print(scores)
    return scores,errors

def compute_acc(preds, labels):
    
    cor= 0#预测正确的样本数量
    for u in range(len(preds)):
        if preds[u] == labels[u]:
            cor+=1
    return (cor/len(preds))#计算出预测正确的样本数量，并将其除以总样本数量，得到准确率

def evaluate(args, model, rgcn, rgcn_c, tokenizer, mode,relation_map,concept_graphs, concept_graphs_c, unique_nodes_mapping):

    eval_dataset = load_and_cache_examples(args, tokenizer, mode=mode,unique_nodes_mapping=unique_nodes_mapping)

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss, eval_steps = 0.0, 0
    logit_list, label_list = [],[]
    ids = []
    related = []
    model.eval()
    rgcn.eval()
    rgcn_c.eval()

    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            graph_feature = torch.zeros((batch[4].shape[0], 100))
            for j in range(batch[4].shape[0]):
                sd = [batch[4][j],batch[5][j]]
                if concept_graphs[int(sd[0])]==[]:
                    xg = np.array(concept_graphs[int(sd[1])])
                elif concept_graphs[int(sd[1])] == []:
                    xg = np.array(concept_graphs[int(sd[0])])
                else:
                    xg = np.concatenate([concept_graphs[int(item)] for item in sd])                  
                    
                if (xg.shape[0]>0):
                    xg = unique_rows(xg).astype('int64')
                    if len(xg) > 500:
                        xg = xg[:500, :]

                    sg = generate_graph(xg, len(relation_map)).to(args.device)
                    features = rgcn(sg.entity, sg.edge_index, sg.edge_type, sg.edge_norm)
                    graph_feature[j] = features.mean(axis=0)   
                             
            graph_feature = graph_feature.cuda()
            
            graph_feature_c = torch.zeros((batch[4].shape[0], 100))#创建一个全零的图特征张量
            for j in range(batch[4].shape[0]):
                sd = [batch[4][j],batch[5][j]]#获取当前样本的图信息，batch[4]：源节点的图信息，batch[5]：目标节点的图信息
                if concept_graphs_c[int(sd[0])]==[]:#根据sd中的节点索引选择对应的概念图
                    xg = np.array(concept_graphs_c[int(sd[1])])
                elif concept_graphs_c[int(sd[1])] == []:
                    xg = np.array(concept_graphs_c[int(sd[0])])
                else:
                    xg = np.concatenate([concept_graphs_c[int(item)] for item in sd])
                if (xg.shape[0]>0):#处理图结构
                    xg = unique_rows(xg).astype('int64')
                    if len(xg) > 500:
                        xg = xg[:500, :]
                    sg_c = generate_graph(xg, len(relation_map)).to(args.device)#使用xg和关系映射（relation_map）生成图（sg）
                    features_c = rgcn_c(sg_c.entity, sg_c.edge_index, sg_c.edge_type, sg_c.edge_norm)#将实体、边索引、边类型和边标准化值传递给rgcn模型
                    graph_feature_c[j] = features_c.mean(axis=0)#在rgcn模型上计算图的特征   
            
            graph_feature_c = graph_feature_c.cuda()

            eval_loss = torch.tensor(0, dtype=float).to(args.device)

            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'sent_labels': batch[3],
                         'graph_feature':graph_feature,
                         'graph_feature_c':graph_feature_c,
                          }

            ids.extend(batch[0])
            logits,l  = model(**inputs)
            eval_loss += l
            logit_list.append(torch.argmax(logits, axis=-1))#将预测的logits添加到logit_list中
            label_list.append(batch[3])#将真实标签添加到label_list中
        eval_steps += 1
    preds = torch.cat(logit_list, axis=0)
    labels =torch.cat(label_list, axis=0)


    result,errors = compute_metrics_absa(preds, labels)

    return result

def main():#程序入口
    args = init_args()#该对象包含了程序运行时使用的各种参数

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.cuda.set_device(args.gpu_n)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu =0
    
    print("GPU number is : ~~~~~~~~~~~~~~~~  "+ str(args.n_gpu))
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(args)

    args.model_type = args.model_type.lower()#将模型类型转换为小写字母

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          output_hidden_states=True)
    config.sent_number = 3#情感类别
    if len(args.task)==5:#设置任务类型为cross-domain实验
        args.task='all'
    # tokenizer = tokenizer_class.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #     do_lower_case=args.do_lower_case, cache_dir='./cache')
    tokenizer = RobertaTokenizer.from_pretrained("/home/bigdata10/cjh/StanceRel-main/roberta")#加载预训练的RoBERTa分词器
    #加载关系映射、唯一节点映射和概念图
    relation_map = pickle.load(open(args.graph_dir+'/relation_n_map_bert.pkl', 'rb'))
    unique_nodes_mapping = pickle.load(open(args.graph_dir+'/unique_nodes_n_mapping_bert.pkl', 'rb'))
    concept_graphs = pickle.load(open(args.graph_dir+'/concept_n_graphs_bert.pkl', 'rb'))
    concept_graphs_c = pickle.load(open(args.graph_dir+'/concept_n_graphs_c_bert.pkl', 'rb'))

    if args.do_train:#进行模型训练
        rgcn = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=4, dropout=0.25)
        rgcn.load_state_dict(torch.load('weights/model_n_epoch_' + str(1800) +'.pt'))
        rgcn_c = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=4, dropout=0.25)
        rgcn_c.load_state_dict(torch.load('weights/model_n_c_epoch_' + str(1800) +'.pt'))
        rgcn.to(args.device)#将 rgcn 模型移到指定的设备上
        rgcn_c.to(args.device)
    
        '''model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config, cache_dir='./cache')'''
        model = model_class.from_pretrained("/home/bigdata10/cjh/StanceRel-main/roberta")
        model.to(args.device)
        
        if args.n_gpu >1:
            model = torch.nn.DataParallel(model)
        train_dataset = load_and_cache_examples(args, tokenizer,unique_nodes_mapping, mode='train')
        
        train(args,train_dataset,model,rgcn,rgcn_c,tokenizer,relation_map,concept_graphs,concept_graphs_c,unique_nodes_mapping)
        del train_dataset 
        gc.collect()
        del model
        torch.cuda.empty_cache()

    if args.do_test:#进行模型测试 
        rgcn = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=4, dropout=0.25)
        rgcn.load_state_dict(torch.load(os.path.join(args.output_dir, '_rgcn.pt')))
        rgcn.to(args.device)

        args.model_type = args.model_type.lower()
        r = 0
        with open (args.output_dir+'/test_results.txt','w') as f:
            model = model_class.from_pretrained(args.output_dir)
            model.to(args.device)

            results = evaluate(args, model, rgcn, rgcn_c,tokenizer, 'test',relation_map,concept_graphs,concept_graphs_c, unique_nodes_mapping)
            f.write("results "+str(results)+'\n')
            print('result',results)
        del model
        torch.cuda.empty_cache()
        
    if args.do_evaluate:#ATAi1234
    
        rgcn = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=4, dropout=0.25)
        rgcn.load_state_dict(torch.load(os.path.join(args.output_dir, '_rgcn.pt')))
        rgcn.to(args.device)

        args.model_type = args.model_type.lower()
        r = 0
        with open (args.output_dir+'/test_results.txt','w') as f:
            model = model_class.from_pretrained(args.output_dir)
            model.to(args.device)

            results = evaluate(args, model,rgcn, tokenizer, 'evaluate',relation_map,concept_graphs,unique_nodes_mapping)
            f.write("results "+str(results)+'\n')
        del model
        torch.cuda.empty_cache()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()