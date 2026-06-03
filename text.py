import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import gzip
import html
import os
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import ftfy
import regex as re
import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Text_Net(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(torch.float32)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)  # torch.Size([77, 6, 512])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(torch.float32)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:

            batch = [[i+j*self.num_per_class for i in torch.randperm(self.num_per_class)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_per_class for i in range(self.num_per_class)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

class ClipLoss(torch.nn.Module):

    def __init__(self, logit_scale=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  #####
        # self.logit_scale = logit_scale

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        ## cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, 2, dim=1)
        text_features = F.normalize(text_features, 2, dim=1)
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2

        return total_loss


class Queue:
    def __init__(self, capacity, dim):
        self.capacity = capacity
        self.size = 0
        self.queue = torch.empty((capacity, dim), dtype=torch.float32)
        self.front = 0
        self.rear = -1

    def is_full(self):
        return self.size == self.capacity

    def is_empty(self):
        return self.size == 0

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full")
            return
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item

    def peek(self):
        if self.is_empty():
            print("Queue is empty")
            return
        return self.queue[self.front]

    def display(self):
        if self.is_empty():
            print("Queue is empty")
            return
        if self.front <= self.rear:
            print(self.queue[self.front:self.rear+1])
        else:
            print(torch.cat((self.queue[self.front:], self.queue[:self.rear+1])))

    def get_queue(self):
        return self.queue

    def get_num(self):
        return self.size

def get_loss_clip(capacity, dim_feature, label, features, text_features):
    loss_clip_all = 0
    Loss = ClipLoss(logit_scale=5)
    queue_0 = Queue(capacity=capacity, dim=dim_feature)
    queue_1 = Queue(capacity=capacity, dim=dim_feature)
    queue_2 = Queue(capacity=capacity, dim=dim_feature)
    queue_3 = Queue(capacity=capacity, dim=dim_feature)
    queue_4 = Queue(capacity=capacity, dim=dim_feature)
    queue_5 = Queue(capacity=capacity, dim=dim_feature)
    queue_6 = Queue(capacity=capacity, dim=dim_feature)
    queue_7 = Queue(capacity=capacity, dim=dim_feature)
    queue_8 = Queue(capacity=capacity, dim=dim_feature)
    queue_9 = Queue(capacity=capacity, dim=dim_feature)
    queue_10 = Queue(capacity=capacity, dim=dim_feature)
    queue_11 = Queue(capacity=capacity, dim=dim_feature)
    queue_12 = Queue(capacity=capacity, dim=dim_feature)
    queue_13 = Queue(capacity=capacity, dim=dim_feature)
    queue_14 = Queue(capacity=capacity, dim=dim_feature)
    num=np.zeros(15)
    for n, vaule in enumerate(label):
        num[vaule]=num[vaule] + 1
        if (vaule == 0):
            queue_0.enqueue(features[n, :])  # 还是假定每个块里各个类别是相同数量的
        elif (vaule == 1):
            queue_1.enqueue(features[n, :])
        elif (vaule == 2):
            queue_2.enqueue(features[n, :])
        elif (vaule == 3):
            queue_3.enqueue(features[n, :])
        elif (vaule == 4):
            queue_4.enqueue(features[n, :])
        elif (vaule == 5):
            queue_5.enqueue(features[n, :])
        elif (vaule == 6):
            queue_6.enqueue(features[n, :])
        elif (vaule == 7):
            queue_7.enqueue(features[n, :])
        elif (vaule == 8):
            queue_8.enqueue(features[n, :])
        elif (vaule == 9):
            queue_9.enqueue(features[n, :])
        elif (vaule == 10):
            queue_10.enqueue(features[n, :])
        elif (vaule == 11):
            queue_11.enqueue(features[n, :])
        elif (vaule == 12):
            queue_12.enqueue(features[n, :])
        elif (vaule == 13):
            queue_13.enqueue(features[n, :])
        elif (vaule == 14):
            queue_14.enqueue(features[n, :])
    # print(num)
    for n in range(capacity):
        temp = torch.empty((1, dim_feature))
        temp = torch.unsqueeze(queue_0.dequeue(), dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_1.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_2.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_3.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_4.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_5.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_6.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_7.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_8.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_9.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_10.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_11.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_12.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_13.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_14.dequeue(), dim=0)], dim=0).cuda(0)
        loss_clip_all += Loss(image_features=temp, text_features=text_features)

    return loss_clip_all


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    p_output = F.normalize(p_output, p=2, dim=1)
    q_output = F.normalize(q_output, p=2, dim=1)
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output, dim=-1)
        q_output = F.softmax(q_output, dim=-1)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def get_loss_clip_trento(capacity, dim_feature, label, features, text_features):
    loss_clip_all = 0
    Loss = ClipLoss(logit_scale=5)
    queue_0 = Queue(capacity=capacity, dim=dim_feature)
    queue_1 = Queue(capacity=capacity, dim=dim_feature)
    queue_2 = Queue(capacity=capacity, dim=dim_feature)
    queue_3 = Queue(capacity=capacity, dim=dim_feature)
    queue_4 = Queue(capacity=capacity, dim=dim_feature)
    queue_5 = Queue(capacity=capacity, dim=dim_feature)
    for n, vaule in enumerate(label):
        if (vaule == 0):
            queue_0.enqueue(features[n, :])  # 还是假定每个块里各个类别是相同数量的
        elif (vaule == 1):
            queue_1.enqueue(features[n, :])
        elif (vaule == 2):
            queue_2.enqueue(features[n, :])
        elif (vaule == 3):
            queue_3.enqueue(features[n, :])
        elif (vaule == 4):
            queue_4.enqueue(features[n, :])
        elif (vaule == 5):
            queue_5.enqueue(features[n, :])
    for n in range(capacity):
        temp = torch.empty((1, dim_feature))
        temp = torch.unsqueeze(queue_0.dequeue(), dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_1.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_2.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_3.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_4.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_5.dequeue(), dim=0)], dim=0)
        temp = temp.to(features.device)
        loss_clip_all += Loss(image_features=temp, text_features=text_features)

    return loss_clip_all


def get_loss_clip_augsburg(capacity, dim_feature, label, features, text_features):
    loss_clip_all = 0
    Loss = ClipLoss(logit_scale=5)
    queue_0 = Queue(capacity=capacity, dim=dim_feature)
    queue_1 = Queue(capacity=capacity, dim=dim_feature)
    queue_2 = Queue(capacity=capacity, dim=dim_feature)
    queue_3 = Queue(capacity=capacity, dim=dim_feature)
    queue_4 = Queue(capacity=capacity, dim=dim_feature)
    queue_5 = Queue(capacity=capacity, dim=dim_feature)
    queue_6 = Queue(capacity=capacity, dim=dim_feature)
    for n, vaule in enumerate(label):
        if (vaule == 0):
            queue_0.enqueue(features[n, :])  # 还是假定每个块里各个类别是相同数量的
        elif (vaule == 1):
            queue_1.enqueue(features[n, :])
        elif (vaule == 2):
            queue_2.enqueue(features[n, :])
        elif (vaule == 3):
            queue_3.enqueue(features[n, :])
        elif (vaule == 4):
            queue_4.enqueue(features[n, :])
        elif (vaule == 5):
            queue_5.enqueue(features[n, :])
        elif (vaule == 6):
            queue_6.enqueue(features[n, :])
    for n in range(capacity):
        temp = torch.empty((1, dim_feature))
        temp = torch.unsqueeze(queue_0.dequeue(), dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_1.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_2.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_3.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_4.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_5.dequeue(), dim=0)], dim=0)
        temp = torch.cat([temp, torch.unsqueeze(queue_6.dequeue(), dim=0)], dim=0).cuda(0)
        loss_clip_all += Loss(image_features=temp, text_features=text_features)

    return loss_clip_all


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = SimpleTokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
       # model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.

        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
