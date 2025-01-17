# 深度学习库pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch.utils.data import TensorDataset


class Tokenizer:
    """
    词典编码器
    """
    UNKNOWN = "<unknown>"
    PAD = "<pad>"
    BOS = "<bos>" 
    EOS = "<eos>" 

    def __init__(self, tokens):
        # 补上特殊词标记：未知词标记、填充字符标记、开始标记、结束标记
        tokens = [Tokenizer.UNKNOWN, Tokenizer.PAD, Tokenizer.BOS, Tokenizer.EOS] + tokens
        # 词汇表大小
        self.dict_size = len(tokens)
        # 生成映射关系
        self.token_id = {} # 映射: 词 -> 编号
        self.id_token = {} # 映射: 编号 -> 词
        for idx, word in enumerate(tokens):
            self.token_id[word] = idx
            self.id_token[idx] = word
        
        # 各个特殊标记的编号id，方便其他地方使用
        self.unknown_id = self.token_id[Tokenizer.UNKNOWN]
        self.pad_id = self.token_id[Tokenizer.PAD]
        self.bos_id = self.token_id[Tokenizer.BOS]
        self.eos_id = self.token_id[Tokenizer.EOS]
    
    def id_to_token(self, token_id):
        """
        编号 -> 词
        """
        return self.id_token.get(token_id)

    def token_to_id(self, token):
        """
        词 -> 编号，取不到时给 UNKNOWN
        """
        return self.token_id.get(token, self.unknown_id)

    def encode(self, tokens):
        """
        词列表 -> <bos>编号 + 编号列表 + <eos>编号
        """
        token_ids = [self.bos_id, ] # 起始标记
        # 遍历，词转编号
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        token_ids.append(self.eos_id) # 结束标记
        return token_ids

    def decode(self, token_ids):
        """
        编号列表 -> 词列表(去掉起始、结束标记)
        """
        tokens = []
        for idx in token_ids:
            # 跳过起始、结束标记
            if idx != self.bos_id and idx != self.eos_id:
                tokens.append(self.id_to_token(idx))
        return tokens
    
    def __len__(self):
        return self.dict_size


def index2onehot(word_ids, vocab_size, device):
    r"""
    由索引转化为独热编码
    Args:
        word_ids (torch.Tensor): 
            A 1D or 2D tensor containing word indices. 
            (seq_len, ) or (batch_size, seq_len)

        vocab_size (int): 
            The size of the vocabulary.

    Returns: 
        torch.Tensor: 
            A tensor containing one-hot encoded vectors.
            (seq_len, vocab_size) or (batch_size, seq_len, vocab_size)

    Raises:
        ValueError: If `word_ids` is not a 1D or 2D tensor.
    """
    if word_ids.dim() == 1:
        # 一维情况：(seq_len,)
        onehot_tensor = torch.zeros(len(word_ids), vocab_size, device=device)
        for i, s in enumerate(word_ids): 
            onehot_tensor[i, s] = 1
    elif word_ids.dim() == 2:
        # 二维情况：(batch_size, seq_len)
        batch_size, seq_len = word_ids.size()
        onehot_tensor = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32, device = device)
        onehot_tensor.scatter_(2, word_ids.unsqueeze(2), 1)
    else:
        raise ValueError("word_ids must be a 1D or 2D tensor")
    return onehot_tensor

def onehot2index(word_ids):
    """
    独热编码转化为索引 (*, vocab_size) ---> (*,)
    """
    return torch.argmax(word_ids, dim=-1)


class MyDataset(TensorDataset):
    """
    数据集定义
    """
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len  # 每条数据的最大长度
        
    def __getitem__(self, index):
        """
        将文本转化为索引并返回
        """
        line = self.data[index]
        word_ids = self.encode_pad_line(line)
        return torch.tensor(word_ids)
    
    def __len__(self):
        return len(self.data)
    
    def encode_pad_line(self, line):
        """
        将文本转化为索引并返回，对齐序列长度为`max_len`
        """
        word_ids = self.tokenizer.encode(line)
        # 如果句子长度不足max_len，填充PAD；超过max_len，截断
        if len(word_ids) <= self.max_len:
            word_ids = word_ids + [self.tokenizer.pad_id] * (self.max_len - len(word_ids))
        else:
            word_ids = word_ids[:self.max_len - 1].append(self.tokenizer.eos_id)
        return word_ids

class Embedding(nn.Module):
    """
    嵌入层 将索引转化为独热向量，并线性嵌入
    """
    def __init__(self, v, h, device):
        """
        v: 词汇表大小
        h: 嵌入后维度
        """
        super().__init__()
        self.embedding = nn.Linear(v, h)
        self.h = h
        self.v = v
        self.device = device

    def forward(self, src):
        # print(src.size())
        onehot_tensor = index2onehot(src, self.v, device = self.device)
        # print(onehot_tensor.size())
        return self.embedding(onehot_tensor)

class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, h, dropout=0.1, max_len=64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, h)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, h, 2).float() * (-math.log(10000.0) / h))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.size(), self.pe.size())
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        return self.dropout(x)

class Attention(nn.Module):
    """
    注意力模块，提供 q = k = v 的自注意力模式和 k = v 的交叉注意力模式
    """
    def __init__(self, h, a, dropout=0.1, type = 'self', device = 'cpu'):
        '''
        h: 嵌入层维度
        a: 注意力头数
        d_k: 每个注意力头的第二个维度 d_k = h//a

        X: (s,h) ---Wq, Wk, Wv: (h, h//a) ---> Q,K,V: (s, h//a) 

            ---> softmax(Q * K.t / sqrt(d_k)) * V: (s, h//a)

            ---> output: (s, h) ---out_proj: (h, h)---> output: (s, h)
        '''
        super().__init__()  # 注意这里的修正，使用super()而不是super.__init__()
        self.h = h
        self.a = a
        self.d_k = h // a
        self.types = type
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        
        # 初始化Q, K, V的权重矩阵
        # 每个权重矩阵的维数是(s, h//a) 这里是(h, h)，是将每个头的相应矩阵拼接到一起了
        self.Wq = nn.Linear(h, h)
        self.Wk = nn.Linear(h, h)
        self.Wv = nn.Linear(h, h)
        
        # 缩放因子，用于缩放点积结果
        self.scale = 1 / math.sqrt(self.d_k)

        self.out_proj = nn.Linear(h, h)

    def forward(self, x, y = None, padding_mask=None, tgt_sequence_mask = None):
        """
        x: (batch_size, s = tgt_s, h), 自注意力的q k v
        y: (batch_size, s = src_s, h), 交叉注意力的k v
        tgt_sequence_mask: (tgt_s, tgt_s)
        padding_mask : (batch_size, src_s)
        padding_mask: 添加给key的掩码，用于掩盖pad的影响
        tgt_sequence_mask: decoder自注意力添加给key的掩码，用于遮蔽未来信息

        Step #1 通过线性变换得到Q, K, V
        q,k,v: (batch_size, s, h) ---> (batch_size, s, a, d_k) ---> (batch_size, a, s, d_k)
        crros attention时q的s=tgt_s, kv的s=src_s

        Step#2 应用掩码，计算注意力分数
        k: (batch_size, a, src_s, d_k) ---> (batch_size, a, d_k, src_s)
        tgt_sequence_mask: (tgt_s, tgt_s) ---> (batch_size, a, tgt_s, tgt_s)
        padding_mask : (batch_size, src_s) ---> (batch_size, a, tgt_s, src_s)
        """
        batch_size = x.size(0)
        """
        Step #1 通过线性变换得到Q, K, V
        q,k,v: (batch_size, s, h) ---> (batch_size, s, a, d_k) ---> (batch_size, a, s, d_k)
        crros attention时q的s=tgt_s, kv的s=src_s
        """
        if self.types == 'self':            # 自注意力机制，均来自输入x            
            assert y is None, ("Self Attention but different input for Q K V")
            q = k = v = x
        elif self.types == 'cross':         # 交叉注意力机制，q来自x，k v来自y
            assert y is not None, ("Cross Attention but the same input for Q K V")
            q = x
            k = v = y
        else: raise ValueError("Undefined Attention Type")

        q = self.Wq(q).view(batch_size, -1, self.a, self.d_k).transpose(1, 2)
        k = self.Wk(k).view(batch_size, -1, self.a, self.d_k).transpose(1, 2)
        v = self.Wv(v).view(batch_size, -1, self.a, self.d_k).transpose(1, 2)

        """
        Step#2 应用掩码，计算注意力分数
        k: (batch_size, a, src_s, d_k) ---> (batch_size, a, d_k, src_s)
        tgt_sequence_mask: (tgt_s, tgt_s) ---> (batch_size, a, tgt_s, tgt_s)
        padding_mask : (batch_size, src_s) ---> (batch_size, a, tgt_s, src_s)
        """
        k_len  = k.size()[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if padding_mask is not None:
            # print(padding_mask)
            mask = padding_mask.view(batch_size, 1, 1, k_len).expand(batch_size, self.a, q.size()[2], k_len).to(self.device)
            if tgt_sequence_mask is not None: 
                assert self.types == 'self' , \
                        (f"Only Self Attention in Decoder Needs Sequence Mask, but now {self.types} attetion!")
                s_mask = tgt_sequence_mask.view(1, 1, k_len, k_len).   \
                expand(batch_size, self.a, -1, -1)
                mask = s_mask.logical_or(mask)
            # print(mask.size(), scores.size())
            # print(mask)
            scores = scores.masked_fill(mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.h)
        output = self.out_proj(output)
        return output

class FeedForward(nn.Module):
    """
    FeedForward层，默认hiddenDim = 4 * h
    """
    def __init__(self, h, hiddenDim = None, outDim = None, dropout = 0.1, type = 'relu'):
        """
        h: 嵌入层维度
        hiddenDim: 隐层维度，默认4*h
        outDim: 输出维度，默认h
        """
        super().__init__()
        self.h = h
        if hiddenDim is None: hiddenDim = 4 * h
        if outDim is None: outDim = h
        self.W1 = nn.Linear(h, hiddenDim)
        self.dropout = nn.Dropout(dropout)
        self.W2 = nn.Linear(hiddenDim, outDim)
        self.types = type
    
    def forward(self, x):
        """
        W1: (h, hiddenDim)
        W2: (hiddenDim, outDim)
        x: (h, h) ---> x * W_1: (h, hiddenDim) ---> relu/gelu: (h, hiddenDim) ---> A' * W2: (h, outDim)
        """
        x = self.W1(x)
        if self.types == 'relu': x = F.relu(x)
        elif self.types == 'gelu': x = F.gelu(x)
        else: raise ValueError("Unsupported activation type")
        x = self.dropout(x)
        x = self.W2(x)
        return x

class LayerNorm(nn.Module):
    """
    层归一化
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input):
        """
        层归一化，计算input最后一个维度均值和方差并标准化
        input: (*, h) ---> (*, h)
        """
        # 计算均值和方差
        assert self.normalized_shape[0] == input.size()[-1], ("Unmatched Shape.")
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)
        
        # 应用层归一化公式
        normalized_input = (input - mean) / std
        normalized_input = normalized_input * self.weight + self.bias
        
        return normalized_input

class TransformerEncoderDecoder(nn.Module):
    """
    Transformer中解码器和编码器架构
    """
    def __init__(self, h, a, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1, device = 'cpu'):
        """
        h: 输入维度
        a: 注意力头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        """
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                Attention(h, a, dropout, device = device),
                LayerNorm((h,)),
                FeedForward(h, dropout = dropout),
                LayerNorm((h,))
            ]) for _ in range(num_encoder_layers)
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                Attention(h, a, dropout, device = device),
                LayerNorm((h,)),
                Attention(h, a, dropout, type='cross', device = device),
                LayerNorm((h,)),
                FeedForward(h, dropout = dropout),
                LayerNorm((h,))
            ]) for _ in range(num_decoder_layers)
        ])

    def forward(self, encoder_input, decoder_input, src_padding_mask=None, tgt_padding_mask = None, tgt_sequence_mask=None, norm_first = False):
        """
        Transformer前向传播
        encoder_input: 编码器输入
        decoder_input: 解码器输入
        src_padding_mask: 编码器pad掩码
        tgt_padding_mask: 解码器pad掩码
        tgt_sequence_mask: 解码器自注意力序列掩码，用于遮蔽未来信息
        norm_first: 是否调整LN结构，如果为True，则先进行归一化和相应计算，再进行残差连接
        """
        for enc in self.encoders:
            attention, norm1, ff, norm2 = enc
            if not norm_first:
                encoder_input = norm1(attention(encoder_input, padding_mask=src_padding_mask) + encoder_input)
                encoder_input = norm2(ff(encoder_input) + encoder_input)
            else:
                encoder_input = attention(norm1(encoder_input), padding_mask=src_padding_mask) + encoder_input
                encoder_input = ff(norm2(encoder_input)) + encoder_input

        for dec in self.decoders:
            self_attention, norm1, cross_attention, norm2, ff, norm3 = dec
            if not norm_first:
                decoder_input = norm1(self_attention(decoder_input, padding_mask=tgt_padding_mask, \
                                                    tgt_sequence_mask = tgt_sequence_mask) + decoder_input)
                decoder_input = norm2(cross_attention(decoder_input, encoder_input, \
                                                    padding_mask=src_padding_mask) + decoder_input)
                decoder_input = norm3(ff(decoder_input) + decoder_input)
            else:
                decoder_input = self_attention(norm1(decoder_input), padding_mask=tgt_padding_mask, \
                                                    tgt_sequence_mask = tgt_sequence_mask) + decoder_input
                decoder_input = cross_attention(norm2(decoder_input), encoder_input, \
                                                    padding_mask=src_padding_mask) + decoder_input
                decoder_input = ff(norm3(decoder_input)) + decoder_input
        return decoder_input

class Prediction(nn.Module):
    """
    预测层
    """
    def __init__(self, h, v):
        super().__init__()
        self.w = nn.Linear(h, v)

    def forward(self, x):
        return self.w(x)

class Transformer(nn.Module):
    """
    Transformer架构
    """
    def __init__(self, v, h, a, num_encoder_layers, num_decoder_layers, dimFF, dropout, max_len, device):
        super().__init__()
        self.embedding = Embedding(v,h, device)
        self.posEncoding = PositionalEncoding(h, 0, max_len)
        self.transformer = TransformerEncoderDecoder(h, a, num_encoder_layers, num_decoder_layers, dimFF, dropout, device = device)
        self.predict = Prediction(h, v)
        self.max_len = max_len
        self.device = device

    def forward(self, src, tgt, src_padding_mask = None, tgt_padding_mask = None, tgt_sequence_mask = None):
        """
        src/tgt: 两段index序列，分别被嵌入层转化为编码器和解码器输入
        src_padding_mask: 编码器pad掩码
        tgt_padding_mask: 解码器pad掩码
        tgt_sequence_mask: 解码器自注意力序列掩码，用于遮蔽未来信息
        如果不手动提供上述掩码，会自动生成默认pad掩码和序列掩码
        """
        if src_padding_mask is None: 
            src_padding_mask = self.get_key_padding_mask(src, src.device).to(src.device)
        if tgt_padding_mask is None: 
            tgt_padding_mask = self.get_key_padding_mask(tgt, tgt.device).to(tgt.device)
        if tgt_sequence_mask is None: 
            tgt_sequence_mask = self.get_sequence_mask(tgt, tgt.device).to(tgt.device)

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.posEncoding(src)
        tgt = self.posEncoding(tgt)

        output = self.transformer(src, tgt, src_padding_mask, tgt_padding_mask, tgt_sequence_mask)

        return output
    
    @staticmethod
    def get_sequence_mask(tgt, device):
        """
        tgt: (s,) ---> (s,s)
        生成序列掩码
        """
        size = tgt.size()[-1]
        sr = torch.triu(torch.full((size, size), True, device = device), diagonal=1)
        # print(sr)
        return sr

    @staticmethod
    def get_key_padding_mask(tokens, device):
        """
        tokens: (s,) ---> (s,)
        生成pad掩码
        """
        key_padding_mask = torch.full(tokens.size(), False, dtype=bool, device = device)
        key_padding_mask[tokens == 1] = True
        return key_padding_mask