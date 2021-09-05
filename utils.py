import torch
import numpy as np
from matplotlib import pyplot as plt
from config import device

#优化器
class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warm_steps=4):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warm_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        """
        # rsqrt 函数用于计算 x 元素的平方根的倒数.  即= 1 / sqrt{x}
        arg1 = torch.rsqrt(torch.tensor(self._step_count, dtype=torch.float32))
        arg2 = torch.tensor(self._step_count * (self.warmup_steps ** -1.5), dtype=torch.float32)
        dynamic_lr = torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2)
        """
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        dynamic_lr = (self.d_model ** (-0.5)) * min(arg1, arg2)
        return [dynamic_lr for group in self.optimizer.param_groups]

pad = 1
loss_object = torch.nn.CrossEntropyLoss(reduction='none')
#几种损失函数
def mask_loss_func(real, pred):
    _loss = loss_object(pred.transpose(-1, -2), real)  #[B, E]
    mask = torch.logical_not(real.eq(pad)).type(_loss.dtype)
    _loss *= mask
    return _loss.sum() / mask.sum().item()

def mask_loss_func2(real, pred):
    _loss = loss_object(pred.transpose(-1, -2), real)
    mask = torch.logical_not(real.eq(pad))
    _loss = _loss.masked_select(mask)
    return _loss.mean()

#几种准确率指标函数
def mask_accuracy_func(real, pred):
    _pred = pred.argmax(dim=-1)  # [B, E, V]=>[B, E]
    corrects = _pred.eq(real)  # [B, E]
    mask = torch.logical_not(real.eq(pad))
    corrects *= mask
    return corrects.sum().float() / mask.sum().item()

# 另一种实现方式
def mask_accuracy_func2(real, pred):
    _pred = pred.argmax(dim=-1)
    corrects = _pred.eq(real).type(torch.float32)
    mask = torch.logical_not(real.eq(pad))
    corrects = corrects.masked_select(mask)
    return corrects.mean()

def mask_accuracy_func3(real, pred):
    _pred = pred.argmax(dim=-1)
    corrects = _pred.eq(real)
    mask = torch.logical_not(real.eq(pad))
    corrects = torch.logical_and(corrects, mask)
    return corrects.sum().float()/mask.sum().item()

#填充Mask机制
def create_padding_mask(seq):
    #seq [B, E]
    seq = torch.eq(seq, torch.tensor(pad)).float()
    return seq[:, np.newaxis, np.newaxis, :]  # =>[B, 1, 1, E]

#防止数据泄露的Mask机制
def create_look_ahead_mask(size):  # seq_len
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    #[E, E]
    return mask

#Mask总体函数
def create_mask(inp, targ):
    # encoder padding mask
    enc_padding_mask = create_padding_mask(inp)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    # decoder's first attention block(self-attention)
    # 使用的padding create_mask & look-ahead create_mask
    look_ahead_mask = create_look_ahead_mask(targ.shape[-1])  # =>[targ_seq_len,targ_seq_len] ##################
    dec_targ_padding_mask = create_padding_mask(targ)  # =>[b,1,1,targ_seq_len]
    combined_mask = torch.max(look_ahead_mask, dec_targ_padding_mask)  # 结合了2种mask =>[b,1,targ_seq_len,targ_seq_len]

    # decoder's second attention block(encoder-decoder attention) 使用的padding create_mask
    # 【注意】：这里的mask是用于遮挡encoder output的填充pad，而encoder的输出与其输入shape都是[b,inp_seq_len,d_model]
    # 所以这里mask的长度是inp_seq_len而不是targ_mask_len
    dec_padding_mask = create_padding_mask(inp)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    return enc_padding_mask, combined_mask, dec_padding_mask

# 绘制训练曲线
def plot_metric(df_history, metric, cwd):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(cwd + '/imgs/' + metric + '.png')  # 保存图片
    plt.show()