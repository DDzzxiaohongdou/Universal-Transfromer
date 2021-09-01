import torch

ngpu = 4

#检测是否有可用的gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")

project_dir = 'D:/'
cwd = project_dir + 'english_french_transfromer'
data_dir = cwd + '/data/'
save_dir = cwd + '/save/'

#Transformer 的基础模型使用的数值为：num_layers=6，d_model = 512，dff = 2048
#为了让本示例小且相对较快，已经减小了num_layers、 d_model 和 dff 的值。
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
MAX_LENGTH = 10
BATCH_SIZE = 64 * ngpu
EPOCHS = 40 # 50 # 30  # 20
print_trainstep_every = 50  # 每50个step做一次打印