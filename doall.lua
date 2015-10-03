
---------- library ----------

require 'nn'
require 'cunn'

---------- settings ----------

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 91, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- path:
cmd:option('-path_models', 'models', 'subdirectory to save models')
cmd:option('-path_saveimg', 'save', 'subdirectory to save images')
cmd:option('-path_submission', 'submission', 'subdirectory to submission file')
-- training:
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

---------- read dataset and function ----------

dofile "1_data.lua"
dofile "2_model.lua"
dofile "3_train.lua"
dofile "4_test.lua"
dofile "5_valid.lua"

---------- execute ----------

print("==> training")
train()

print("==> validation")
valid()

print("==> test")
test()
