
---------- library ----------

require 'torch'
require 'image'
require 'nn'
require 'cunn'

---------- functions ----------

function newmodel()

	-- input dimensions
	local nfeats = 1

	-- filter size
	local filtsize = 3

	-- hidden units
	local nstates = {96,96,96,96,96}

	-- model:
	local model = nn.Sequential()

	-- stage 1 : Convolution
	model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize, 1, 1, 1))
	model:add(nn.LeakyReLU(0.1))

	-- stage 2 : Convolution
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], 1, 1, 1))
	model:add(nn.LeakyReLU(0.1))

	-- stage 3 : Convolution
	model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize, filtsize, 1, 1, 1))
	model:add(nn.LeakyReLU(0.1))

	-- stage 4 : Convolution
	model:add(nn.SpatialConvolutionMM(nstates[3], nstates[4], 1, 1, 1))
	model:add(nn.LeakyReLU(0.1))

	-- stage 5 : Convolution
	model:add(nn.SpatialConvolutionMM(nstates[4], nstates[5], filtsize, filtsize, 1, 1, 1))
	model:add(nn.LeakyReLU(0.1))

	-- stage 6 : Convolution
	model:add(nn.SpatialConvolutionMM(nstates[5], 1, filtsize, filtsize, 1, 1, 1))

	model:add(nn.LeakyReLU(0.001))
	model:add(nn.AddConstant(-1,true))
	model:add(nn.MulConstant(-1,true))
	model:add(nn.LeakyReLU(0.001))
	model:add(nn.MulConstant(-1,true))
	model:add(nn.AddConstant(1,true))

	return model
end

function newthreshold()
	local model = nn.Sequential()
	model:add(nn.ReLU())
	model:add(nn.AddConstant(-1,true))
	model:add(nn.MulConstant(-1,true))
	model:add(nn.ReLU())
	model:add(nn.MulConstant(-1,true))
	model:add(nn.AddConstant(1,true))

	return model
end

---------- main ----------

torch.setdefaulttensortype('torch.CudaTensor')

model = newmodel()
print(model)

thresholding = newthreshold()
print(thresholding)

-- loss function
criterion = nn.MSECriterion()
print '==> here is the loss function:'
print(criterion)

torch.setdefaulttensortype('torch.FloatTensor')

model:cuda()
thresholding:cuda()
criterion:cuda()
