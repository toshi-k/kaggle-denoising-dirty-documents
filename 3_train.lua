
---------- library ----------

require 'torch'
require 'xlua'
require 'optim'
require 'cunn'

-- parameter:

-- Retrieve parameters and gradients:
if model then
	parameters,gradParameters = model:getParameters()
end

print '==> configuring optimizer'
-- optim
optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
}
optimMethod = optim.adam

---------- functions ----------

function train()

	-- number of data
	local train_nrow = train_data:size(1)

	-- epoch tracker
	epoch = epoch or 1

	-- set model to training mode (for modules that differ in training and testing, like Dropout)
	model:training()

	-- shuffle at each epoch
	shuffle = torch.randperm(train_nrow)
	local mse_loss = 0

	local view = nn.View(-1)
	view:cuda()

	-- do one epoch
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,train_nrow,opt.batchSize do

		-- disp progress
		xlua.progress(t, train_nrow)

		-- create mini batch
		local inputs = {}
		local targets = {}
		for i = t,math.min(t+opt.batchSize-1,train_nrow) do
			-- load new sample
			local input = train_data[{{shuffle[i]},}]
			local target = train_cleaned_data[{{shuffle[i]},}]

			target = nn.View(patch_size^2):forward(target):cuda()
			input = input:cuda()

			table.insert(inputs, input)
			table.insert(targets, target)
		end

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- f is the average of all criterions
			local f = 0

			-- evaluate function for complete mini batch
			for i = 1,#inputs do
				-- estimate f
				local output = model:forward(inputs[i])
				local output_view = view:forward(output)
				local err = criterion:forward(output_view, targets[i])
				mse_loss = mse_loss + err

				-- estimate df/dW
				local df_do = criterion:backward(output_view, targets[i])
				model:backward(inputs[i], df_do)
			end

			-- normalize gradients and f(X)
			gradParameters:div(#inputs)
			f = f/#inputs
			mse_loss = mse_loss/#inputs

			-- return f and df/dX
			return f,gradParameters
		end

		-- optimize on current mini-batch
		optimMethod(feval, parameters, optimState)
	end

	-- calc
	train_score = math.sqrt(mse_loss)
	print("train_score: " .. string.format("%.5f", train_score))

	-- save/log current net
	local filename = paths.concat(opt.path_models, 'model.net')
	path.mkdir(sys.dirname(filename))
	print('=> saving model to '..filename)
	torch.save(filename, model)

	-- next epoch
	epoch = epoch + 1
end
