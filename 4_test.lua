
---------- library ----------

require 'torch'
require 'xlua'

---------- functions ----------

function gen_submission(cleaned_images, img_names)

	print("=> Save Submission File")

	local valid_score = valid_score or 0
	local path = "submission/submission_valid" .. string.format("%.4f", valid_score) .. ".csv"
	os.execute('mkdir -p ' .. sys.dirname(path))
	local fp = io.open(path, "w")

	fp:write("id,value\n")

	for i = 1,#cleaned_images do
		xlua.progress(i, #cleaned_images)

		local img_id = img_names[i]
		img_id = img_id:sub(1,img_id:len()-4)
		local cleaned_img = cleaned_images[i]

		for k = 1, cleaned_images[i]:size(2) do
			for j = 1, cleaned_images[i]:size(1) do
				fp:write(img_id .. "_" .. j .. "_" ..  k .. "," .. tostring(cleaned_img[{j,k}]), "\n")
			end
		end
	end

	fp:close()
end

function test()

	-- set model to evaluate mode
	model:evaluate()

	local test_cleaned_data = torch.Tensor(test_data:size(1), test_data:size(2), test_data:size(3))

	-- test over test data
	print('==> testing on test set:')
	local test_nrow = test_data:size(1)
	for t = 1,test_nrow do

		-- disp progress
		xlua.progress(t, test_nrow)

		-- get new sample
		local input = test_data[{{t},}]
		input = input:cuda()

		-- test sample
		local pred = - thresholding:forward(model:forward(input)) + 1
		test_cleaned_data[{{t,}}] = pred:float()
	end

	local testimg_names = getFilename("dataset/test/")

	os.execute('mkdir -p ' .. opt.path_saveimg)
	test_cleaned_images = patch2img(test_cleaned_data, test_images)
	for i = 1,#test_cleaned_images do
		local testimg_id = testimg_names[i]
		testimg_id = testimg_id:sub(1,testimg_id:len()-4)

		image.save("save/test_cleaned_images_" .. testimg_id .. ".png", test_cleaned_images[i])
	end

	gen_submission(test_cleaned_images, testimg_names)
end
