
torch.setdefaulttensortype('torch.FloatTensor')

---------- library ----------

require 'torch'
require 'image'

require "lib/getfilename"
require "lib/window"
require "lib/patch"
require "lib/calc_rsme"

-- read dataset:

---------- functions ----------

function load_imgs(folderpath)

	filepaths = getFilename(folderpath)
	images = {}

	for i, finame in pairs(filepaths) do
		images[i] = - image.load(folderpath .. finame)[1] + 1
	end

	return images
end

---------- main ----------

patch_size = 50
overlap = 40

train_images = load_imgs("dataset/train/")
train_cleaned_images = load_imgs("dataset/train_cleaned/")
test_images = load_imgs("dataset/test/")

print("  num_all_images: " .. #train_images)

valid_images = {}
valid_cleaned_images = {}

for i = 1,10 do
	valid_images[i] = table.remove(train_images)
	valid_cleaned_images[i] = table.remove(train_cleaned_images)
end

print("num_train_images: " .. #train_images)
print("num_valid_images: " .. #valid_images)

train_data = img2patch(train_images)
retrain_images = patch2img(train_data, train_images)

train_cleaned_data = img2patch(train_cleaned_images)
retrain_cleaned_images = patch2img(train_cleaned_data, train_cleaned_images)

test_data = img2patch(test_images)
valid_data = img2patch(valid_images)
