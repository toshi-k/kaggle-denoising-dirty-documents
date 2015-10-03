
-- img2patch
function img2patch(images)

	patch_num = 0
	for i = 1,#images do
		num_y = math.ceil((images[i]:size(1) - patch_size) / (patch_size - overlap )) + 1
		num_x = math.ceil((images[i]:size(2) - patch_size) / (patch_size - overlap )) + 1
		patch_num = patch_num + num_x * num_y
	end

	count = 1
	data = torch.Tensor(patch_num, patch_size, patch_size)

	for i = 1,#images do

		size_x = images[i]:size(2)
		size_y = images[i]:size(1)
		num_y = math.ceil((size_y - patch_size) / (patch_size - overlap )) + 1
		num_x = math.ceil((size_x - patch_size) / (patch_size - overlap )) + 1

		for sx = 1,num_x do
			for sy = 1,num_y do
				x = 1 + (patch_size - overlap) * (sx-1)
				y = 1 + (patch_size - overlap) * (sy-1)
				if x+patch_size-1 > size_x then x = size_x - patch_size + 1 end
				if y+patch_size-1 > size_y then y = size_y - patch_size + 1 end

				img = images[i]
				data[{{count}}] = img[{{y,y+patch_size-1},{x,x+patch_size-1}}]
				count = count + 1
			end
		end
	end

	return data
end

-- patch2img
function patch2img(data, original_images)

	local count = 1
	local images = {}
	local window = hanning(patch_size)

	for i = 1,#original_images do
		size_x = original_images[i]:size(2)
		size_y = original_images[i]:size(1)
		images[i] = torch.Tensor(size_y, size_x):zero()
		local weight = torch.Tensor(size_y, size_x):zero()
		num_y = math.ceil((size_y - patch_size) / (patch_size - overlap )) + 1
		num_x = math.ceil((size_x - patch_size) / (patch_size - overlap )) + 1

		for sx = 1,num_x do
			for sy = 1,num_y do
				x = 1 + (patch_size - overlap) * (sx-1)
				y = 1 + (patch_size - overlap) * (sy-1)
				if x+patch_size-1 > size_x then x = size_x - patch_size + 1 end
				if y+patch_size-1 > size_y then y = size_y - patch_size + 1 end
				img_copy = images[i]
				img_copy[{{y,y+patch_size-1},{x,x+patch_size-1}}] = img_copy[{{y,y+patch_size-1},{x,x+patch_size-1}}] + torch.cmul(data[{count}], window)
				weight[{{y,y+patch_size-1},{x,x+patch_size-1}}] = weight[{{y,y+patch_size-1},{x,x+patch_size-1}}] + window
				images[i] = img_copy
				count = count + 1
			end
		end

		images[i] = torch.cdiv(images[i], weight)
	end

	return images
end
