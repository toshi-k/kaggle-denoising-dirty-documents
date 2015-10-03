
function calc_rsme(images1, images2)
	local value = 0
	local num_pixel = 0

	for i = 1,#images1 do
		local dif = images1[i] - images2[i]
		value = value + dif:pow(2):sum()
		num_pixel = num_pixel + dif:nElement()
	end

	value = math.sqrt(value / num_pixel)

	return value
end
