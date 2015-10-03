
function getFilename( folder )

	local fp
	local msg
	local str
	local t = {}
	local i = 1

	local tmpFile = "files.tmp"
	local filenames = {}

	os.execute( "ls -l '"..folder.."' > "..tmpFile )

	-- Open file
	fp,msg = io.open( tmpFile, "r")
	if not(fp) then
		dialog( tmpFile .. "Can not be open", "Stop Program", 0 )
		return
	end

	-- Read data
	str = fp:read("*l") 
	while true do
		str = fp:read("*l")
		if str == nil then break end
		t = string.split( str, " " )
		filenames[i] = t[#t]
		i = i + 1
	end

	-- Close file
	io.close(fp)

	os.execute( "rm -f "..tmpFile )

	return filenames  
end
