
local LReLU, parent = torch.class('nn.LReLU','nn.Module')
 
function LReLU:__init(a)
	parent.__init(self)
	self.a = a or 0.1
	self.ReLU_p = nn.ReLU()
end

function LReLU:updateOutput(input)
	self.output:resizeAs(input)
	self.output:copy(input)
	self.output:mul(self.a)
	self.ReLU_p.output = self.ReLU_p:updateOutput(input)
	self.ReLU_p.output:mul(1-self.a)
	self.output:add(self.ReLU_p.output)
	return self.output
end

function LReLU:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput)
	self.gradInput:copy(gradOutput)
	self.gradInput:mul(self.a)
	self.ReLU_p.gradInput = self.ReLU_p:updateGradInput(input, gradOutput)
	self.ReLU_p.gradInput:mul(1-self.a)
	self.gradInput:add(self.ReLU_p.gradInput)
	return self.gradInput
end
