import json


class SummaryWriter:
	def __init__(self, filename):
		self.filename = filename
		self.container = {}
		
	def add_scalar(self, key, scalar, epoch):
		if key not in self.container:
			self.container[key] = []
		else:
			dic = {}
			dic[epoch] = scalar
			self.container[key].append(dic)
		
	def close(self):
		with open(self.filename, 'wb') as json_file:
			json.dump(self.container, json_file)
