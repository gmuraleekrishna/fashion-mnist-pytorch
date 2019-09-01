import json


class SummaryWriter:
	def __init__(self, filename):
		self.filename = filename
		self.container = {}
		
	def add_scalar(self, key, scalar, epoch):
		dic = dict()
		if key not in self.container:
			dic[epoch] = scalar
			self.container[key] = dic
		else:
			self.container[key][epoch] = scalar
		
	def close(self):
		with open(self.filename, 'w+') as json_file:
			json.dump(self.container, json_file)
