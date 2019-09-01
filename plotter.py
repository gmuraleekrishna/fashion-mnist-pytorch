import matplotlib.pyplot as plt
import json


with open('log.json', 'rb') as log_file:
	params = json.load(log_file)
	train_acc = params['TAcc']
	train_loss = params['TLoss']
	val_acc = params['VAcc']
	val_loss = params['VLoss']
	ta_y = []
	tl_y = []
	va_y = []
	vl_y = []
	x = []
	for (epoch, tacc), (epoch, tloss), (epoch, vacc), (epoch, vloss) in zip(train_acc.items(), train_loss.items(), val_acc.items(), val_loss.items()):
		ta_y.append(tacc)
		tl_y.append(tloss)
		va_y.append(vacc)
		vl_y.append(vloss)
		x.append(int(epoch))
	
	plt.figure(1)
	plt.plot(x, ta_y, label='Training Accuracy')
	plt.plot(x, va_y, label='Validation Accuracy')
	plt.title("Accuracy Plot")
	plt.legend()
	plt.show()
	
	plt.figure(2)
	plt.plot(x, tl_y, label='Training Loss')
	plt.plot(x, vl_y, label='Validation Loss')
	plt.legend()
	plt.title("Loss Plot")
	# plt.plot(x, x ** 2, label='quadratic')
	# plt.plot(x, x ** 3, label='cubic')
	#
	# plt.xlabel('x label')
	# plt.ylabel('y label')
	
	
	
	plt.show()