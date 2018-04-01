# reference : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch



config = {
	'vgg16':{
			'arch' : ['64', '64', 'M', '128','128', 'M', '256', '256', '256', 'M',
				'512', '512', '512', 'M', '512' , '512', '512', 'M'],
			'scale_factor' : 32
		},
	'dvgg16' :{
			'arch' : ['64', '64', 'M', '128','128', 'M', '256', '256', '256', 'M',
				'512', '512', '512', '512d', '512d', '512d'],
			'scale_factor' : 8,
			'weight_url' : 'https://www.dropbox.com/s/8ts4mtf4xvkds19/dvgg16-457ecedb3636.pth?dl=1'
		}
}



class Encoder(nn.Module):

	def __init__(self, features):
		super(Encoder, self).__init__()

		self.features = features
		self.classifier = nn.Conv2d(512,1, kernel_size=1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		feat = self.classifier(self.features(x))
		b, c, w, h = feat.size()
		return self.sigmoid(feat.view(b,-1)).view(b,c,w,h)
		###############################
		# for name, module in self.features._modules.items():
		# 	x = module(x)
		# 	output.append(x)
		# return output
		###############################
		# return self.features(x)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_weights(self, w_path):
		self.load_state_dict(torch.load(w_path)['state_dict'])


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			if 'd' in v:
				v = int(''.join(i for i in v if i.isdigit()))
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
			else:
				v = int(''.join(i for i in v if i.isdigit()))
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)



def make_encoder(config, pretrained=False, **kwargs):
	model = Encoder(make_layers(config['arch']), **kwargs)
	if pretrained:
		weights = model_zoo.load_url(config['weight_url'])
		model.load_state_dict(weights['state_dict'])
	return model
