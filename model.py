import torch
import torch.nn as nn
import timm # model zoo

def replace_layers(model, old, new):
    for n, module in model.named_children(): # namedchildren = 모델이 가지고 있는 모든 parameter(conv layer/activation layer)
        if len(list(module.children())) > 0: #model 안의 old exist?
            replace_layers(module,old,new)
        if isinstance(module, old): # old의 객체?
            setattr(model, n, new)

class MaskDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = timm.create_model('resnet18', pretrained=True)
        self.feature_extractor.reset_classifier(0,'')
    
        replace_layers(self.feature_extractor, nn.ReLU, nn.ELU())

        self.DME_classifier = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256,2)
        )

    def forward(self,x):
        feature = self.feature_extractor(x) # 512*7*7
        feature = feature.mean(-1).mean(-1) # 위의 3차원 값을 512로 만듦

        y = self.DME_classifier(feature)
        return y # predicton

if __name__ == '__main__':
    model = MaskDetectionModel()

    dummy_input = torch.randn(1, 3, 224, 224)
    pred = model(dummy_input)
    print(pred)