import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.embed_size = embed_size

        pretrained_model = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self,x,captions):
        x = self.backbone(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        embeddings = self.embed(captions)
        print('Embedding dims:',embeddings.shape)
        x = x.unsqueeze(0)
        print('X unsqueezed:',x.shape)
        embeddings = torch.cat((x, embeddings), dim=0)

        hidden,_ = self.lstm(embeddings)
        output = self.linear(hidden)

        return output
    
    def generate_caption(self, img, vocab, max_length = 50):
        caption = []
        with torch.no_grad():
            features = self.backbone(img)
            features = torch.flatten(features,1)
            features = self.fc(features)
            inputs = features.unsqueeze(0)
            states = None

            for _ in range(max_length):
                hidden, states = self.lstm(inputs, states)
                output = self.linear(hidden.squeeze(0))
                pred = output.argmax(1)
                caption.append(pred.item())

                inputs = self.embed(pred).unsqueeze(0)

                if vocab.itos[pred.item()] == '<EOS>': break

        return [vocab.itos[i] for i in caption]

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ImageCaptioningModel(64,64,64,64).to(device)
# x = torch.randn(1,3,224,224).to(device)
# captions = torch.randint(low=4, high=10, size=(20,)).to(device)
# y = model(x,captions)
# print(y.shape)
# print(model)