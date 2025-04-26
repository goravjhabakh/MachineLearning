import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loader
from torchvision import transforms
from model import ImageCaptioningModel
from tqdm import tqdm

root = '../Datasets/Flickr8k/Images'
annot = '../Datasets/Flickr8k/captions.txt'

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def train():
    loader,vocab = get_loader(root, annot, transform=transform)
    torch.backends.cudnn.benchmark = True 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'No of batches: {len(loader)}')
    print(f'Vocab size: {len(vocab)}')

    # Hyperparametes
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1
    lr = 3e-4
    num_epochs = 1

    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loop = tqdm(loader, desc=f'Epoch: [{epoch+1}/{num_epochs}]')

        for img,captions in loop:
            # Format data into correct shape
            img, captions = img.to(device), captions.to(device)

            # Forward
            outputs = model(img, captions)
            targets = captions 
            print(outputs.shape)
            print(targets.shape)
            exit()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train()