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
    loader,vocab = get_loader(root, annot, transform=transform, batch_size=64)
    torch.backends.cudnn.benchmark = True 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparametes
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1
    lr = 3e-4
    num_epochs = 10

    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loop = tqdm(loader, desc=f'Epoch: [{epoch+1}/{num_epochs}]')
        epoch_loss = 0
        for img,captions in loop:
            # Format data into correct shape
            img, captions = img.to(device), captions.to(device)

            # Forward
            outputs = model(img, captions[:-1])
            outputs = outputs.reshape(-1,outputs.shape[2])
            captions = captions.reshape(-1)
            loss = criterion(outputs,captions)            

            # Backward
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f'Loss: {avg_loss:.4f}')

    print('Saving model...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab.stoi
    }, 'model.pth')
    print('Saved model...')

if __name__ == '__main__':
    train()