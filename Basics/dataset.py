import os
import json
import pandas as pd
from PIL import Image
import spacy # tokenizer
import spacy.tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms

# ImageFolder
# batch_size = 8

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])

# dataset = datasets.ImageFolder(root='../Datasets/soil_types', transform=transform)

# train_set, test_set = torch.utils.data.random_split(dataset, [124,32])
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Image Captioning Dataset
# (Flickr8k) Image Input --> Caption Output
# 1. Convert text --> numerical values
# 2. We need a vocabulary to map each word to an index
# 3. Setup PyTorch dataset to load the data
# 4. Setup padding to make all examples to same sequence length
# 5. Setup DataLoader
#
# python -m spacy download en_core_web_sm

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_th):
        self.itos = {0:'<PAD>', 1:'<SOS>', 2:'<EOS>', 3:'<UNK>'}
        self.stoi = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.freq_th = freq_th
    
    def __len__(self):
        return len(self.itos)
    
    def tokenizer_eng(eself,text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentences):
        freq = {}
        idx = 4
        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in freq: freq[word] = 1
                else : freq[word] += 1
                if freq[word] == self.freq_th:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]

class Flickr8kDataset(Dataset):
    def __init__(self, root, captions_file, transform = None, freq_th = 5):
        self.root = root
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, captions columns
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # Initialize and build vocab
        self.vocab = Vocabulary(freq_th)
        self.vocab.build_vocabulary(self.captions.to_list())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = self.imgs[idx]
        img = Image.open(os.path.join(self.root,img_id)).convert('RGB')
        if self.transform: img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(numericalized_caption)
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    
def get_loader(root, annot_file, transform = None, batch_size = 32, num_workers = 4, shuffle = True, pin_memory = True):
    dataset = Flickr8kDataset(root, annot_file, transform=transform)
    with open('vocab.txt', 'w') as f:
        json.dump(dataset.vocab.stoi, f, indent=4)
    print(f'Total Images: {len(dataset)}')

    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle,
        num_workers = num_workers,pin_memory = pin_memory, collate_fn=MyCollate(pad_idx))
    return loader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
root = '../Datasets/Flickr8k/Images'
annot = '../Datasets/Flickr8k/captions.txt'

def main():
    data_loader = get_loader(root, annot, transform=transform)
    print(f'Total batches: {len(data_loader)}')
    for idx,(x,y) in enumerate(data_loader):
        print(f'Batch{idx+1}: {x.shape}')
        print(f'Batch{idx+1}: {y.shape}')
        if idx == 5: break

if __name__ == '__main__':
    main()