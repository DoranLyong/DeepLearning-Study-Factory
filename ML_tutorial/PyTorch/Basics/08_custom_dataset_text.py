# coding=<utf-8> 
"""
(ref) https://youtu.be/9sHcLvVXsns
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py
(Image-captions dataset) https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'
"""

"""
이미지 캡셔닝 과제:
    - Let's convert text -> numerical values 

1. Make a Vocabulary mapping each word to a index 

2. Setup a Pytorch dataset to load the data 

3. Setup padding of every batch (all examples should be of same seq_len and setupt dataloader)


"""


#%% 임포트 라이브러리 
import os 
import os.path as osp

import pandas as pd  # .csv 형식의 데이터 프레임을 다루는 라이브러리 (for lookup in annotation file)
from PIL import Image  # Load img
import spacy  # for tokenizer
import torch 
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches
from torch.utils.data import Dataset      # 가져다쓸 데이터셋 객체를 지칭하는 클래스 (ref) https://huffon.github.io/2020/05/26/torch-data/


import torchvision.transforms as transforms  # Transformations we can perform on our dataset




# ================================================================= #
#                   1. Create FlickrDataset                         #
# ================================================================= #
# %% 01. 데이터셋 로더 생성하기 
"""
(dataset) https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
에서 데이터을 먼저 받는다. 

데이터셋은 ./dataset/flickr8k 에 위치시킨다. 


# Dataset 클래스와 DataLoader 클래스의 관계:
- DataLoader와 Dataset 클래스의 상속관계 (ref) https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/
- Dataset 클래스로 커스텀 데이터 셋을 만든다 => 생성된 데이터셋을 DataLoader 클래스로 전달해서 불러온다 (ref) https://wikidocs.net/57165
- (ref) https://doranlyong-ai.tistory.com/42
"""


# Download with: python -m spacy download en
spacy_eng = spacy.load("en")    # 영어 모델을 사용한다 
                                #(ref) https://yujuwon.tistory.com/entry/spaCy-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%82%AC%EC%9A%A9

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]



class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        """
        가져다쓸 데이터셋의 정보를 초기화한다. 
        """        
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())


    def __len__(self):
        """
        초기화된 객체가 컨테이너 자료형을 가지고 있으면, 그것의 길이를 반환한다

        __len__() 매직 함수를 사용하면 내장 함수 len()을 사용할 수 있다 

        (ref) https://dgkim5360.tistory.com/entry/python-duck-typing-and-protocols-why-is-len-built-in-function
        (ref) https://kwonkyo.tistory.com/234
        (ref) https://medium.com/humanscape-tech/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%98-%EC%8A%A4%ED%8E%98%EC%85%9C-%EB%A9%94%EC%84%9C%EB%93%9C-special-method-2aea6bc4f2b9

        """
        return len(self.df) # 로드된 데이터의 개수(길이) 를 반환한다 


    def __getitem__(self, index):
        """
        데이터셋 시퀀스에서 특정 index에 해당하는 아이템을 가져온다 (= 객체에 indexing 기능을 사용할 수 있음). 

        (ref) http://hyeonjae-blog.logdown.com/posts/776615-python-getitem-len
        """
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB") # index 에 해당하는 이미지를 가져온다. 


        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])            

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





        

# ================================================================= #
#                         2. Set device                             #
# ================================================================= #
# %% 02. 프로세스 장비 설정 
gpu_no = 0  # gpu_number 
device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')


# ================================================================= #
#                       3. Hyperparameters                          #
# ================================================================= #
# %% 03. 하이퍼파라미터 설정 
in_channel = 1
num_classes = 10 
learning_rate = 0.001
batch_size = 64
num_epochs = 10

load_model = False  # 체크포인트 모델을 가져오려면 True 


# ================================================================= #
#                      4.  Load Data your dataset                   #
# ================================================================= #
# %% 04. 커스텀 데이터 로드 

"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)


def get_loader( root_folder, annotation_file,
                transform, batch_size=32,
                num_workers=8, shuffle=True,
                pin_memory=True, 
                ):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset




transform = transforms.Compose( [transforms.Resize((224, 224)), 
                                transforms.ToTensor(),]
                                )



loader, dataset = get_loader( "./dataset/flickr8k/images/", "./dataset/flickr8k/captions.txt", transform=transform  )




for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
# %%
