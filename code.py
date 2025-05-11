import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms, models
from PIL import Image

# 数据集类
class RumorDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        image_path = self.data.iloc[index]['image_path']
        label = self.data.iloc[index]['label']

        # 文本处理
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 图像处理
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)

        return input_ids, attention_mask, image, torch.tensor(label, dtype=torch.long)

# 模型设计
class MultiModalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', num_classes=2):
        super(MultiModalModel, self).__init__()
        # 文本模型
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 128)

        # 图像模型
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 128)

        # 融合层
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, input_ids, attention_mask, images):
        # 文本分支
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_fc(text_output.pooler_output)

        # 图像分支
        image_features = self.image_model(images)

        # 融合
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)

        return output

# 数据预处理
def preprocess_data(data_path, dataset_type='twitter'):
    """
    数据预处理函数
    :param data_path: 数据集路径
    :param dataset_type: 数据集类型 ('twitter' 或 'weibo')
    :return: 数据框和图像预处理方法
    """
    if dataset_type == 'twitter':
        # 处理 Twitter 数据集
        posts_file = f"{data_path}/posts.txt"
        images_folder = f"{data_path}/images"

        # 加载文本数据
        with open(posts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 3:  # 假设格式为: text \t image_name \t label
                text, image_name, label = parts
                image_path = f"{images_folder}/{image_name}"
                data.append({'text': text, 'image_path': image_path, 'label': int(label)})

        dataframe = pd.DataFrame(data)

    elif dataset_type == 'weibo':
        # 处理 Weibo 数据集
        tweets_folder = f"{data_path}/tweets"
        rumor_images_folder = f"{data_path}/rumor_images"
        nonrumor_images_folder = f"{data_path}/nonrumor_images"

        # 加载文本数据
        tweets_files = [f"{tweets_folder}/{file}" for file in os.listdir(tweets_folder) if file.endswith('.txt')]

        data = []
        for tweet_file in tweets_files:
            with open(tweet_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # 判断对应的图像类别
            if 'rumor' in tweet_file:
                image_folder = rumor_images_folder
                label = 1
            else:
                image_folder = nonrumor_images_folder
                label = 0

            # 假设每个文本文件对应一个图像文件
            image_name = os.path.splitext(os.path.basename(tweet_file))[0] + '.jpg'
            image_path = f"{image_folder}/{image_name}"
            data.append({'text': text, 'image_path': image_path, 'label': label})

        dataframe = pd.DataFrame(data)

    else:
        raise ValueError("Unsupported dataset type. Use 'twitter' or 'weibo'.")

    # 图像预处理
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return dataframe, image_transform

# 训练函数
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for input_ids, attention_mask, images, labels in dataloader:
        input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# 测试函数
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, images, labels in dataloader:
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def main():
    # 参数设置
    data_path = 'data/twitter_dataset'  # 或 'data/weibo_dataset'
    dataset_type = 'twitter'  # 或 'weibo'
    batch_size = 16
    max_len = 128
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    data, image_transform = preprocess_data(data_path, dataset_type)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 数据集和数据加载器
    dataset = RumorDataset(data, tokenizer, image_transform, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型、损失函数和优化器
    model = MultiModalModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和评估
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # 测试模型
    test_loss, test_acc = evaluate_model(model, dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()