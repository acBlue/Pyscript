# -*- coding: utf-8 -*-
#  CNN图片识别训练

import os
import random
import string

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ======================================================================================
# 1. 配置参数
# ======================================================================================
class Config:
    # 数据集路径
    TRAIN_DATA_DIR = './dataset/train/'
    TEST_DATA_DIR = './dataset/test/'
    # 训练参数
    EPOCHS = 50
    BATCH_SIZE = 16  # 减小批次大小，避免内存问题
    LEARNING_RATE = 0.001  # 稍微增大学习率
    TARGET_FRAMES = 10  # 减少目标帧数，提高训练效率
    # 验证码参数
    CAPTCHA_LENGTH = 4
    CHAR_SET = string.digits + string.ascii_uppercase
    CHAR_SET_LENGTH = len(CHAR_SET)
    # 图像参数
    IMAGE_HEIGHT = 45
    IMAGE_WIDTH = 120
    # 模型保存路径
    MODEL_PATH = './captcha_model.pth'


def text_to_vec(text):
    """将文本转换为one-hot编码向量"""
    vector = torch.zeros(Config.CAPTCHA_LENGTH, Config.CHAR_SET_LENGTH)
    for i, char in enumerate(text):
        if i >= Config.CAPTCHA_LENGTH:  # 防止越界
            break
        idx = Config.CHAR_SET.find(char)
        if idx != -1:
            vector[i][idx] = 1.0
    return vector


def vec_to_text(vector):
    """将预测向量转换为文本"""
    char_indices = vector.argmax(dim=-1)  # 使用dim=-1更安全
    return ''.join([Config.CHAR_SET[i] for i in char_indices])


def sample_frames(frames, target_count):
    """将输入的帧列表采样或填充到指定的数量"""
    if not frames:
        return []
    current_count = len(frames)
    if current_count == target_count:
        return frames
    # 如果当前帧数多于目标帧数，则均匀采样
    if current_count > target_count:
        indices = np.linspace(0, current_count - 1, target_count, dtype=int)
        return [frames[i] for i in indices]
    # 如果当前帧数少于目标帧数，则循环填充
    else:
        padded_frames = frames[:]
        while len(padded_frames) < target_count:
            padded_frames.extend(frames)
        return padded_frames[:target_count]


# ======================================================================================
# 2. 生成模拟验证码数据
# ======================================================================================
def generate_captcha_gif(text, save_path, num_frames=15):
    """生成GIF验证码"""
    frames = []
    colors = ['black', 'darkblue', 'darkred', 'darkgreen']

    for frame_idx in range(num_frames):
        # 创建图像
        img = Image.new('L', (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), 255)
        draw = ImageDraw.Draw(img)

        # 尝试加载字体，失败则使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        # 绘制文字，添加一些随机偏移
        char_width = Config.IMAGE_WIDTH // Config.CAPTCHA_LENGTH
        for i, char in enumerate(text):
            x = i * char_width + random.randint(-5, 5) + 10
            y = random.randint(5, 15)
            # 使用灰度值而不是颜色名称
            color = random.randint(0, 100)  # 深色文字
            draw.text((x, y), char, fill=color, font=font)

        # 添加噪声线条
        for _ in range(random.randint(1, 3)):
            start = (random.randint(0, Config.IMAGE_WIDTH), random.randint(0, Config.IMAGE_HEIGHT))
            end = (random.randint(0, Config.IMAGE_WIDTH), random.randint(0, Config.IMAGE_HEIGHT))
            draw.line([start, end], fill=random.randint(100, 200), width=1)

        # 添加噪声点
        for _ in range(random.randint(20, 50)):
            x = random.randint(0, Config.IMAGE_WIDTH - 1)
            y = random.randint(0, Config.IMAGE_HEIGHT - 1)
            draw.point((x, y), fill=random.randint(150, 255))

        frames.append(img)

    # 保存为GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )


def generate_sample_data():
    """生成示例数据"""
    print("生成训练数据...")
    for i in tqdm(range(1000)):
        text = ''.join(random.choices(Config.CHAR_SET, k=Config.CAPTCHA_LENGTH))
        filename = f"{text}.gif"
        filepath = os.path.join(Config.TRAIN_DATA_DIR, filename)
        generate_captcha_gif(text, filepath)

    print("生成测试数据...")
    for i in tqdm(range(50)):
        text = ''.join(random.choices(Config.CHAR_SET, k=Config.CAPTCHA_LENGTH))
        filename = f"{text}.gif"
        filepath = os.path.join(Config.TEST_DATA_DIR, filename)
        generate_captcha_gif(text, filepath)


# ======================================================================================
# 3. 改进的Dataset
# ======================================================================================
class GifSequenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []

        # 过滤有效的图片文件
        for f in os.listdir(data_dir):
            if f.lower().endswith(('.gif', '.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(data_dir, f))

        print(f"找到 {len(self.image_paths)} 个有效图片文件")

        # 预处理transforms，添加归一化
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_text = os.path.basename(img_path).split('.')[0].upper()

        # 确保标签长度正确
        if len(label_text) != Config.CAPTCHA_LENGTH:
            print(f"警告: 标签长度不匹配 {label_text}, 跳过")
            return None, None

        label = text_to_vec(label_text)

        raw_frames = []
        try:
            with Image.open(img_path) as img:
                # 处理单帧图像
                if not hasattr(img, 'n_frames'):
                    frame = img.convert('L')
                    if frame.size != (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT):
                        frame = frame.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
                    raw_frames = [frame] * Config.TARGET_FRAMES
                else:
                    # 处理GIF多帧
                    for i in range(min(img.n_frames, 50)):  # 限制最大帧数
                        img.seek(i)
                        frame = img.convert('L')
                        if frame.size != (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT):
                            frame = frame.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
                        raw_frames.append(frame)

            if not raw_frames:
                print(f"警告: 无帧数据 {img_path}")
                return None, None

            # 采样到目标帧数
            processed_frames = sample_frames(raw_frames, Config.TARGET_FRAMES)

            # 转换为tensor
            tensor_frames = []
            for frame in processed_frames:
                tensor_frame = self.transform(frame)
                tensor_frames.append(tensor_frame)

            sequence = torch.stack(tensor_frames)
            return sequence, label

        except Exception as e:
            print(f"警告: 无法加载 {img_path}: {e}")
            return None, None


def collate_fn(batch):
    """自定义的batch整理函数"""
    # 过滤掉None数据
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    sequences, labels = zip(*batch)

    # 确保所有序列长度一致
    sequences = torch.stack(sequences)  # [batch, frames, channels, height, width]
    labels = torch.stack(labels)  # [batch, captcha_length, char_set_length]

    return sequences, labels


# ======================================================================================
# 4. 改进的CNN + RNN 模型
# ======================================================================================
class CaptchaCNN_RNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN_RNN, self).__init__()

        # CNN特征提取器
        self.cnn = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        # 计算CNN输出特征大小
        cnn_output_h = Config.IMAGE_HEIGHT // 8  # 经过3次池化
        cnn_output_w = Config.IMAGE_WIDTH // 8
        cnn_output_features = 128 * cnn_output_h * cnn_output_w

        # RNN层
        self.rnn = nn.LSTM(
            input_size=cnn_output_features,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        # 分类器 - 每个位置单独预测
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, Config.CHAR_SET_LENGTH)
            ) for _ in range(Config.CAPTCHA_LENGTH)
        ])

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape

        # 重塑输入以便CNN处理
        x = x.view(batch_size * num_frames, C, H, W)

        # CNN特征提取
        cnn_out = self.cnn(x)  # [batch*frames, 128, H', W']

        # 重塑为RNN输入
        cnn_out = cnn_out.view(batch_size, num_frames, -1)

        # RNN处理序列
        rnn_out, _ = self.rnn(cnn_out)  # [batch, frames, 256]

        # 使用最后一帧的输出
        final_features = rnn_out[:, -1, :]  # [batch, 256]

        # 每个字符位置分别预测
        outputs = []
        for i in range(Config.CAPTCHA_LENGTH):
            char_output = self.classifiers[i](final_features)  # [batch, char_set_length]
            outputs.append(char_output)

        # 堆叠输出
        output = torch.stack(outputs, dim=1)  # [batch, captcha_length, char_set_length]
        return output


# ======================================================================================
# 5. 主训练函数
# ======================================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 检查并生成数据
    if not os.path.exists(Config.TRAIN_DATA_DIR) or not os.listdir(Config.TRAIN_DATA_DIR):
        os.makedirs(Config.TRAIN_DATA_DIR, exist_ok=True)
        os.makedirs(Config.TEST_DATA_DIR, exist_ok=True)
        generate_sample_data()

    print("正在加载数据集...")
    train_dataset = GifSequenceDataset(data_dir=Config.TRAIN_DATA_DIR)

    if len(train_dataset) == 0:
        print("错误: 训练数据集为空!")
        return

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows上建议设为0
        drop_last=True
    )

    print("正在初始化模型...")
    model = CaptchaCNN_RNN().to(device)

    # 使用交叉熵损失函数 - 这是关键修复!
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    print("开始训练...")
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            if images.nelement() == 0:
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [batch, captcha_length, char_set_length]

            # 计算损失 - 每个位置分别计算
            loss = 0
            for i in range(Config.CAPTCHA_LENGTH):
                char_loss = criterion(outputs[:, i, :], labels[:, i, :].argmax(dim=1))
                loss += char_loss
            loss = loss / Config.CAPTCHA_LENGTH

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算准确率
            with torch.no_grad():
                pred_chars = outputs.argmax(dim=-1)  # [batch, captcha_length]
                true_chars = labels.argmax(dim=-1)  # [batch, captcha_length]

                # 整个验证码完全正确才算对
                correct_samples = (pred_chars == true_chars).all(dim=1).sum().item()
                correct_predictions += correct_samples
                total_predictions += images.size(0)

            total_loss += loss.item()
            current_acc = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_predictions

        print(f"\nEpoch {epoch + 1} 完成 | 平均损失: {avg_loss:.4f} | 准确率: {accuracy:.2f}%")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{Config.MODEL_PATH}_epoch_{epoch + 1}")

    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"模型已保存至 {Config.MODEL_PATH}")


# ======================================================================================
# 6. 预测函数
# ======================================================================================
def predict(gif_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaCNN_RNN().to(device)

    try:
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"模型加载成功: {Config.MODEL_PATH}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {Config.MODEL_PATH}。请先运行训练。")
        return

    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    raw_frames = []
    try:
        with Image.open(gif_path) as img:
            if not hasattr(img, 'n_frames'):
                frame = img.convert('L')
                if frame.size != (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT):
                    frame = frame.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
                raw_frames = [frame] * Config.TARGET_FRAMES
            else:
                for i in range(img.n_frames):
                    img.seek(i)
                    frame = img.convert('L')
                    if frame.size != (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT):
                        frame = frame.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
                    raw_frames.append(frame)
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    processed_frames = sample_frames(raw_frames, Config.TARGET_FRAMES)
    tensor_frames = [transform(frame) for frame in processed_frames]
    sequence = torch.stack(tensor_frames).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sequence)
        pred_chars = output.argmax(dim=-1).squeeze(0)
        predict_text = ''.join([Config.CHAR_SET[i] for i in pred_chars])

    true_text = os.path.basename(gif_path).split('.')[0].upper()
    print(f"图片: {os.path.basename(gif_path)}")
    print(f"真实标签: {true_text}")
    print(f"预测结果: {predict_text}")
    print(f"是否正确: {'✓' if predict_text == true_text else '✗'}")
    print("-" * 50)


if __name__ == '__main__':
    # main()

    # 测试预测
    print("\n--- 开始预测测试集中的图片 ---")
    if os.path.exists(Config.TEST_DATA_DIR):
        test_files = os.listdir(Config.TEST_DATA_DIR)  # 只测试前10个
        if test_files:
            for test_file in test_files:
                predict(os.path.join(Config.TEST_DATA_DIR, test_file))
        else:
            print("测试集为空，无法进行预测。")
    else:
        print("测试目录不存在。")
