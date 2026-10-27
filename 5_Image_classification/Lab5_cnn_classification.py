from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time


# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Ошибка довольно большая потому что сеть простая и мало эпох обучения
# Существенно улучшить результаты можно если взять предобученную сеть


# Так как сеть, которую мы планируем взять за базу натренирована на изображениях 
# определенного размера, то наши изображения необходимо к ним преобразовать
data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225] )
    ])


# Пересоздадим датасеты с учетом новых размеров и нормировки яркости
train_dataset = torchvision.datasets.ImageFolder(root=str(Path(__file__).resolve().parent / 'data/train'),
                                                 transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=str(Path(__file__).resolve().parent / 'data/test'),
                                             transform=data_transforms)

class_names = train_dataset.classes  
batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True,  num_workers=0)


test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=0) 


# В качестве донора возьмем преобученную на ImageNet наборе сеть AlexNet
# Список доступных предобученных сетей можно посмотреть тут https://pytorch.org/vision/main/models.html
torch.manual_seed(42)
torch.set_num_threads(2)
net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)

# можно посмотреть структуру этой сети
print(net)

# Так как веса feature_extractor уже обучены, нам нужно их "заморозить", чтобы 
# быстрее научился наш классификатор
#  для этого отключаем у всех слоев (включая слои feature_extractor-а) градиенты
for param in net.parameters():
    param.requires_grad = False


# Выходной слой AlexNet содержит 1000 нейронов (по количеству классов в ImageNet).
# Нам нужно его заменить на слой, содержащий только 3 класса.

num_classes = 3

new_classifier = net.classifier[:-1] # берем все слой классификатора кроме последнего
new_classifier.add_module('fc',nn.Linear(4096,num_classes))# добавляем последним слой с тремя нейронами на выходе
net.classifier = new_classifier # меняем классификатор сети

net = net.to(device)

# проверим эффективность новой сети
net.eval()
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
# явно требуется обучение


# Перейдем к обучению.
# Зададим количество эпох обучения, функционал потерь и оптимизатор.
num_epochs = 10
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# создаем цикл обучения и замеряем время его выполнения
t = time.time()
save_loss = []
for epoch in range(num_epochs):
    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # прямой проход
        outputs = net(images)
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
         # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        # делаем шаг оптимизации весов
        optimizer.step()
        save_loss.append(loss.item())
        # выводим немного диагностической информации
        if i%100==0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)

# Посмотрим как уменьшался loss в процессе обучения
plt.figure()
plt.plot(save_loss)
plt.xlabel("Training step")
plt.ylabel("Cross entropy")
result_dir = Path(__file__).resolve().parent / "results"
result_dir.mkdir(exist_ok=True)
plt.savefig(result_dir / "loss.png", dpi=160, bbox_inches="tight")

# Еще раз посчитаем точность нашей модели
net.eval()
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
# уже лучше


# Реализуем отображение картинок и их класса, предсказанного сетью
inputs, classes = next(iter(test_loader))
net.eval()
with torch.no_grad():
    pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i,j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[int(j.item())])
    plt.pause(2)
# Сохраняем обученный выходной слой и пример предсказаний.
torch.save({'classes': class_names, 'head_state_dict': net.classifier[-1].state_dict()}, result_dir / 'classifier_head.pt')
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for k, ax in enumerate(axes.flat):
    image = inputs[k].numpy().transpose((1, 2, 0))
    image = np.clip(image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    ax.imshow(image)
    ax.set_title(f'True: {class_names[int(classes[k])]}\nPred: {class_names[int(pred_class[k])]}')
    ax.axis('off')
fig.tight_layout()
fig.savefig(result_dir / 'predictions.png', dpi=160)
