import torchvision
import torch
from detection.detector import LicensePlateDetector
from utils.image_utils import load_image, save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from detection.dataset import LicensePlateDataset



def main():
    # Initialize the license plate detector
    #detector = LicensePlateDetector()
    #detector.load_model()

    # Define any transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the dataset and dataloader
    dataset = LicensePlateDataset(image_dir='C:\\Code\\license-plate-detection\\archive\\images', annotation_dir='C:\\Code\\license-plate-detection\\archive\\annotations', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 30

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()   
            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}") 

    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')



   
if __name__ == "__main__":
    main()

