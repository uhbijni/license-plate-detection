import torchvision
import torch
import os
from detection.detector import LicensePlateDetector
from utils.image_utils import load_image, save_image
from utils.visualization_utils import draw_boxes_text
from utils.ocr_utils import extract_text_from_box
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from detection.dataset import LicensePlateDataset


def main():
    # Initialize the license plate detector
    # detector = LicensePlateDetector()
    # detector.load_model()

    # Define any transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Create the dataset and dataloader
    dataset = LicensePlateDataset(
        image_dir="C:\\Code\\license-plate-detection\\archive\\images",
        annotation_dir="C:\\Code\\license-plate-detection\\archive\\annotations",
        transform=transform,    
    )
    # Define the train-test split ratio
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the trained model
    if os.path.exists("fasterrcnn_resnet50_fpn.pth"):
        model.load_state_dict(
            torch.load("fasterrcnn_resnet50_fpn.pth", weights_only=True)
        )

    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        num_epochs = 30

        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, targets in train_dataloader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()

            lr_scheduler.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        torch.save(model.state_dict(), "fasterrcnn_resnet50_fpn.pth")

    # Set model to evaluation mode, ensuring that layers like dropout and batchnorm are operating in evaluation mode.
    model.eval()
    j = 0
    with torch.no_grad():
        bad_ocr = 0
        for images, targets in test_dataloader:

            images = list(image.to(device) for image in images)
            predictions = model(images)


            for i in range(len(predictions)):
                if len(predictions[i]["boxes"]) == 0:
                    continue

                """
                print(f"Image {i+1} predictions: {predictions[i]['boxes'][0]}")
                print(f"Image {i+1} ground truth: {targets[i]['boxes']}")
                print("\n")
                """

                image = transforms.ToPILImage()(images[i].cpu())
                plate_text = extract_text_from_box(
                    image, targets[i]["boxes"][0].cpu().numpy()
                )
                if plate_text == "no text detected!" or plate_text == "bounding box too small!":
                    bad_ocr += 1    
                
                image_with_boxes_text = draw_boxes_text(
                    image,
                    predictions[i]["boxes"][0].cpu().numpy(),
                    targets[i]["boxes"][0].cpu().numpy(),
                    plate_text,
                )

                image_with_boxes_text.save(
                    f"C:\\Code\\license-plate-detection\\evaluation\\epoch_{j+1}_output_image_{i+1}.png"
                )

            j += 1
        print(f"Bad OCR: {bad_ocr}")


if __name__ == "__main__":
    main()
