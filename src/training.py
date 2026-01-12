from model import build_model
from data import load_data

def main():
    img_size = (224, 224)
    num_classes = 3  # adjust based on your categories
    batch_size = 32

    train_ds, val_ds = load_data(
        data_dir="data",
        img_size=img_size,
        batch_size=batch_size
    )

    model = build_model(
        img_size=img_size,
        num_classes=num_classes
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    model.save("microdetect_model.h5")

if __name__ == "__main__":
    main()
