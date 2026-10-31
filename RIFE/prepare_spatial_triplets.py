import os
import shutil
import argparse
import random
from pathlib import Path


def prepare_triplets(
    input_dir,
    output_dir,
    image_ext=".png",
    train_ratio=0.8,
    seed=1234,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    seq_dir = output_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob(f"*{image_ext}"))

    if len(image_files) < 3:
        raise ValueError(
            f" requires at least  3  observed slices are required to construct  triplet,  currently only  {len(image_files)}  images ."
        )

    triplet_names = []

    for i in range(len(image_files) - 2):
        name = f"{i:06d}"
        out_dir = seq_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(image_files[i], out_dir / "im1.png")
        shutil.copy(image_files[i + 1], out_dir / "im2.png")
        shutil.copy(image_files[i + 2], out_dir / "im3.png")

        triplet_names.append(name)

        print(
            f"{name}: "
            f"im1={image_files[i].name}, "
            f"im2={image_files[i + 1].name}, "
            f"im3={image_files[i + 2].name}"
        )

    random.seed(seed)
    random.shuffle(triplet_names)

    n_train = max(1, int(len(triplet_names) * train_ratio))

    
    if len(triplet_names) >= 2:
        n_train = min(n_train, len(triplet_names) - 1)

    train_names = triplet_names[:n_train]
    val_names = triplet_names[n_train:]

    if len(val_names) == 0:
        val_names = train_names

    with open(output_dir / "trainlist.txt", "w") as f:
        for name in train_names:
            f.write(name + "\n")

    with open(output_dir / "vallist.txt", "w") as f:
        for name in val_names:
            f.write(name + "\n")

    print("\nDone.")
    print(f"Total slices: {len(image_files)}")
    print(f"Total triplets: {len(triplet_names)}")
    print(f"Train triplets: {len(train_names)}")
    print(f"Val triplets: {len(val_names)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-ext", type=str, default=".png")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    prepare_triplets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_ext=args.image_ext,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )