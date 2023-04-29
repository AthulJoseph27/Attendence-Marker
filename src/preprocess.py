import os
import cv2
import random
import numpy as np
from tqdm import tqdm

dir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "datasets", "_Unknown"
)
frameSize = (224, 224)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotMat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotMat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def translateImage(image, tx, ty):
    translationMatrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    translatedImage = cv2.warpAffine(
        src=image, M=translationMatrix, dsize=image.shape[1::-1]
    )
    return translatedImage


def augmentImageAndSave(filename: str, srcPath: str, dstPath: str) -> None:
    image = cv2.imread(srcPath)
    image = cv2.resize(image, frameSize)

    filename = filename.split(".")[0]
    flippedFace = cv2.flip(image, 1)
    facePath = os.path.join(dstPath, f"{filename}_f.jpg")
    cv2.imwrite(facePath, flippedFace)

    for angle in range(-20, 20, 2):
        if angle == 0:
            continue
        translatedFace = translateImage(
            image, random.randint(-30, 30), random.randint(-30, 30)
        )
        rotatedFace = rotateImage(translatedFace, angle)
        facePath = os.path.join(dstPath, f"{filename}_r{angle}.jpg")
        cv2.imwrite(facePath, rotatedFace)


def main():
    path = dir
    saveDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        "training",
        "unknown",
    )
    os.makedirs(saveDir, exist_ok=True)

    for filename in tqdm(os.listdir(path)):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue
        srcPath = os.path.join(path, filename)
        augmentImageAndSave(filename=filename, srcPath=srcPath, dstPath=saveDir)


if __name__ == "__main__":
    main()
