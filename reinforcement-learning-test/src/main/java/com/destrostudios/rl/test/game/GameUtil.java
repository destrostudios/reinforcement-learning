package com.destrostudios.rl.test.game;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

public class GameUtil {

    public static BufferedImage loadBufferedImage(String imagePath) {
        try {
            return ImageIO.read(new FileInputStream(imagePath));
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return null;
    }

    public static int getRandomNumber(int min, int max) {
        return (int) ((Math.random() * (max - min)) + min);
    }

    public static NDArray preprocessImage(BufferedImage image, int width, int height) {
        return NDImageUtils.toTensor(NDImageUtils.resize(ImageFactory.getInstance().fromImage(image).toNDArray(NDManager.newBaseManager(), Image.Flag.GRAYSCALE), width, height));
    }
}
