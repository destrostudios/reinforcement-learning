package com.destrostudios.rl.test.flappybird;

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
}
