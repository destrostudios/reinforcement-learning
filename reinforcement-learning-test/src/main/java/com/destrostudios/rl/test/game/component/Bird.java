package com.destrostudios.rl.test.game.component;

import java.awt.*;
import java.awt.image.BufferedImage;

import com.destrostudios.rl.test.game.Constant;
import com.destrostudios.rl.test.game.FlappyBird;
import com.destrostudios.rl.test.game.GameUtil;
import lombok.Getter;

public class Bird {

    private static final int RECT_DESCALE = 2;
    private static final int ACC_FLAP = 15; // players speed on flapping
    private static final double ACC_Y = 4; // players downward acceleration
    private static final int MAX_VEL_Y = -25; // max vel along Y, max descend speed
    private static BufferedImage birdImages;
    public static final int BIRD_WIDTH;
    public static final int BIRD_HEIGHT;
    static {
        birdImages = GameUtil.loadBufferedImage(Constant.BIRDS_IMAGE_PATH);
        BIRD_WIDTH = birdImages.getWidth();
        BIRD_HEIGHT = birdImages.getHeight();
    }
    public static final int BOTTOM_BOUNDARY = Constant.FRAME_HEIGHT - Ground.GROUND_HEIGHT - (BIRD_HEIGHT >> 1);

    public Bird(FlappyBird game) {
        this.game = game;

        x = Constant.FRAME_WIDTH >> 2;
        y = Constant.FRAME_HEIGHT >> 1;

        int rectX = x - (BIRD_WIDTH >> 1);
        int rectY = y - (BIRD_HEIGHT >> 1) + RECT_DESCALE * 2;
        birdCollisionRect = new Rectangle(
            rectX + RECT_DESCALE,
            rectY + RECT_DESCALE * 2,
            BIRD_WIDTH - RECT_DESCALE * 3,
            BIRD_HEIGHT - RECT_DESCALE * 4
        );
    }
    private FlappyBird game;
    @Getter
    private int x;
    private int y;
    @Getter
    private Rectangle birdCollisionRect;
    @Getter
    private int velocity = 0; // Bird's velocity along Y, default same as playerFlapped

    public void update() {
        if (velocity > MAX_VEL_Y) {
            velocity -= ACC_Y;
        }
        y = Math.min((y - velocity), BOTTOM_BOUNDARY);
        birdCollisionRect.y = birdCollisionRect.y - velocity;
    }

    public boolean isBelowOrAbovePipeHoles() {
        return ((birdCollisionRect.y < (Pipes.MIN_Y - Pipes.VERTICAL_INTERVAL)) || (birdCollisionRect.y > Pipes.MAX_Y));
    }

    public boolean isOutOfBounds() {
        return (birdCollisionRect.y < Constant.WINDOW_BAR_HEIGHT) || (birdCollisionRect.y >= (BOTTOM_BOUNDARY - 10));
    }

    public void flap() {
        velocity = ACC_FLAP;
    }

    public void reset() {
        y = Constant.FRAME_HEIGHT >> 1;
        velocity = 0;
        birdCollisionRect.y = y + (RECT_DESCALE * 4) - (birdImages.getHeight() / 2);
    }

    public void draw(Graphics graphics) {
        graphics.drawImage(birdImages, x - (BIRD_WIDTH >> 1), y - (BIRD_HEIGHT >> 1), null);
        // graphics.setColor(Color.white);
        // graphics.drawRect((int) birdCollisionRect.getX(), (int)birdCollisionRect.getY(), (int) birdCollisionRect.getWidth(), (int) birdCollisionRect.getHeight());
    }
}
