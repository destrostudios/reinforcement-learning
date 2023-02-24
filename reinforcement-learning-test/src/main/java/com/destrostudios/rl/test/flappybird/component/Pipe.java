package com.destrostudios.rl.test.flappybird.component;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import com.destrostudios.rl.test.flappybird.Constant;
import com.destrostudios.rl.test.flappybird.GameUtil;
import lombok.Getter;

public class Pipe {

    public static final int TYPE_TOP_NORMAL = 0;
    public static final int TYPE_BOTTOM_NORMAL = 1;
    public static final int TOP_PIPE_LENGTHENING = 100;
    private static BufferedImage[] images;
    static {
        int PIPE_IMAGE_COUNT = 3;
        images = new BufferedImage[PIPE_IMAGE_COUNT];
        for (int i = 0; i < PIPE_IMAGE_COUNT; i++) {
            images[i] = GameUtil.loadBufferedImage(Constant.PIPE_IMAGE_PATHS[i]);
        }
    }
    public static final int PIPE_WIDTH = images[0].getWidth();
    public static final int PIPE_HEIGHT = images[0].getHeight();
    public static final int PIPE_HEAD_WIDTH = images[1].getWidth();
    public static final int PIPE_HEAD_HEIGHT = images[1].getHeight();

    public Pipe() {
        this.velocity = Constant.GAME_SPEED;
        pipeCollisionRect = new Rectangle();
        pipeCollisionRect.width = PIPE_WIDTH;
    }
    @Getter
    private int x;
    @Getter
    private int y;
    private int height;
    @Getter
    private boolean visible;
    private int type;
    private int velocity;
    @Getter
    private Rectangle pipeCollisionRect;

    public void setAttribute(int x, int y, int height, int type, boolean visible) {
        this.x = x;
        this.y = y;
        this.height = height;
        this.type = type;
        this.visible = visible;
        pipeCollisionRect.x = x + 5;
        pipeCollisionRect.y = y;
        pipeCollisionRect.height = height;
    }

    public void update(Bird bird) {
        x -= velocity;
        pipeCollisionRect.x -= velocity;
        if (x < (-1 * PIPE_HEAD_WIDTH)) {
            visible = false;
        }
    }

    public void draw(Graphics graphics) {
        switch (this.type) {
            case TYPE_TOP_NORMAL:
                drawTopNormal(graphics);
                break;
            case TYPE_BOTTOM_NORMAL:
                drawBottomNormal(graphics);
                break;
        }
        // graphics.setColor(Color.white);
        // graphics.drawRect((int) pipeRect.getX(), (int) pipeRect.getY(), (int) pipeRect.getWidth(), (int) pipeRect.getHeight());
    }

    private void drawTopNormal(Graphics graphics) {
        int count = ((height - PIPE_HEAD_HEIGHT) / PIPE_HEIGHT) + 1;
        for (int i = 0; i < count; i++) {
            graphics.drawImage(images[0], x, y + i * PIPE_HEIGHT, null);
        }
        graphics.drawImage(images[1], x - ((PIPE_HEAD_WIDTH - PIPE_WIDTH) >> 1), height - TOP_PIPE_LENGTHENING - PIPE_HEAD_HEIGHT, null);
    }

    private void drawBottomNormal(Graphics graphics) {
        int count = (height - PIPE_HEAD_HEIGHT - Ground.GROUND_HEIGHT) / PIPE_HEIGHT + 1;
        for (int i = 0; i < count; i++) {
            graphics.drawImage(images[0], x, Constant.FRAME_HEIGHT - PIPE_HEIGHT - Ground.GROUND_HEIGHT - i * PIPE_HEIGHT, null);
        }
        graphics.drawImage(images[2], x - ((PIPE_HEAD_WIDTH - PIPE_WIDTH) >> 1), Constant.FRAME_HEIGHT - height, null);
    }

    public boolean isInFrame() {
        return ((x + PIPE_WIDTH) < Constant.FRAME_WIDTH);
    }

    static class PipePool {
        public static final int FULL_PIPE = (Constant.FRAME_WIDTH / (Pipe.PIPE_HEAD_WIDTH + Pipes.HORIZONTAL_INTERVAL) + 2) * 2;
        public static final int MAX_PIPE_COUNT = 30;

        private static final List<Pipe> pool = new ArrayList<>();
        static {
            for (int i = 0; i < FULL_PIPE; i++) {
                pool.add(new Pipe());
            }
        }

        public static Pipe get() {
            int size = pool.size();
            if (size > 0) {
                return pool.remove(size - 1);
            } else {
                return new Pipe();
            }
        }

        public static void giveBack(Pipe pipe) {
            if (pool.size() < MAX_PIPE_COUNT) {
                pool.add(pipe);
            }
        }
    }
}
