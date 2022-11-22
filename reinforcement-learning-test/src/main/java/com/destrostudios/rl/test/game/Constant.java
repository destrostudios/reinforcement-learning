package com.destrostudios.rl.test.game;

import java.awt.Color;

public class Constant {

    public static final int FRAME_WIDTH = 288;
    public static final int FRAME_HEIGHT = 512;

    public static final String GAME_TITLE = "RL Flappy Bird";

    public static final int FRAME_X = 0;
    public static final int FRAME_Y = 0;

    public static final int GAME_SPEED = 6;

    public static final Color BG_COLOR = new Color(0x000000);

    public static final int FPS = 1000 / 30;

    public static final int WINDOW_BAR_HEIGHT = 30;

    public static final int[] DO_NOTHING = {1, 0};
    public static final int[] FLAP = {0, 1};

    public static final String RESOURCE_PATH = "./reinforcement-learning-test/src/main/resources";
    public static final String BG_IMG_PATH = RESOURCE_PATH + "/img/background.png";
    public static final String BIRDS_IMG_PATH = RESOURCE_PATH + "/img/0.png";
    public static final String[] PIPE_IMG_PATH = {
        RESOURCE_PATH + "/img/pipe.png",
        RESOURCE_PATH + "/img/pipe_top.png",
        RESOURCE_PATH + "/img/pipe_bottom.png"
    };
}
