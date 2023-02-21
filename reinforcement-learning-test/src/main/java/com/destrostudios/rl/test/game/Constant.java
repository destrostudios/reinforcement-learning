package com.destrostudios.rl.test.game;

import ai.djl.ndarray.types.Shape;

public class Constant {

    public static final String GAME_TITLE = "RL Flappy Bird";

    public static final int FRAME_X = 0;
    public static final int FRAME_Y = 0;
    public static final int FRAME_WIDTH = 288;
    public static final int FRAME_HEIGHT = 512;
    public static final int WINDOW_BAR_HEIGHT = 30;

    public static final int FPS = 1000 / 60;
    public static final int GAME_SPEED = 6;

    public static final int OBSERVATION_CONTINUOUS_LENGTH = 4;
    public static final int OBSERVATION_WIDTH = 80;
    public static final int OBSERVATION_HEIGHT = 80;
    public static final Shape[] SHAPES = new Shape[] {
        new Shape(2, 6),
        new Shape(2, 2)
    };

    public static final String RESOURCE_PATH = "./reinforcement-learning-test/src/main/resources";
    public static final String BACKGROUND_IMAGE_PATH = RESOURCE_PATH + "/img/background.png";
    public static final String BIRDS_IMAGE_PATH = RESOURCE_PATH + "/img/0.png";
    public static final String[] PIPE_IMAGE_PATHS = {
        RESOURCE_PATH + "/img/pipe.png",
        RESOURCE_PATH + "/img/pipe_top.png",
        RESOURCE_PATH + "/img/pipe_bottom.png"
    };
}
