package com.destrostudios.rl.test.game;

import com.destrostudios.rl.test.game.component.Ground;
import com.destrostudios.rl.test.game.component.Bird;
import com.destrostudios.rl.test.game.component.GameElementLayer;
import com.destrostudios.rl.util.NDContinuousArray;
import com.destrostudios.rl.Environment;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;

public class FlappyBird implements Environment {

    private static final Logger logger = LoggerFactory.getLogger(FlappyBird.class);

    public static final int GAME_START = 1;
    public static final int GAME_OVER = 2;
    private static final int[] DO_NOTHING = {1, 0};
    private static final int[] FLAP = {0, 1};

    public FlappyBird() {
        gameState = GAME_START;
        ground = new Ground();
        bird = new Bird(this);
        gameElement = new GameElementLayer();

        image = new BufferedImage(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT, BufferedImage.TYPE_4BYTE_ABGR);
        continuousObservation = new NDContinuousArray(Constant.OBSERVATION_CONTINUOUS_LENGTH);
        updateObservation();
    }
    @Setter
    private int gameState;
    private int score;
    private Ground ground;
    private Bird bird;
    private GameElementLayer gameElement;
    @Getter
    private ArrayList<NDList> actionSpace;
    @Getter
    private BufferedImage image;
    private NDContinuousArray continuousObservation;
    @Setter
    private float reward;
    @Setter
    @Getter
    private boolean terminated;
    @Getter
    private NDList observation;

    @Override
    public void initialize(NDManager manager) {
        actionSpace = new ArrayList<>();
        actionSpace.add(new NDList(manager.create(DO_NOTHING)));
        actionSpace.add(new NDList(manager.create(FLAP)));
    }

    @Override
    public float takeAction(NDList action) {
        terminated = false;
        reward = 0.2f;

        // action[0] == 1 : do nothing
        // action[1] == 1 : flap the bird
        if ((!bird.isDead()) && (action.singletonOrThrow().getInt(1) == 1)) {
            bird.birdFlap();
        }

        bird.update();
        ground.update(bird);
        gameElement.update(bird);

        drawImage();

        updateObservation();

        logger.info("ACTION " + Arrays.toString(action.singletonOrThrow().toArray()) + " / REWARD " + reward + " / SCORE " + score);

        if (gameState == GAME_OVER) {
            restartGame();
        }

        return reward;
    }

    private void updateObservation() {
        NDArray currentObservation = GameUtil.preprocessImage(image, Constant.OBSERVATION_WIDTH, Constant.OBSERVATION_HEIGHT);
        observation = continuousObservation.push(currentObservation);
    }

    public void score() {
        score += 1;
        reward = 1;
    }

    private void restartGame() {
        gameState = GAME_START;
        score = 0;
        gameElement.reset();
        bird.reset();
    }

    private void drawImage() {
        Graphics graphics = image.getGraphics();
        graphics.setColor(Color.BLACK);
        graphics.fillRect(0, 0, Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        ground.draw(graphics);
        bird.draw(graphics);
        gameElement.draw(graphics);
    }
}
