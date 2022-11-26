package com.destrostudios.rl.test.game;

import com.destrostudios.rl.test.game.component.Ground;
import com.destrostudios.rl.test.game.component.Bird;
import com.destrostudios.rl.test.game.component.Pipes;
import com.destrostudios.rl.util.NDContinuousArray;
import com.destrostudios.rl.Environment;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;

public class FlappyBird implements Environment {

    private static final Logger logger = LoggerFactory.getLogger(FlappyBird.class);

    private static final int[] DO_NOTHING = {1, 0};
    private static final int[] FLAP = {0, 1};

    public FlappyBird() {
        ground = new Ground();
        bird = new Bird(this);
        pipes = new Pipes();
        image = new BufferedImage(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT, BufferedImage.TYPE_4BYTE_ABGR);
        continuousObservation = new NDContinuousArray(Constant.OBSERVATION_CONTINUOUS_LENGTH);
        updateObservation();
    }
    private Ground ground;
    private Bird bird;
    private Pipes pipes;
    private int score;
    @Getter
    private ArrayList<NDList> actionSpace;
    @Getter
    private BufferedImage image;
    private NDContinuousArray continuousObservation;
    @Getter
    private NDList observation;
    @Getter
    private boolean terminated;

    @Override
    public void initialize(NDManager manager) {
        actionSpace = new ArrayList<>();
        actionSpace.add(new NDList(manager.create(DO_NOTHING)));
        actionSpace.add(new NDList(manager.create(FLAP)));
    }

    @Override
    public float takeAction(NDList action) {
        if (terminated) {
            restart();
        }

        boolean isFlapAction = (action.singletonOrThrow().getInt(1) == 1);
        if (isFlapAction) {
            bird.flap();
        }

        bird.update();
        ground.update();
        pipes.update(bird);

        float reward;
        if (bird.isOutOfBounds() || pipes.isCollidingWithPipe(bird)) {
            reward = -1;
            terminated = true;
        } else if (bird.isBelowOrAbovePipeHoles()) {
            reward = 0.1f;
        } else {
            reward = pipes.getScoreReward();
            if (reward == 1) {
                score++;
            }
        }

        drawImage();

        updateObservation();

        logger.info("ACTION " + Arrays.toString(action.singletonOrThrow().toArray()) + " / REWARD " + reward + " / SCORE " + score);

        return reward;
    }

    private void updateObservation() {
        NDArray currentObservation = GameUtil.preprocessImage(image, Constant.OBSERVATION_WIDTH, Constant.OBSERVATION_HEIGHT);
        observation = continuousObservation.push(currentObservation);
    }

    private void restart() {
        bird.reset();
        pipes.reset();
        score = 0;
        terminated = false;
    }

    private void drawImage() {
        Graphics graphics = image.getGraphics();
        graphics.setColor(Color.BLACK);
        graphics.fillRect(0, 0, Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        ground.draw(graphics);
        bird.draw(graphics);
        pipes.draw(graphics);
    }
}
