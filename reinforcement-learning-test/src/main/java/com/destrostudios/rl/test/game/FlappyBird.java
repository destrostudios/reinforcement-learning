package com.destrostudios.rl.test.game;

import ai.djl.modality.rl.ActionSpace;
import com.destrostudios.rl.test.game.component.Ground;
import com.destrostudios.rl.test.game.component.Bird;
import com.destrostudios.rl.test.game.component.Pipe;
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

public class FlappyBird implements Environment<float[]> {

    private static final Logger logger = LoggerFactory.getLogger(FlappyBird.class);

    private static final int[] DO_NOTHING = {-1, 1};
    private static final int[] FLAP = {1, -1};

    public FlappyBird() {
        ground = new Ground();
        bird = new Bird(this);
        pipes = new Pipes();
        image = new BufferedImage(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT, BufferedImage.TYPE_4BYTE_ABGR);
        continuousObservation = new NDContinuousArray(Constant.OBSERVATION_CONTINUOUS_LENGTH);
    }
    private Ground ground;
    private Bird bird;
    private Pipes pipes;
    private int score;
    private int highscore;
    private int games;
    private int sum;
    @Getter
    private ActionSpace actionSpace;
    @Getter
    private BufferedImage image;
    private NDContinuousArray continuousObservation;
    @Getter
    private float[] observation;
    @Getter
    private boolean terminated;
    private NDManager manager;

    @Override
    public void initialize(NDManager baseManager) {
        manager = baseManager.newSubManager();

        actionSpace = new ActionSpace();
        actionSpace.add(new NDList(manager.create(DO_NOTHING)));
        actionSpace.add(new NDList(manager.create(FLAP)));

        updateObservation();
    }

    @Override
    public void reset() {
        bird.reset();
        pipes.reset();
        score = 0;
        terminated = false;
    }

    @Override
    public float takeAction(NDList action) {
        boolean isFlapAction = (action.get(0).toIntArray()[1] == -1);
        if (isFlapAction) {
            bird.flap();
        }

        int oldDistance = (int) Math.abs(pipes.getNextPipeHoleY() - bird.getBirdCollisionRect().getY());

        bird.update();
        ground.update();
        pipes.update(bird);

        float reward;
        if (bird.isOutOfBounds() || pipes.isCollidingWithPipe(bird)) {
            reward = -10;
            terminated = true;
            games++;
            sum += score;
            int interval = 100;
            if ((games % interval) == 0) {
                double average = (((double) sum) / interval);
                System.out.println(games + " games, highscore " + highscore + ", last average: " + average);
                sum = 0;
            }
        } else {
            reward = pipes.getScoreReward();
            if (reward == 10) {
                score++;
                if (score > highscore) {
                    highscore = score;
                }
            } else {
                int newDistance = (int) Math.abs(pipes.getNextPipeHoleY() - bird.getBirdCollisionRect().getY());
                reward += ((oldDistance - newDistance) / 10f);
                reward *= 1;
            }
        }

        drawImage();

        updateObservation();

        // logger.info("ACTION " + Arrays.toString(action.singletonOrThrow().toArray()) + " / REWARD " + reward + " / SCORE " + score + " / HIGHSCORE " + highscore);

        /*try {
            Thread.sleep(1000 / 60);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }*/

        return reward;
    }

    private void updateObservation() {
        observation = new float[] {
            (float) bird.getBirdCollisionRect().getY(),
            bird.getVelocity(),
            pipes.getNextPipeX(),
            pipes.getNextPipeHoleY(),
            pipes.getPreviousPipeX(),
            pipes.getPreviousPipeHoleY(),
        };
        //NDArray currentObservation = GameUtil.preprocessImage(image, Constant.OBSERVATION_WIDTH, Constant.OBSERVATION_HEIGHT);
        //observation = continuousObservation.push(currentObservation);
    }

    @Override
    public NDList mapObservation(NDManager manager, float[] observation) {
        return new NDList(manager.create(observation));
    }

    private void drawImage() {
        Graphics graphics = image.getGraphics();
        graphics.setColor(Color.BLACK);
        graphics.fillRect(0, 0, Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        ground.draw(graphics);
        bird.draw(graphics);
        pipes.draw(graphics);
        if (false) {
            graphics.setColor(new Color(1, 1, 1, 0.5f));
            graphics.fillRect(pipes.getNextPipeX(), 0, Pipe.PIPE_WIDTH, Constant.FRAME_HEIGHT);
            graphics.setColor(new Color(1, 0, 0, 0.5f));
            graphics.fillRect(bird.getBirdCollisionRect().x, bird.getBirdCollisionRect().y, bird.getBirdCollisionRect().width, bird.getBirdCollisionRect().height);
            graphics.setColor(new Color(0, 0, 1, 0.5f));
            graphics.fillRect(pipes.getNextPipeX(), pipes.getNextPipeHoleY() - (Pipes.VERTICAL_INTERVAL / 2), Pipe.PIPE_WIDTH, Pipes.VERTICAL_INTERVAL);
            graphics.setColor(new Color(1, 1, 0, 0.5f));
            graphics.fillRect(pipes.getPreviousPipeX(), 0, Pipe.PIPE_WIDTH, Constant.FRAME_HEIGHT);
            graphics.setColor(new Color(0, 1, 1, 0.5f));
            graphics.fillRect(pipes.getPreviousPipeX(), pipes.getPreviousPipeHoleY() - (Pipes.VERTICAL_INTERVAL / 2), Pipe.PIPE_WIDTH, Pipes.VERTICAL_INTERVAL);
        }
    }
}
