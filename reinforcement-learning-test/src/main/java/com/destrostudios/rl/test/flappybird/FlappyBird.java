package com.destrostudios.rl.test.flappybird;

import com.destrostudios.rl.test.flappybird.component.Ground;
import com.destrostudios.rl.test.flappybird.component.Bird;
import com.destrostudios.rl.test.flappybird.component.Pipe;
import com.destrostudios.rl.test.flappybird.component.Pipes;
import lombok.Getter;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.action.space.IntegerActionSpace;
import org.nd4j.linalg.api.rng.Random;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

public class FlappyBird implements Environment<IntegerAction> {

    public FlappyBird(Random random) {
        ground = new Ground();
        bird = new Bird(this);
        pipes = new Pipes();
        image = new BufferedImage(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT, BufferedImage.TYPE_4BYTE_ABGR);
        actionSpace = new IntegerActionSpace(ACTIONS_COUNT, DO_NOTHING, random);
    }
    public static final int ACTIONS_COUNT = 2;
    private static final int DO_NOTHING = 0;
    private static final int FLAP = 1;
    @Getter
    private IntegerActionSpace actionSpace;
    private Ground ground;
    private Bird bird;
    private Pipes pipes;
    @Getter
    private int score;
    @Getter
    private int highscore;
    private int games;
    private int sum;
    @Getter
    private BufferedImage image;
    @Getter
    private boolean episodeFinished;

    @Override
    public Map<String, Object> reset() {
        bird.reset();
        pipes.reset();
        score = 0;
        episodeFinished = false;
        return getChannelsData();
    }

    @Override
    public StepResult step(IntegerAction action) {
        boolean isFlapAction = (action.toInteger() == FLAP);
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
            episodeFinished = true;
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

        // System.out.println("ACTION " + action + " / REWARD " + reward + " / SCORE " + score + " / HIGHSCORE " + highscore);

        /*try {
            Thread.sleep(1000 / 60);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }*/

        return new StepResult(getChannelsData(), reward, isEpisodeFinished());
    }

    private Map<String, Object> getChannelsData() {
        HashMap<String, Object> channelsData = new HashMap<>();
        channelsData.put("mydata", new double[] {
            bird.getBirdCollisionRect().getY(),
            bird.getVelocity(),
            pipes.getNextPipeX(),
            pipes.getNextPipeHoleY(),
            pipes.getPreviousPipeX(),
            pipes.getPreviousPipeHoleY(),
        });
        return channelsData;
    }

    @Override
    public void close() {
        // Not needed
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
