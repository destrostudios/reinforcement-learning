package com.destrostudios.rl.test.game.component;

import java.awt.*;
import java.util.ArrayList;

import com.destrostudios.rl.test.game.Constant;
import com.destrostudios.rl.test.game.GameUtil;
import com.destrostudios.rl.test.game.component.Pipe.PipePool;
import lombok.Getter;

public class Pipes {

    public static final int VERTICAL_INTERVAL = Constant.FRAME_HEIGHT >> 2;
    public static final int HORIZONTAL_INTERVAL = Constant.FRAME_HEIGHT >> 2;
    public static final int MIN_Y = Constant.FRAME_HEIGHT / 5;
    public static final int MAX_Y = Constant.FRAME_HEIGHT / 3;

    public Pipes() {
        pipes = new ArrayList<>();
    }
    private ArrayList<Pipe> pipes;
    @Getter
    private float scoreReward;

    public void update(Bird bird) {
        updatePipes(bird);
        generatePipesAndCheckDistance(bird);
    }

    private void updatePipes(Bird bird) {
        for (int i = 0; i < pipes.size(); i++) {
            Pipe pipe = pipes.get(i);
            if (pipe.isVisible()) {
                pipe.update(bird);
            } else {
                Pipe remove = pipes.remove(i);
                PipePool.giveBack(remove);
                i--;
            }
        }
    }

    private void generatePipesAndCheckDistance(Bird bird) {
        scoreReward = 0.2f;
        if (pipes.size() == 0) {
            int topHeight = GameUtil.getRandomNumber(MIN_Y, MAX_Y + 1);

            Pipe top = PipePool.get();
            top.setAttribute(Constant.FRAME_WIDTH, -Pipe.TOP_PIPE_LENGTHENING,topHeight + Pipe.TOP_PIPE_LENGTHENING, Pipe.TYPE_TOP_NORMAL, true);

            Pipe bottom = PipePool.get();
            bottom.setAttribute(Constant.FRAME_WIDTH, topHeight + VERTICAL_INTERVAL, Constant.FRAME_HEIGHT - topHeight - VERTICAL_INTERVAL, Pipe.TYPE_BOTTOM_NORMAL, true);

            pipes.add(top);
            pipes.add(bottom);
        } else {
            Pipe lastPipe = pipes.get(pipes.size() - 1);
            int currentDistance = lastPipe.getX() - bird.getX() + Bird.BIRD_WIDTH / 2;
            int SCORE_DISTANCE = (2 * Pipe.PIPE_WIDTH) + HORIZONTAL_INTERVAL;
            if (pipes.size() >= PipePool.FULL_PIPE) {
                if ((currentDistance <= (SCORE_DISTANCE + Pipe.PIPE_WIDTH * 3/2))
                && (currentDistance > (SCORE_DISTANCE + Pipe.PIPE_WIDTH * 3/2 - Constant.GAME_SPEED))) {
                    scoreReward = 0.8f;
                }
                if ((currentDistance <= SCORE_DISTANCE)
                && (currentDistance > (SCORE_DISTANCE - Constant.GAME_SPEED))) {
                    scoreReward = 1;
                }
            }
            if (lastPipe.isInFrame()) {
                addNormalPipe(lastPipe);
            }
        }
    }

    private void addNormalPipe(Pipe lastPipe) {
        int topHeight = GameUtil.getRandomNumber(MIN_Y, MAX_Y + 1);
        int x = lastPipe.getX() + HORIZONTAL_INTERVAL;

        Pipe top = PipePool.get();
        top.setAttribute(x, -Pipe.TOP_PIPE_LENGTHENING, topHeight + Pipe.TOP_PIPE_LENGTHENING, Pipe.TYPE_TOP_NORMAL, true);

        Pipe bottom = PipePool.get();
        bottom.setAttribute(x, topHeight + VERTICAL_INTERVAL, Constant.FRAME_HEIGHT - topHeight - VERTICAL_INTERVAL, Pipe.TYPE_BOTTOM_NORMAL, true);

        pipes.add(top);
        pipes.add(bottom);
    }

    public boolean isCollidingWithPipe(Bird bird) {
        return pipes.stream().anyMatch(pipe -> pipe.getPipeCollisionRect().intersects(bird.getBirdCollisionRect()));
    }

    public void draw(Graphics graphics) {
        for (Pipe pipe : pipes) {
            pipe.draw(graphics);
        }
    }

    public void reset() {
        for (Pipe pipe : pipes) {
            PipePool.giveBack(pipe);
        }
        pipes.clear();
    }
}
