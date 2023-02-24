package com.destrostudios.rl.test.flappybird.component;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import com.destrostudios.rl.test.flappybird.Constant;
import com.destrostudios.rl.test.flappybird.GameUtil;
import com.destrostudios.rl.test.flappybird.component.Pipe.PipePool;
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
    private Pipe previousPipe;
    private Pipe nextPipe;

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
        scoreReward = 0;
        if (pipes.size() == 0) {
            int topHeight = GameUtil.getRandomNumber(MIN_Y, MAX_Y + 1);

            Pipe top = PipePool.get();
            top.setAttribute(Constant.FRAME_WIDTH, -Pipe.TOP_PIPE_LENGTHENING,topHeight + Pipe.TOP_PIPE_LENGTHENING, Pipe.TYPE_TOP_NORMAL, true);

            Pipe bottom = PipePool.get();
            bottom.setAttribute(Constant.FRAME_WIDTH, topHeight + VERTICAL_INTERVAL, Constant.FRAME_HEIGHT - topHeight - VERTICAL_INTERVAL, Pipe.TYPE_BOTTOM_NORMAL, true);

            pipes.add(top);
            pipes.add(bottom);
        } else {
            Pipe tmpNextPipe = nextPipe;
            previousPipe = getPreviousPipe(bird);
            nextPipe = getNextPipe(bird);
            if ((tmpNextPipe != null) && (nextPipe != tmpNextPipe)) {
                scoreReward = 10;
            }
            Pipe lastPipe = pipes.get(pipes.size() - 1);
            if (lastPipe.isInFrame()) {
                addNormalPipe(lastPipe);
            }
        }
    }

    public int getNextPipeX() {
        return (nextPipe != null ? nextPipe.getX() : -100);
    }

    public int getNextPipeHoleY() {
        return (nextPipe != null ? nextPipe.getY() - (VERTICAL_INTERVAL / 2) : (Constant.FRAME_HEIGHT / 2));
    }

    public int getPreviousPipeX() {
        return (previousPipe != null ? previousPipe.getX() : -100);
    }

    public int getPreviousPipeHoleY() {
        return (previousPipe != null ? previousPipe.getY() - (VERTICAL_INTERVAL / 2) : (Constant.FRAME_HEIGHT / 2));
    }

    private Pipe getNextPipe(Bird bird) {
        return pipes.stream().filter(pipe -> pipe.getX() > bird.getBirdCollisionRect().getX()).skip(1).findFirst().orElse(null);
    }

    private Pipe getPreviousPipe(Bird bird) {
        List<Pipe> previousPipes = pipes.stream()
                .filter(pipe -> pipe.getX() <= bird.getBirdCollisionRect().getX())
                .collect(Collectors.toList());
        return ((previousPipes.size() > 0) ? previousPipes.get(1) : null);
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
        previousPipe = null;
        nextPipe = null;
    }
}
