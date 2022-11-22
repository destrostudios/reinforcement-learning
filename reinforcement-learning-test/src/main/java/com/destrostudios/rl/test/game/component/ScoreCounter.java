package com.destrostudios.rl.test.game.component;

import com.destrostudios.rl.test.game.FlappyBird;
import lombok.Getter;

public class ScoreCounter {

    public ScoreCounter(FlappyBird game) {
        this.game = game;
    }
    private FlappyBird game;
    @Getter
    private long score = 0;

    public void score(Bird bird) {
        if (!bird.isDead()) {
            game.setCurrentReward(1f);
            score += 1;
        }
    }

    public void reset() {
        score = 0;
    }
}
