package com.destrostudios.rl.test.game;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.destrostudios.rl.EnvironmentStep;
import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
public class FlappyBirdStep implements EnvironmentStep {

    @Getter
    private final NDManager manager;
    @Getter
    private final NDList preObservation;
    @Getter
    private final NDList postObservation;
    @Getter
    private final NDList action;
    private final float reward;
    @Getter
    private final boolean terminal;

    @Override
    public NDList getPreObservation(NDManager manager) {
        preObservation.attach(manager);
        return preObservation;
    }

    @Override
    public NDList getPostObservation(NDManager manager) {
        postObservation.attach(manager);
        return postObservation;
    }

    @Override
    public NDArray getReward() {
        return manager.create(reward);
    }

    @Override
    public void close() {
        this.manager.close();
    }
}