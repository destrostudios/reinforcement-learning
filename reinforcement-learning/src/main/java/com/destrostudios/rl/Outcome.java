package com.destrostudios.rl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.Getter;

public class Outcome implements AutoCloseable {

    public Outcome(NDManager manager, NDList preObservation, NDList action, NDList postObservation, float reward, boolean isTerminal) {
        this.manager = manager;
        this.preObservation = preObservation;
        this.action = action;
        this.postObservation = postObservation;
        this.reward = manager.create(reward);
        this.terminal = isTerminal;
    }
    @Getter
    private final NDManager manager;
    @Getter
    private final NDList preObservation;
    @Getter
    private final NDList action;
    @Getter
    private final NDList postObservation;
    @Getter
    private final NDArray reward;
    @Getter
    private final boolean terminal;

    @Override
    public void close() {
        this.manager.close();
    }
}
