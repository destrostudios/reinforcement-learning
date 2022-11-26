package com.destrostudios.rl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class Replay implements AutoCloseable {

    private NDManager manager;
    private NDList preObservation;
    private NDList action;
    private NDList postObservation;
    private NDArray reward;
    private boolean terminated;

    @Override
    public void close() {
        manager.close();
    }
}
