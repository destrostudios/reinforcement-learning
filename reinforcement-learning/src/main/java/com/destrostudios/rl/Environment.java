package com.destrostudios.rl;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.util.ArrayList;

public interface Environment {

    void initialize(NDManager manager);

    ArrayList<NDList> getActionSpace();

    float takeAction(NDList action);

    NDList getCurrentObservation();

    boolean isTerminated();
}
