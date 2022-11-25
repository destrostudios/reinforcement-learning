package com.destrostudios.rl;

import ai.djl.ndarray.NDList;

import java.util.ArrayList;

public interface Environment {

    ArrayList<NDList> getActionSpace();

    Outcome takeAction(NDList action);

    NDList getCurrentObservation();
}
