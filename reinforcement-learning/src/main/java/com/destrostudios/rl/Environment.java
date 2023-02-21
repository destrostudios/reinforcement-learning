package com.destrostudios.rl;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

public interface Environment<Observation> {

    void initialize(NDManager baseManager);

    void reset();

    ActionSpace getActionSpace();

    float takeAction(NDList action);

    Observation getObservation();

    NDList mapObservation(NDManager manager, Observation observation);

    boolean isTerminated();
}
