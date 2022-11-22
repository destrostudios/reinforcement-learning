package com.destrostudios.rl;

import ai.djl.ndarray.NDList;

/**
 * An is the model or technique to decide the actions to take in an {@link Environment}.
 */
public interface Agent {

    /**
     * Chooses the next action to take within the {@link Environment}.
     */
    NDList chooseAction(Environment environment, boolean isTraining);

    /**
     * Trains this agent on a batch of {@link EnvironmentStep}s.
     */
    void trainBatch(EnvironmentStep[] batchSteps);
}
