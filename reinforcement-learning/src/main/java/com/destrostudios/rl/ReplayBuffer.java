package com.destrostudios.rl;

/**
 * Records {@link EnvironmentStep}s so that they can be trained on.
 * Using a replay buffer ensures that a variety of states are trained on for every training batch making the training more stable.
 */
public interface ReplayBuffer {

    /**
     * Returns a batch of steps from this buffer.
     */
    EnvironmentStep[] getBatch();

    /**
     * Close the step not pointed to.
     */
    void closeStep();

    /**
     * Adds a new step to the buffer.
     */
    void addStep(EnvironmentStep step);
}
