package com.destrostudios.rl;

import ai.djl.ndarray.NDList;

import java.util.ArrayList;

/**
 * An environment to use for reinforcement learning.
 */
public interface Environment extends AutoCloseable {

    /**
     * Resets the environment to it's default state.
     */
    void reset();

    /**
     * Returns the observation detailing the current state of the environment.
     */
    NDList getObservation();

    /**
     * Returns the current actions that can be taken in the environment.
     */
    ArrayList<NDList> getActionSpace();

    /**
     * Takes a step by performing an action in this environment.
     *
     * @param action   the action to perform
     * @param training true if the step is during training
     */
    void step(NDList action, boolean training);

    /**
     * Runs the environment from reset until done.
     *
     * @param agent    the agent to choose the actions with
     * @param training true to run while training. When training, the steps will be recorded
     * @return the replayMemory
     */
    EnvironmentStep[] runEnvironment(Agent agent, boolean training);

    /**
     * Returns a batch of steps from the environment {@link ai.djl.modality.rl.ReplayBuffer}.
     */
    EnvironmentStep[] getBatch();

    int getEnvironmentStep();

    @Override
    void close();

}
