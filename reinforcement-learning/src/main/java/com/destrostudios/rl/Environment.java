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
     * Runs the environment from reset until done.
     *
     * @param agent    the agent to choose the actions with
     * @param training true to run while training. When training, the steps will be recorded
     * @return the replayMemory
     */
    EnvironmentStep[] runEnvironment(Agent agent, boolean training);

    int getEnvironmentStep();

    @Override
    void close();

}
