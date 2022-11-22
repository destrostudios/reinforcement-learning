package com.destrostudios.rl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/**
 * A record of taking a step in the environment.
 */
public interface EnvironmentStep extends AutoCloseable {

    /**
     * Returns the observation before the action which attach to manager.
     */
    NDList getPreObservation(NDManager manager);

    /**
     * Returns the observation before the action.
     */
    NDList getPreObservation();

    /**
     * Returns the action taken.
     */
    NDList getAction();

    /**
     * Returns the observation after the action which attach to manager.
     */
    NDList getPostObservation(NDManager manager);

    /**
     * Return the observation after the action.
     */
    NDList getPostObservation();

    /**
     * Returns the manager which manage this step.
     */
    NDManager getManager();


    /**
     * Returns the reward given for the action.
     */
    NDArray getReward();

    /**
     * Returns whether the environment is finished or can accept further actions.
     */
    boolean isTerminal();

    @Override
    void close();
}