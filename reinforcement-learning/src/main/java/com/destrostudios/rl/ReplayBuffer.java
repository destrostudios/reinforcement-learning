package com.destrostudios.rl;

public interface ReplayBuffer {

    void addStep(EnvironmentStep step);

    void cleanupInterval();

    EnvironmentStep[] getTrainingBatch();
}
