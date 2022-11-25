package com.destrostudios.rl;

public interface ReplayBuffer {

    void addReplay(Replay replay);

    void cleanupInterval();

    Replay[] getTrainingBatch();
}
